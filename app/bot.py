import asyncio
import json
import logging
import os
import re
import string
import sys
import time
from datetime import UTC, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import a2s
import discord
import docker
import requests
from discord.ext import commands, tasks
from discord.ui import Button, View, Modal, TextInput
from dotenv import load_dotenv
from dotty_dict import dotty

# ------------------------------
# Type definitions
# ------------------------------


class ContainerConfig(TypedDict):
    alias: str
    name: str
    restart_allowed: bool
    description: str
    port: str
    query_port: str
    a2s_enabled: bool


class MessageState(TypedDict, total=False):
    channel_id: int
    message_id: int


StatsDict = Dict[str, Any]

# ------------------------------
# Logging configuration
# ------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
if os.getenv("DEBUG", "0") == "1":
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled")

# ------------------------------
# Dotty-enabled template rendering
# ------------------------------


class DottyFormatter(string.Formatter):
    """
    Enables dot-notation placeholders in str.format-style templates:
      - {container.alias}, {stats.cpu}, {external.ip}, {a2s.info.server_name}, etc.
    """

    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__()
        self._dot = dotty(context)

    def get_field(self, field_name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        try:
            return self._dot[field_name], field_name
        except Exception:
            return super().get_field(field_name, args, kwargs)


def render_template(template: str, context: Dict[str, Any]) -> str:
    return DottyFormatter(context).vformat(template, args=(), kwargs=context)


# ------------------------------
# Configuration
# ------------------------------


class AppConfig:
    """
    Handles environment loading, container parsing, and templates.
    """

    def __init__(self, config_dir: str = "/config") -> None:
        self.config_dir: str = config_dir

        # Load .env
        dotenv_path: str = os.path.join(self.config_dir, ".env")
        logger.debug("Loading environment variables from %s", dotenv_path)
        load_dotenv(dotenv_path)

        # Required env vars
        self.token: str = self._load_required("DISCORD_TOKEN")

        channel_id_raw: str = self._load_required("DISCORD_CHANNEL_ID")
        try:
            self.channel_id: int = int(channel_id_raw)
        except ValueError:
            logger.critical("DISCORD_CHANNEL_ID must be an integer. Exiting.")
            sys.exit(1)

        # Containers
        self.containers: List[ContainerConfig] = self._load_containers()

        # Optional settings
        self.allowed_users_raw: str = os.getenv("RESTART_ALLOWED_USERS", "")
        self.embed_title: str = os.getenv("EMBED_TITLE", "Docker Status")
        self.embed_color_raw: str = os.getenv("EMBED_COLOR", "0x3498DB")
        self.message_state_file: str = os.path.join(
            self.config_dir, os.getenv("MESSAGE_STATE_FILE", "message_state.json")
        )
        self.restart_rate_limit_count: int = int(os.getenv("RESTART_RATE_LIMIT_COUNT", "2"))
        self.restart_rate_limit_period: float = float(os.getenv("RESTART_RATE_LIMIT_PERIOD", "300"))

        # A2S timeout
        self.a2s_timeout: float = float(os.getenv("A2S_TIMEOUT", "3.0"))

        # Templates
        self.field_template: str = os.getenv(
            "CONTAINER_FIELD_TEMPLATE",
            (
                "**Description:** {description}\n"
                "**Status:** {status}\n"
                "**CPU:** {cpu:.2f}%\n"
                "**RAM:** {ram} ({ram_percent:.2f}%)\n"
                "**Disk:** {disk}\n"
                "**Port:** {port}\n"
                "**Uptime:** {uptime}\n"
                "**Host IP:** {external_ip}"
            ),
        ).replace("\\n", "\n")

        self.field_name_template: str = os.getenv(
            "CONTAINER_FIELD_NAME_TEMPLATE",
            "{alias} (`{name}`)",
        ).replace("\\n", "\n")

        logger.debug("RESTART_ALLOWED_USERS: %s", self.allowed_users_raw)
        logger.debug("EMBED_TITLE: %s", self.embed_title)
        logger.debug("FIELD_TEMPLATE: %s", self.field_template)
        logger.debug("FIELD_NAME_TEMPLATE: %s", self.field_name_template)
        logger.debug("EMBED_COLOR: %s", self.embed_color_raw)
        logger.debug("MESSAGE_STATE_FILE: %s", self.message_state_file)

        self.embed_color: int = self._parse_embed_color(self.embed_color_raw)
        self.allowed_users: set[int] = self._parse_allowed_users(self.allowed_users_raw)

        self._validate_templates()

        # Cache whether templates reference A2S placeholders (used for warnings)
        self.template_uses_a2s_placeholders: bool = self._templates_use_namespace("a2s")
        if self.template_uses_a2s_placeholders:
            disabled_aliases = [c["alias"] for c in self.containers if not c.get("a2s_enabled", True)]
            if disabled_aliases:
                logger.warning(
                    "A2S placeholders are used in templates, but A2S is disabled for: %s",
                    ", ".join(disabled_aliases),
                )

    @staticmethod
    def _load_required(key: str) -> str:
        value = os.getenv(key)
        if not value:
            logger.critical("%s is not set. Exiting.", key)
            sys.exit(1)
        return value

    @staticmethod
    def _load_containers() -> List[ContainerConfig]:
        container_pattern: re.Pattern[str] = re.compile(r"^CONTAINER_(\d+)$")
        container_env_entries: List[Tuple[str, str]] = [
            (env_name, env_value)
            for env_name, env_value in os.environ.items()
            if container_pattern.match(env_name)
        ]

        logger.debug(
            "Detected container environment variables: %s",
            [env_name for env_name, _ in container_env_entries],
        )

        if not container_env_entries:
            logger.critical(
                'No containers configured. Define environment variables like "CONTAINER_1, CONTAINER_2, etc. Exiting."'
            )
            sys.exit(1)

        container_env_entries.sort(
            key=lambda entry: int(container_pattern.match(entry[0]).group(1))  # type: ignore[call-arg]
        )

        containers: List[ContainerConfig] = []
        for env_var_name, env_var_value in container_env_entries:
            container_parts: List[str] = env_var_value.split(":", 6)
            if len(container_parts) < 7:
                logger.critical(
                    "Container entry '%s' configuration not properly formatted. Exiting.",
                    env_var_name,
                )
                sys.exit(1)
         
            container_config: ContainerConfig = {
                "alias": container_parts[0].strip(),
                "name": container_parts[1].strip(),
                "port": container_parts[2].strip(),
                "query_port": container_parts[3].strip(),
                "restart_allowed": container_parts[4].strip().lower() == "yes",
                "a2s_enabled": container_parts[5].strip().lower() == "yes",
                "description": container_parts[6].strip(),
            }

            containers.append(container_config)
            logger.debug("Parsed container '%s': %s", env_var_name, container_config)

        logger.info("Total containers configured: %d", len(containers))
        return containers

    @staticmethod
    def _parse_embed_color(raw: str) -> int:
        try:
            return int(raw, 16)
        except ValueError:
            logger.warning("Invalid EMBED_COLOR '%s', defaulting to 0x3498DB", raw)
            return 0x3498DB

    @staticmethod
    def _parse_allowed_users(raw: str) -> set[int]:
        allowed: set[int] = set()
        if raw:
            for raw_user_id in raw.split(","):
                stripped_user_id: str = raw_user_id.strip()
                if stripped_user_id.isdigit():
                    allowed.add(int(stripped_user_id))
                else:
                    logger.warning(
                        "'%s' in RESTART_ALLOWED_USERS is not a valid Discord ID and will be ignored.",
                        stripped_user_id,
                    )
        logger.debug("ALLOWED_USERS set: %s", allowed)
        return allowed

    def _templates_use_namespace(self, namespace: str) -> bool:
        template_formatter: string.Formatter = string.Formatter()
        for template_value in [self.field_template, self.field_name_template]:
            for _, placeholder_name, _, _ in template_formatter.parse(template_value):
                if not placeholder_name:
                    continue
                if placeholder_name == namespace or placeholder_name.startswith(namespace + "."):
                    return True
        return False

    def _validate_templates(self) -> None:
        """
        Validates templates for obvious placeholder mistakes.

        Rules:
          - Non-dotted placeholders must be one of the legacy flat keys.
          - Dotted placeholders must start with a known namespace:
              container.*, stats.*, external.*, a2s.*
            (We do NOT validate deeper leaves because a2s fields/rules are game-specific.)
        """
        valid_flat = {
            "alias",
            "name",
            "status",
            "cpu",
            "ram",
            "ram_percent",
            "disk",
            "description",
            "external_ip",
            "port",
            "query_port",
            "uptime",
            "health",
            "status_icon",
        }
        valid_namespaces = {"container", "stats", "external", "a2s"}

        template_formatter: string.Formatter = string.Formatter()
        for template_name, template_value in [
            ("FIELD_TEMPLATE", self.field_template),
            ("FIELD_NAME_TEMPLATE", self.field_name_template),
        ]:
            used_placeholders = {
                placeholder_name
                for _, placeholder_name, _, _ in template_formatter.parse(template_value)
                if placeholder_name
            }

            invalid: set[str] = set()
            for p in used_placeholders:
                if "." in p:
                    ns = p.split(".", 1)[0]
                    if ns not in valid_namespaces:
                        invalid.add(p)
                else:
                    if p not in valid_flat:
                        invalid.add(p)

            logger.debug("%s placeholders detected: %s", template_name, used_placeholders)
            if invalid:
                logger.critical(
                    "%s contains invalid placeholders: %s. Exiting.",
                    template_name,
                    ", ".join(sorted(invalid)),
                )
                sys.exit(1)


# ------------------------------
# Message state storage
# ------------------------------


class MessageStateStore:
    def __init__(self, state_file: str) -> None:
        self.state_file: str = state_file

    def save_message_id(self, channel_id: int, message_id: int) -> None:
        state: MessageState = {"channel_id": channel_id, "message_id": message_id}
        try:
            with open(self.state_file, "w", encoding="utf-8") as state_file:
                json.dump(state, state_file)
            logger.debug("Saved message state: channel=%d, message=%d", channel_id, message_id)
        except Exception as save_error:
            logger.warning("Failed to save message state: %s", save_error)

    def load_message_id(self) -> Tuple[Optional[int], Optional[int]]:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as state_file:
                    loaded_data_raw: Any = json.load(state_file)
                loaded_state: MessageState = dict(loaded_data_raw)
                logger.debug("Loaded message state: %s", loaded_state)
                return loaded_state.get("channel_id"), loaded_state.get("message_id")
            except Exception as load_error:
                logger.warning("Failed to load message state: %s", load_error)
        return None, None


# ------------------------------
# External IP service
# ------------------------------


class ExternalIPService:
    def __init__(self) -> None:
        self.ip: str = "N/A"

    def _update_sync(self) -> None:
        try:
            response = requests.get("https://icanhazip.com", timeout=5)
            if response.status_code == 200:
                self.ip = response.text.strip()
                logger.info("External IP updated: %s", self.ip)
        except Exception as ip_error:
            logger.warning("Failed to fetch external IP: %s", ip_error)

    async def update_async(self) -> None:
        await asyncio.to_thread(self._update_sync)


# ------------------------------
# A2S service (python-a2s)
# ------------------------------


def _obj_to_plain(obj: Any) -> Any:
    """
    Convert python-a2s return objects into dicts/lists/primitives so templates can access them.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _obj_to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_obj_to_plain(x) for x in obj]
    if hasattr(obj, "__dict__"):
        return {k: _obj_to_plain(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


class A2SService:
    def __init__(self, timeout: float = 3.0) -> None:
        self.timeout = timeout

    def _query_sync(self, host: str, port: int) -> Dict[str, Any]:
        addr = (host, port)
        info_obj = a2s.info(addr, timeout=self.timeout)
        rules_obj = a2s.rules(addr, timeout=self.timeout)
        players_obj = a2s.players(addr, timeout=self.timeout)
        return {
            "info": _obj_to_plain(info_obj),
            "rules": _obj_to_plain(rules_obj),
            "players": _obj_to_plain(players_obj),
        }

    async def query(self, host: str, port: int) -> Dict[str, Any]:
        try:
            return await asyncio.to_thread(self._query_sync, host, port)
        except Exception as e:
            return {"error": str(e), "info": None, "rules": None, "players": None}


# ------------------------------
# Docker stats / disk service
# ------------------------------


class DockerStatsService:
    def __init__(self, external_ip_service: ExternalIPService, a2s_timeout: float = 3.0) -> None:
        try:
            self.client: docker.DockerClient = docker.from_env()
            logger.info("Docker client initialized successfully")
        except Exception as docker_error:
            logger.critical("Failed to connect to Docker: %s. Exiting.", docker_error)
            sys.exit(1)

        self.external_ip_service = external_ip_service
        self.a2s_service = A2SService(timeout=a2s_timeout)

        self.disk_update_interval: float = 6 * 60 * 60  # 6 hours
        self.last_disk_update: Dict[str, float] = {}
        self.disk_usage_cache: Dict[str, float] = {}

    @staticmethod
    def format_size(value_mb: float) -> str:
        if value_mb < 1024:
            return f"{value_mb:.2f} MB"
        value_gb: float = value_mb / 1024
        if value_gb < 1024:
            return f"{value_gb:.2f} GB"
        value_tb: float = value_gb / 1024
        return f"{value_tb:.2f} TB"

    @staticmethod
    def format_uptime(container_obj: docker.models.containers.Container) -> str:
        try:
            started_at: str = container_obj.attrs["State"]["StartedAt"]
            start_time: datetime = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            uptime_delta = datetime.now(timezone.utc) - start_time
            days: int = uptime_delta.days
            seconds: int = uptime_delta.seconds
            hours: int = seconds // 3600
            minutes: int = (seconds % 3600) // 60
            return f"{days}d {hours}h {minutes}m"
        except Exception:
            return "N/A"

    @staticmethod
    def calculate_cpu_percent_from_stats(
        first_stats: Dict[str, Any],
        second_stats: Dict[str, Any],
    ) -> float:
        try:
            cpu_delta: int = (
                second_stats["cpu_stats"]["cpu_usage"]["total_usage"]
                - first_stats["cpu_stats"]["cpu_usage"]["total_usage"]
            )
            system_delta: int = (
                second_stats["cpu_stats"]["system_cpu_usage"]
                - first_stats["cpu_stats"].get("system_cpu_usage", 0)
            )
            percpu_count: int = first_stats["cpu_stats"].get(
                "online_cpus",
                len(second_stats["cpu_stats"]["cpu_usage"].get("percpu_usage", [])),
            )
            if system_delta > 0 and cpu_delta > 0:
                return (cpu_delta / system_delta) * percpu_count * 100.0
            return 0.0
        except Exception:
            return 0.0

    @staticmethod
    def get_directory_size_mb(path: str) -> float:
        """
        Return the *actual* disk usage in MB for a directory tree.
        Uses st_blocks * 512 when available (allocated bytes).
        """
        total_bytes: int = 0
        for dirpath, _dirnames, filenames in os.walk(path):
            for filename in filenames:
                try:
                    file_path: str = os.path.join(dirpath, filename)
                    stat_result = os.stat(file_path, follow_symlinks=False)
                    if hasattr(stat_result, "st_blocks") and stat_result.st_blocks is not None:
                        total_bytes += stat_result.st_blocks * 512
                    else:
                        total_bytes += stat_result.st_size
                except Exception:
                    pass
        return total_bytes / (1024**2)

    def _get_container_disk_usage_sync(self, container_obj: docker.models.containers.Container) -> float:
        total_mb: float = 0.0
        logger.debug("[disk] Calculating disk usage for %s", container_obj.name)

        # 1) Writable layer
        try:
            docker_df: Dict[str, Any] = self.client.api.df()
            container_summaries: List[Dict[str, Any]] = docker_df.get("Containers", [])
            for container_summary in container_summaries:
                if container_summary["Id"].startswith(container_obj.id):
                    writable_layer_mb: float = container_summary.get("SizeRw", 0) / (1024**2)
                    logger.debug(
                        "[disk:%s] Writable layer: %.2f MB",
                        container_obj.name,
                        writable_layer_mb,
                    )
                    total_mb += writable_layer_mb
                    break
        except Exception as writable_error:
            logger.warning(
                "[disk:%s] Failed to get writable layer size: %s",
                container_obj.name,
                writable_error,
            )

        # 2) Mounted volumes
        try:
            mount_info_list: List[Dict[str, Any]] = container_obj.attrs.get("Mounts", [])
            for mount_info in mount_info_list:
                host_path: Optional[str] = mount_info.get("Source")
                if host_path and os.path.exists(host_path):
                    volume_mb: float = self.get_directory_size_mb(host_path)
                    logger.debug(
                        "[disk:%s] Volume %s: %.2f MB",
                        container_obj.name,
                        host_path,
                        volume_mb,
                    )
                    total_mb += volume_mb
        except Exception as volume_error:
            logger.warning(
                "[disk:%s] Failed to get volumes size: %s",
                container_obj.name,
                volume_error,
            )

        # 3) Image size
        try:
            image_obj = self.client.images.get(container_obj.image.id)
            image_size_mb: float = image_obj.attrs.get("Size", 0) / (1024**2)
            logger.debug("[disk:%s] Image size: %.2f MB", container_obj.name, image_size_mb)
            total_mb += image_size_mb
        except Exception as image_error:
            logger.warning("[disk:%s] Failed to get image size: %s", container_obj.name, image_error)

        logger.debug("[disk:%s] Total calculated: %.2f MB", container_obj.name, total_mb)
        return total_mb

    def _get_container_stats_sync(self, container_name: str) -> StatsDict:
        """
        Blocking container stats gathering. Executed in a thread via asyncio.to_thread.
        """
        try:
            container_obj: docker.models.containers.Container = self.client.containers.get(container_name)
            stats_stream = container_obj.stats(decode=True)

            # CPU usage: sample over ~1s
            if container_obj.status in ["running", "starting"]:
                first_stats: Dict[str, Any] = next(stats_stream)
                time.sleep(1)
                second_stats: Dict[str, Any] = next(stats_stream)

                cpu_percent: float = self.calculate_cpu_percent_from_stats(first_stats, second_stats)

                # RAM usage
                mem_usage_mb: float = first_stats["memory_stats"]["usage"] / (1024**2)
                mem_limit_mb: float = first_stats["memory_stats"]["limit"] / (1024**2)
                mem_percent: float = (mem_usage_mb / mem_limit_mb * 100) if mem_limit_mb > 0 else 0.0
            else:
                cpu_percent = 0.0
                mem_usage_mb = 0.0
                mem_percent = 0.0

            # Disk usage (refresh every 6 hours)
            current_time: float = time.time()
            if (
                container_name not in self.last_disk_update
                or current_time - self.last_disk_update[container_name] > self.disk_update_interval
            ):
                logger.debug("[disk:%s] Refreshing cached disk usage...", container_name)
                disk_usage_mb: float = self._get_container_disk_usage_sync(container_obj)
                self.disk_usage_cache[container_name] = disk_usage_mb
                self.last_disk_update[container_name] = current_time
            else:
                disk_usage_mb = self.disk_usage_cache.get(container_name, 0.0)
                cache_age_seconds: float = current_time - self.last_disk_update[container_name]
                logger.debug(
                    "[disk:%s] Using cached disk usage (%.0fs old)",
                    container_name,
                    cache_age_seconds,
                )

            uptime_str: str = self.format_uptime(container_obj)
            external_ip: str = self.external_ip_service.ip

            health_status: str = container_obj.attrs["State"].get("Health", {}).get("Status", "")
            container_status: str = container_obj.status

            # Status icon mapping (feel free to customize)
            if container_status == "running" and (health_status in ["healthy", ""]):
                status_icon: str = "✅"
            elif container_status == "exited" or (container_status == "running" and health_status == "unhealthy"):
                status_icon = "❌"
            elif container_status in ["starting", "restarting"] or (
                container_status == "running" and health_status == "starting"
            ):
                status_icon = "⏳"
            elif container_status == "paused":
                status_icon = "⏸️"
            elif container_status == "dead":
                status_icon = "❌"
            else:
                status_icon = "❓"

            return {
                "status": container_status,
                "health": health_status,
                "status_icon": status_icon,
                "cpu": cpu_percent,
                "ram": self.format_size(mem_usage_mb),
                "ram_percent": mem_percent,
                "disk": self.format_size(disk_usage_mb),
                "uptime": uptime_str,
                "external_ip": external_ip,
            }
        except Exception as stats_error:
            logger.error("Failed to get stats for %s: %s", container_name, stats_error)
            return {"error": str(stats_error)}

    async def get_container_stats(self, container_name: str) -> StatsDict:
        return await asyncio.to_thread(self._get_container_stats_sync, container_name)

    async def get_container_stats_with_a2s(self, container_cfg: ContainerConfig) -> StatsDict:
        """
        Docker stats + A2S query data addressed via:
          host = external_ip
          port = container_cfg["port"]
        """
        stats = await self.get_container_stats(container_cfg["name"])
        # Per-container A2S toggle: skip querying entirely when disabled
        if not container_cfg.get("a2s_enabled", True):
            stats["a2s"] = None
            return stats


        # If docker stats failed, still try to attach an a2s error placeholder consistently.
        host = self.external_ip_service.ip
        port_raw = (container_cfg.get("query_port") or "").strip()

        try:
            port = int(port_raw)
        except Exception:
            port = -1

        a2s_enabled = container_cfg.get("a2s_enabled", True)
        logger.debug(
            "[a2s:%s] enabled=%s host=%s port_raw=%s parsed_port=%s",
            container_cfg.get("name"),
            a2s_enabled,
            host,
            port_raw,
            port,
        )

        # Per-container A2S toggle
        if not a2s_enabled:
            logger.debug("[a2s:%s] Skipping A2S query (disabled)", container_cfg.get("name"))
            stats["a2s"] = None
            return stats

        # A2S enabled: query if possible
        if host and host != "N/A" and port > 0:
            stats["a2s"] = await self.a2s_service.query(host, port)
            if isinstance(stats.get("a2s"), dict) and stats["a2s"].get("error"):
                logger.debug(
                    "[a2s:%s] Query completed with error: %s",
                    container_cfg.get("name"),
                    stats["a2s"].get("error"),
                )
                logger.debug("[a2s:%s] %s", container_cfg.get("name"), stats.get("a2s"))
            else:
                # Best-effort: log server name if present
                server_name = None
                try:
                    server_name = (stats["a2s"] or {}).get("info", {}).get("server_name")
                except Exception:
                    server_name = None

                logger.debug(
                    "[a2s:%s] Query completed OK%s",
                    container_cfg.get("name"),
                    f" (server_name={server_name})" if server_name else "",
                )
                logger.debug("[a2s:%s] %s", container_cfg.get("name"), stats.get("a2s"))
        else:
            logger.debug(
                "[a2s:%s] Skipping A2S query (missing external_ip or invalid port)",
                container_cfg.get("name"),
            )
            stats["a2s"] = {"error": "missing external_ip or invalid port", "info": None, "rules": None, "players": None}

        return stats
        
    def get_container_status_sync(self, container_name: str) -> str:
        try:
            container = self.client.containers.get(container_name)
            return container.status
        except Exception:
            return "unknown"

    async def get_container_status(self, container_name: str) -> str:
        return await asyncio.to_thread(self.get_container_status_sync, container_name)

# ------------------------------
# Restart manager
# ------------------------------


class RestartManager:
    def __init__(
        self,
        docker_client: docker.DockerClient,
        allowed_users: set[int],
        rate_limit_count: int,
        rate_limit_period: float,
    ) -> None:
        self.client = docker_client
        self.allowed_users = allowed_users
        self.rate_limit_count = rate_limit_count
        self.rate_limit_period = rate_limit_period
        self.timestamps: Dict[str, List[float]] = {}

    def can_act(self, container_name: str, user_id: int) -> Tuple[bool, Optional[int]]:
        if user_id not in self.allowed_users:
            return False, None

        now = time.time()
        history = self.timestamps.setdefault(container_name, [])
        history = [t for t in history if now - t < self.rate_limit_period]
        self.timestamps[container_name] = history

        if len(history) >= self.rate_limit_count:
            retry_after = int(self.rate_limit_period - (now - history[0]))
            return False, retry_after

        history.append(now)
        return True, None

    async def start_container(self, name: str) -> None:
        await asyncio.to_thread(lambda: self.client.containers.get(name).start())

    async def stop_container(self, name: str) -> None:
        await asyncio.to_thread(lambda: self.client.containers.get(name).stop())

    async def restart_container(self, name: str) -> None:
        await asyncio.to_thread(lambda: self.client.containers.get(name).restart())

# ------------------------------
# Embed builder
# ------------------------------


class EmbedBuilder:
    def __init__(self, config: AppConfig, stats_service: DockerStatsService) -> None:
        self.config = config
        self.stats_service = stats_service

        self._warned_a2s_disabled: set[str] = set()

    async def build_embed(self) -> discord.Embed:
        embed: discord.Embed = discord.Embed(
            title=self.config.embed_title,
            color=self.config.embed_color,
            timestamp=datetime.now(UTC),
        )

        stats_tasks: List[asyncio.Task[StatsDict]] = [
            asyncio.create_task(self.stats_service.get_container_stats_with_a2s(container_cfg))
            for container_cfg in self.config.containers
        ]
        stats_results: List[StatsDict] = await asyncio.gather(*stats_tasks, return_exceptions=False)

        for container_cfg, stats_dict in zip(self.config.containers, stats_results):
            alias: str = container_cfg["alias"]
            # Validation warning: templates reference A2S placeholders but this container has A2S disabled
            if (
                self.config.template_uses_a2s_placeholders
                and not container_cfg.get("a2s_enabled", True)
                and alias not in self._warned_a2s_disabled
            ):
                logger.warning(
                    "A2S is disabled for container '%s' but templates reference {a2s.*} placeholders. "
                    "Those placeholders will render as empty for this container.",
                    alias,
                )
                self._warned_a2s_disabled.add(alias)

            container_name: str = container_cfg["name"]
            description_template: str = container_cfg["description"]
            port: str = container_cfg["port"]
            query_port: str = container_cfg["query_port"]
            # Pull baseline values
            external_ip_val = stats_dict.get("external_ip", "N/A")

            # Nested namespaces (for dot templates)
            container_ns = {
                "alias": alias,
                "name": container_name,
                "port": port,
                "description": description_template,
                "query_port": query_port,
            }
            stats_ns = {
                "status": stats_dict.get("status", "N/A"),
                "health": stats_dict.get("health", "N/A"),
                "status_icon": stats_dict.get("status_icon", "❌"),
                "cpu": stats_dict.get("cpu", 0.0),
                "ram": stats_dict.get("ram", "N/A"),
                "ram_percent": stats_dict.get("ram_percent", 0.0),
                "disk": stats_dict.get("disk", "N/A"),
                "uptime": stats_dict.get("uptime", "N/A"),
            }
            external_ns = {"ip": external_ip_val}
            a2s_ns = stats_dict.get("a2s", {}) or {}

            # Flat (legacy) keys kept for backward compatibility
            ctx: Dict[str, Any] = {
                "alias": alias,
                "name": container_name,
                "status": stats_ns["status"],
                "health": stats_ns["health"],
                "status_icon": stats_ns["status_icon"],
                "cpu": stats_ns["cpu"],
                "ram": stats_ns["ram"],
                "ram_percent": stats_ns["ram_percent"],
                "disk": stats_ns["disk"],
                "port": port,
                "query_port": query_port,
                "uptime": stats_ns["uptime"],
                "description": description_template,
                "external_ip": external_ip_val,
                # dot namespaces
                "container": container_ns,
                "stats": stats_ns,
                "external": external_ns,
                "a2s": a2s_ns,
            }

            # Allow the container's configured description to be a template too
            rendered_description = render_template(description_template, ctx) if description_template else ""
            ctx["description"] = rendered_description
            ctx["container"]["description"] = rendered_description

            if "error" in stats_dict:
                # In error case, show error and still let templates render a name using context
                ctx["status"] = "N/A"
                ctx["health"] = "N/A"
                ctx["status_icon"] = "❌"
                ctx["cpu"] = 0.0
                ctx["ram"] = "N/A"
                ctx["ram_percent"] = 0.0
                ctx["disk"] = "N/A"
                ctx["uptime"] = "N/A"
                ctx["external_ip"] = "N/A"
                ctx["stats"]["status"] = "N/A"
                ctx["stats"]["health"] = "N/A"
                ctx["stats"]["status_icon"] = "❌"
                ctx["stats"]["cpu"] = 0.0
                ctx["stats"]["ram"] = "N/A"
                ctx["stats"]["ram_percent"] = 0.0
                ctx["stats"]["disk"] = "N/A"
                ctx["stats"]["uptime"] = "N/A"
                ctx["external"]["ip"] = "N/A"

                field_name: str = render_template(self.config.field_name_template, ctx)
                field_value: str = f"❌ Error: `{stats_dict['error']}`"
            else:
                field_name = render_template(self.config.field_name_template, ctx)
                field_value = render_template(self.config.field_template, ctx)

            embed.add_field(name=field_name, value=field_value, inline=False)

        return embed


# ------------------------------
# Discord UI (Restart button / view)
# ------------------------------

class RestartView(View):
    def __init__(
        self,
        config,
        manager: RestartManager,
        stats_service,
    ):
        super().__init__(timeout=None)
        self.config = config
        self.manager = manager
        self.stats_service = stats_service

    async def setup(self):
        for row, cfg in enumerate(self.config.containers):
            status = await self.stats_service.get_container_status(cfg["name"])
            is_running = status == "running"
            allowed = cfg["restart_allowed"]

            self.add_item(ContainerActionButton(
                label="Start",
                style=discord.ButtonStyle.success,
                alias=cfg["alias"],
                container_name=cfg["name"],
                action="start",
                enabled=allowed and not is_running,
                row=row,
                manager=self.manager,
            ))

            self.add_item(ContainerActionButton(
                label="Stop",
                style=discord.ButtonStyle.secondary,
                alias=cfg["alias"],
                container_name=cfg["name"],
                action="stop",
                enabled=allowed and is_running,
                row=row,
                manager=self.manager,
            ))

            self.add_item(ContainerActionButton(
                label="Restart",
                style=discord.ButtonStyle.danger,
                alias=cfg["alias"],
                container_name=cfg["name"],
                action="restart",
                enabled=allowed and is_running,
                row=row,
                manager=self.manager,
            ))

class RestartButton(Button):
    def __init__(
        self,
        alias: str,
        container_name: str,
        restart_allowed: bool,
        restart_manager: RestartManager,
    ) -> None:
        super().__init__(
            label=f"Restart {alias}",
            style=discord.ButtonStyle.danger,
            disabled=not restart_allowed,
        )
        self.alias: str = alias
        self.container_name: str = container_name
        self.restart_manager: RestartManager = restart_manager

    async def callback(self, interaction: discord.Interaction) -> None:  # type: ignore[override]
        if interaction.user is None:
            await interaction.response.send_message("⛔ Unable to determine your identity.", ephemeral=True)
            return

        can_restart, retry_after = self.restart_manager.can_restart(self.container_name, interaction.user.id)
        if not can_restart:
            if retry_after is None:
                await interaction.response.send_message(
                    "⛔ You are not allowed to restart containers.",
                    ephemeral=True,
                )
            else:
                await interaction.response.send_message(
                    f"⏳ You’ve hit the restart limit for **{self.alias}**. Try again in {retry_after}s.",
                    ephemeral=True,
                )
            return

        await interaction.response.defer(ephemeral=True)
        try:
            await self.restart_manager.restart_container(self.container_name)
            await interaction.followup.send(
                f"✅ Restarted **{self.alias}** (`{self.container_name}`).",
                ephemeral=True,
            )
        except Exception as restart_error:
            await interaction.followup.send(
                f"❌ Failed to restart **{self.alias}**:\n`{restart_error}`",
                ephemeral=True,
            )


class RestartView(View):
    def __init__(self, config: AppConfig, restart_manager: RestartManager) -> None:
        super().__init__(timeout=None)
        for container_cfg in config.containers:
            self.add_item(
                RestartButton(
                    alias=container_cfg["alias"],
                    container_name=container_cfg["name"],
                    restart_allowed=container_cfg["restart_allowed"],
                    restart_manager=restart_manager,
                )
            )
            
class ContainerActionButton(Button):
    def __init__(
        self,
        *,
        label: str,
        style: discord.ButtonStyle,
        alias: str,
        container_name: str,
        action: str,
        enabled: bool,
        row: int,
        manager: RestartManager,
    ):
        super().__init__(label=label, style=style, disabled=not enabled, row=row)
        self.alias = alias
        self.container_name = container_name
        self.action = action
        self.manager = manager

    async def callback(self, interaction: discord.Interaction):
        if interaction.user is None:
            await interaction.response.send_message("⛔ Unknown user.", ephemeral=True)
            return

        allowed, retry_after = self.manager.can_act(self.container_name, interaction.user.id)
        if not allowed:
            msg = (
                "⛔ You are not allowed to manage containers."
                if retry_after is None
                else f"⏳ Rate limited. Try again in {retry_after}s."
            )
            await interaction.response.send_message(msg, ephemeral=True)
            return

        if self.action == "stop":
            await interaction.response.send_modal(
                StopConfirmModal(self.alias, self.container_name, self.manager)
            )
            return

        await interaction.response.defer(ephemeral=True)
        try:
            if self.action == "start":
                await self.manager.start_container(self.container_name)
            elif self.action == "restart":
                await self.manager.restart_container(self.container_name)

            await interaction.followup.send(
                f"✅ **{self.action.title()}** executed for **{self.alias}**.",
                ephemeral=True,
            )
        except Exception as e:
            await interaction.followup.send(
                f"❌ Failed to {self.action} **{self.alias}**:\n`{e}`",
                ephemeral=True,
            )

class StopConfirmModal(Modal, title="Confirm Stop"):
    def __init__(self, alias: str, container_name: str, manager: RestartManager):
        super().__init__()
        self.alias = alias
        self.container_name = container_name
        self.manager = manager

        self.confirm = TextInput(
            label=f'Type STOP to confirm stopping "{alias}"',
            placeholder="STOP",
            required=True,
        )
        self.add_item(self.confirm)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if self.confirm.value.strip().upper() != "STOP":
            await interaction.response.send_message(
                "❌ Confirmation failed. Container not stopped.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(ephemeral=True)
        try:
            await self.manager.stop_container(self.container_name)
            await interaction.followup.send(
                f"🛑 **{self.alias}** has been stopped.",
                ephemeral=True,
            )
        except Exception as e:
            await interaction.followup.send(
                f"❌ Failed to stop **{self.alias}**:\n`{e}`",
                ephemeral=True,
            )

# ------------------------------
# Discord Bot
# ------------------------------


class DockerStatusBot(commands.Bot):
    def __init__(
        self,
        config: AppConfig,
        message_state_store: MessageStateStore,
        external_ip_service: ExternalIPService,
        stats_service: DockerStatsService,
        embed_builder: EmbedBuilder,
        restart_manager: RestartManager,
    ) -> None:
        intents: discord.Intents = discord.Intents.default()
        super().__init__(command_prefix="!", intents=intents)

        self.config = config
        self.message_state_store = message_state_store
        self.external_ip_service = external_ip_service
        self.stats_service = stats_service
        self.embed_builder = embed_builder
        self.restart_manager = restart_manager

        self.message_to_update: Optional[discord.Message] = None

        # Attach tasks
        self.update_message_task = tasks.loop(minutes=1)(self._update_message_task)
        self.update_external_ip_task = tasks.loop(hours=6)(self._update_external_ip_task)

    async def _update_message_task(self) -> None:
        if self.message_to_update:
            try:
                embed: discord.Embed = await self.embed_builder.build_embed()
                view = RestartView(self.config, self.restart_manager, self.stats_service)
                await view.setup()
                await self.message_to_update.edit(embed=embed, view=view)
                logger.debug("Updated message %d", self.message_to_update.id)
            except Exception as update_error:
                logger.warning("Failed to update message: %s", update_error)

    async def _update_external_ip_task(self) -> None:
        logger.info("Updating external IP...")
        await self.external_ip_service.update_async()

    async def on_ready(self) -> None:  # type: ignore[override]
        assert self.user is not None
        logger.info("Logged in as %s", self.user)

        saved_channel_id, saved_message_id = self.message_state_store.load_message_id()
        target_channel = self.get_channel(self.config.channel_id)

        if target_channel is None or not isinstance(
            target_channel, (discord.TextChannel, discord.Thread, discord.DMChannel)
        ):
            logger.critical("Channel not found or wrong type. Exiting.")
            await self.close()
            sys.exit(1)

        # Try to resume editing an existing message if saved
        if saved_channel_id == self.config.channel_id and saved_message_id:
            try:
                self.message_to_update = await target_channel.fetch_message(saved_message_id)  # type: ignore[arg-type]
                logger.info("Resuming updates on message %d", saved_message_id)
            except Exception as fetch_error:
                logger.warning("Failed to fetch saved message: %s", fetch_error)
                self.message_to_update = None

        # If no message to resume, create a new one
        if self.message_to_update is None:
            try:
                initial_embed = await self.embed_builder.build_embed()
                initial_view = RestartView(self.config, self.restart_manager, self.stats_service)
                await initial_view.setup()
                sent_message = await target_channel.send(  # type: ignore[arg-type]
                    embed=initial_embed,
                    view=initial_view,
                )
                self.message_to_update = sent_message
                self.message_state_store.save_message_id(self.config.channel_id, sent_message.id)
                logger.info("Created new message %d", sent_message.id)
            except Exception as send_error:
                logger.critical("Failed to send new message: %s. Exiting.", send_error)
                await self.close()
                sys.exit(1)

        self.update_message_task.start()
        self.update_external_ip_task.start()


# ------------------------------
# Main entrypoint
# ------------------------------


def main() -> None:
    config = AppConfig()
    message_state_store = MessageStateStore(config.message_state_file)

    external_ip_service = ExternalIPService()
    stats_service = DockerStatsService(external_ip_service, a2s_timeout=config.a2s_timeout)
    embed_builder = EmbedBuilder(config, stats_service)

    restart_manager = RestartManager(
        docker_client=stats_service.client,
        allowed_users=config.allowed_users,
        rate_limit_count=config.restart_rate_limit_count,
        rate_limit_period=config.restart_rate_limit_period,
    )

    bot = DockerStatusBot(
        config=config,
        message_state_store=message_state_store,
        external_ip_service=external_ip_service,
        stats_service=stats_service,
        embed_builder=embed_builder,
        restart_manager=restart_manager,
    )
    bot.run(config.token)


if __name__ == "__main__":
    main()
