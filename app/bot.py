import os
import re
import json
import string
import sys
import logging
import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import requests
import docker
import discord
from discord.ext import commands, tasks
from discord.ui import View, Button
from dotenv import load_dotenv


# ------------------------------
# Type definitions
# ------------------------------
class ContainerConfig(TypedDict):
    alias: str
    name: str
    restart_allowed: bool
    description: str


class MessageState(TypedDict, total=False):
    channel_id: int
    message_id: int


StatsDict = Dict[str, Any]


# ------------------------------
# Logging configuration
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger: logging.Logger = logging.getLogger(__name__)
if os.getenv("DEBUG", "0") == "1":
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled")


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
            self.config_dir,
            os.getenv("MESSAGE_STATE_FILE", "message_state.json"),
        )

        self.restart_rate_limit_count: int = int(
            os.getenv("RESTART_RATE_LIMIT_COUNT", "2")
        )
        self.restart_rate_limit_period: float = float(
            os.getenv("RESTART_RATE_LIMIT_PERIOD", "300")
        )

        # Templates
        self.field_template: str = os.getenv(
            "CONTAINER_FIELD_TEMPLATE",
            (
                "**Description:** {description}\\n"
                "**Status:** {status}\\n"
                "**CPU:** {cpu:.2f}%\\n"
                "**RAM:** {ram} ({ram_percent:.2f}%)\\n"
                "**Disk:** {disk}\\n"
                "**Port:** {port}\\n"
                "**Uptime:** {uptime}\\n"
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
        self.allowed_users: set[int] = self._parse_allowed_users(
            self.allowed_users_raw
        )

        self._validate_templates()

    def _load_required(self, key: str) -> str:
        value = os.getenv(key)
        if not value:
            logger.critical("%s is not set. Exiting.", key)
            sys.exit(1)
        return value

    def _load_containers(self) -> List[ContainerConfig]:
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
                "No containers configured. Define environment variables like "
                "CONTAINER_1, CONTAINER_2, etc. Exiting."
            )
            sys.exit(1)

        container_env_entries.sort(
            key=lambda entry: int(
                container_pattern.match(entry[0]).group(1)  # type: ignore[call-arg]
            )
        )

        containers: List[ContainerConfig] = []
        for env_var_name, env_var_value in container_env_entries:
            container_parts: List[str] = env_var_value.split(":", 3)
            if len(container_parts) < 2:
                logger.critical(
                    "Container entry '%s' must have at least alias and docker_name. Exiting.",
                    env_var_name,
                )
                sys.exit(1)

            container_alias: str = container_parts[0].strip()
            container_name: str = container_parts[1].strip()
            container_restart_allowed: bool = (
                len(container_parts) > 2
                and container_parts[2].strip().lower() == "yes"
            )
            container_description: str = (
                container_parts[3].strip() if len(container_parts) > 3 else ""
            )

            container_config: ContainerConfig = {
                "alias": container_alias,
                "name": container_name,
                "restart_allowed": container_restart_allowed,
                "description": container_description,
            }
            containers.append(container_config)
            logger.debug("Parsed container '%s': %s", env_var_name, container_config)

        logger.info("Total containers configured: %d", len(containers))
        return containers

    def _parse_embed_color(self, raw: str) -> int:
        try:
            return int(raw, 16)
        except ValueError:
            logger.warning(
                "Invalid EMBED_COLOR '%s', defaulting to 0x3498DB",
                raw,
            )
            return 0x3498DB

    def _parse_allowed_users(self, raw: str) -> set[int]:
        allowed: set[int] = set()
        if raw:
            for raw_user_id in raw.split(","):
                stripped_user_id: str = raw_user_id.strip()
                if stripped_user_id.isdigit():
                    allowed.add(int(stripped_user_id))
                else:
                    logger.warning(
                        "'%s' in RESTART_ALLOWED_USERS is not a valid Discord ID "
                        "and will be ignored.",
                        stripped_user_id,
                    )
        logger.debug("ALLOWED_USERS set: %s", allowed)
        return allowed

    def _validate_templates(self) -> None:
        valid_placeholders = {
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
            "uptime",
            "health",
            "status_icon",
        }

        template_formatter: string.Formatter = string.Formatter()
        for template_name, template_value in [
            ("FIELD_TEMPLATE", self.field_template),
            ("FIELD_NAME_TEMPLATE", self.field_name_template),
        ]:
            used_placeholders = {
                placeholder_name
                for _, placeholder_name, _, _ in template_formatter.parse(
                    template_value
                )
                if placeholder_name
            }
            invalid_placeholders = used_placeholders - valid_placeholders
            logger.debug(
                "%s placeholders detected: %s",
                template_name,
                used_placeholders,
            )
            if invalid_placeholders:
                logger.critical(
                    "%s contains invalid placeholders: %s. Exiting.",
                    template_name,
                    ", ".join(invalid_placeholders),
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
            logger.debug(
                "Saved message state: channel=%d, message=%d",
                channel_id,
                message_id,
            )
        except Exception as save_error:
            logger.warning("Failed to save message state: %s", save_error)

    def load_message_id(self) -> Tuple[Optional[int], Optional[int]]:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as state_file:
                    loaded_data_raw: Any = json.load(state_file)
                loaded_state: MessageState = dict(loaded_data_raw)
                logger.debug("Loaded message state: %s", loaded_state)
                return (
                    loaded_state.get("channel_id"),
                    loaded_state.get("message_id"),
                )
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
# Docker stats / disk service
# ------------------------------
class DockerStatsService:
    def __init__(self, external_ip_service: ExternalIPService) -> None:
        try:
            self.client: docker.DockerClient = docker.from_env()
            logger.info("Docker client initialized successfully")
        except Exception as docker_error:
            logger.critical("Failed to connect to Docker: %s. Exiting.", docker_error)
            sys.exit(1)

        self.external_ip_service = external_ip_service
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
            start_time: datetime = datetime.fromisoformat(
                started_at.replace("Z", "+00:00")
            )
            uptime_delta = datetime.now(timezone.utc) - start_time
            days: int = uptime_delta.days
            seconds: int = uptime_delta.seconds
            hours: int = seconds // 3600
            minutes: int = (seconds % 3600) // 60
            return f"{days}d {hours}h {minutes}m"
        except Exception:
            return "N/A"

    @staticmethod
    def get_first_port(container_obj: docker.models.containers.Container) -> str:
        try:
            port_mappings: Dict[str, Any] = container_obj.attrs.get(
                "NetworkSettings", {}
            ).get("Ports", {})
            for _, host_mappings in port_mappings.items():
                if host_mappings:
                    host_port: Optional[str] = host_mappings[0].get("HostPort")
                    if host_port:
                        return host_port
        except Exception:
            return "N/A"
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
                len(
                    second_stats["cpu_stats"]["cpu_usage"].get(
                        "percpu_usage", []
                    )
                ),
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

        Uses st_blocks * 512 (POSIX block size units) when available, which
        reflects allocated space on disk rather than just logical file size.
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
                    # Skip files that can't be accessed
                    pass

        return total_bytes / (1024**2)

    def _get_container_disk_usage_sync(
        self,
        container_obj: docker.models.containers.Container,
    ) -> float:
        """Blocking disk usage computation (to be called from a thread)."""
        total_mb: float = 0.0
        logger.debug("[disk] Calculating disk usage for %s", container_obj.name)

        # 1. Writable layer
        try:
            docker_df: Dict[str, Any] = self.client.api.df()
            container_summaries: List[Dict[str, Any]] = docker_df.get(
                "Containers", []
            )
            for container_summary in container_summaries:
                if container_summary["Id"].startswith(container_obj.id):
                    writable_layer_mb: float = (
                        container_summary.get("SizeRw", 0) / (1024**2)
                    )
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

        # 2. Mounted volumes
        try:
            mount_info_list: List[Dict[str, Any]] = container_obj.attrs.get(
                "Mounts", []
            )
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

        # 3. Image size
        try:
            image_obj = self.client.images.get(container_obj.image.id)
            image_size_mb: float = image_obj.attrs.get("Size", 0) / (1024**2)
            logger.debug(
                "[disk:%s] Image size: %.2f MB",
                container_obj.name,
                image_size_mb,
            )
            total_mb += image_size_mb
        except Exception as image_error:
            logger.warning(
                "[disk:%s] Failed to get image size: %s",
                container_obj.name,
                image_error,
            )

        logger.debug(
            "[disk:%s] Total calculated: %.2f MB",
            container_obj.name,
            total_mb,
        )
        return total_mb

    def _get_container_stats_sync(self, container_name: str) -> StatsDict:
        """
        Blocking container stats gathering.
        This is executed in a thread via asyncio.to_thread.
        """
        try:
            container_obj: docker.models.containers.Container = (
                self.client.containers.get(container_name)
            )
            stats_stream = container_obj.stats(decode=True)

            # CPU usage: sample over ~1s
            first_stats: Dict[str, Any] = next(stats_stream)
            time.sleep(1)
            second_stats: Dict[str, Any] = next(stats_stream)
            cpu_percent: float = self.calculate_cpu_percent_from_stats(
                first_stats,
                second_stats,
            )

            # RAM usage
            mem_usage_mb: float = first_stats["memory_stats"]["usage"] / (1024**2)
            mem_limit_mb: float = first_stats["memory_stats"]["limit"] / (1024**2)
            mem_percent: float = (
                (mem_usage_mb / mem_limit_mb * 100) if mem_limit_mb > 0 else 0.0
            )

            # Disk usage (refresh every 6 hours)
            current_time: float = time.time()
            if (
                container_name not in self.last_disk_update
                or current_time - self.last_disk_update[container_name]
                > self.disk_update_interval
            ):
                logger.debug(
                    "[disk:%s] Refreshing cached disk usage...",
                    container_name,
                )
                disk_usage_mb: float = self._get_container_disk_usage_sync(
                    container_obj
                )
                self.disk_usage_cache[container_name] = disk_usage_mb
                self.last_disk_update[container_name] = current_time
            else:
                disk_usage_mb = self.disk_usage_cache.get(container_name, 0.0)
                cache_age_seconds: float = (
                    current_time - self.last_disk_update[container_name]
                )
                logger.debug(
                    "[disk:%s] Using cached disk usage (%.0fs old)",
                    container_name,
                    cache_age_seconds,
                )

            host_port: str = self.get_first_port(container_obj)
            uptime_str: str = self.format_uptime(container_obj)
            external_ip: str = self.external_ip_service.ip

            health_status: str = container_obj.attrs["State"].get(
                "Health", {}
            ).get("Status", "")
            container_status: str = container_obj.status

            if container_status == "running" and health_status == "healthy":
                status_icon: str = "🟢"
            elif container_status == "exited" or (
                container_status == "running" and health_status == "unhealthy"
            ):
                status_icon = "🔴"
            elif container_status in ["starting", "restarting"] or (
                container_status == "running" and health_status == "starting"
            ):
                status_icon = "🟠"
            elif container_status == "paused":
                status_icon = "🟡"
            elif container_status == "dead":
                status_icon = "❌"
            else:
                status_icon = "❓"

            result: StatsDict = {
                "status": container_status,
                "health": health_status,
                "status_icon": status_icon,
                "cpu": cpu_percent,
                "ram": self.format_size(mem_usage_mb),
                "ram_percent": mem_percent,
                "disk": self.format_size(disk_usage_mb),
                "port": host_port,
                "uptime": uptime_str,
                "external_ip": external_ip,
            }
            return result
        except Exception as stats_error:
            logger.error("Failed to get stats for %s: %s", container_name, stats_error)
            return {"error": str(stats_error)}

    async def get_container_stats(self, container_name: str) -> StatsDict:
        return await asyncio.to_thread(self._get_container_stats_sync, container_name)


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
        # { container_name: [timestamps] }
        self.restart_timestamps: Dict[str, List[float]] = {}

    def can_restart(
        self,
        container_name: str,
        user_id: int,
    ) -> Tuple[bool, Optional[int]]:
        if user_id not in self.allowed_users:
            return False, None

        now: float = time.time()
        container_history: List[float] = self.restart_timestamps.setdefault(
            container_name,
            [],
        )

        filtered_history: List[float] = [
            timestamp
            for timestamp in container_history
            if now - timestamp < self.rate_limit_period
        ]
        self.restart_timestamps[container_name] = filtered_history

        if len(filtered_history) >= self.rate_limit_count:
            first_timestamp: float = filtered_history[0]
            retry_after: int = int(
                self.rate_limit_period - (now - first_timestamp)
            )
            return False, retry_after

        filtered_history.append(now)
        self.restart_timestamps[container_name] = filtered_history
        return True, None

    async def restart_container(self, container_name: str) -> None:
        def _restart() -> None:
            container_obj: docker.models.containers.Container = (
                self.client.containers.get(container_name)
            )
            container_obj.restart()

        await asyncio.to_thread(_restart)


# ------------------------------
# Embed builder
# ------------------------------
class EmbedBuilder:
    def __init__(
        self,
        config: AppConfig,
        stats_service: DockerStatsService,
    ) -> None:
        self.config = config
        self.stats_service = stats_service

    async def build_embed(self) -> discord.Embed:
        embed: discord.Embed = discord.Embed(
            title=self.config.embed_title,
            color=self.config.embed_color,
            timestamp=datetime.utcnow(),
        )

        stats_tasks: List[asyncio.Task[StatsDict]] = [
            asyncio.create_task(
                self.stats_service.get_container_stats(container_cfg["name"])
            )
            for container_cfg in self.config.containers
        ]
        stats_results: List[StatsDict] = await asyncio.gather(
            *stats_tasks, return_exceptions=False
        )

        for container_cfg, stats_dict in zip(
            self.config.containers, stats_results
        ):
            alias: str = container_cfg["alias"]
            container_name: str = container_cfg["name"]
            description: str = container_cfg["description"]

            if "error" in stats_dict:
                placeholder_values: Dict[str, Any] = {
                    "alias": alias,
                    "name": container_name,
                    "status": "N/A",
                    "health": "N/A",
                    "status_icon": "❌",
                    "cpu": 0.0,
                    "ram": "N/A",
                    "ram_percent": 0.0,
                    "disk": "N/A",
                    "port": "N/A",
                    "uptime": "N/A",
                    "description": description,
                    "external_ip": "N/A",
                }
                field_name: str = self.config.field_name_template.format(
                    **placeholder_values
                )
                field_value: str = f"❌ Error: `{stats_dict['error']}`"
            else:
                placeholder_values = {
                    "alias": alias,
                    "name": container_name,
                    "status": stats_dict.get("status", "N/A"),
                    "health": stats_dict.get("health", "N/A"),
                    "status_icon": stats_dict.get("status_icon", "❌"),
                    "cpu": stats_dict.get("cpu", 0.0),
                    "ram": stats_dict.get("ram", "N/A"),
                    "ram_percent": stats_dict.get("ram_percent", 0.0),
                    "disk": stats_dict.get("disk", "N/A"),
                    "port": stats_dict.get("port", "N/A"),
                    "uptime": stats_dict.get("uptime", "N/A"),
                    "description": description,
                    "external_ip": stats_dict.get("external_ip", "N/A"),
                }
                field_name = self.config.field_name_template.format(
                    **placeholder_values
                )
                field_value = self.config.field_template.format(
                    **placeholder_values
                )

            embed.add_field(name=field_name, value=field_value, inline=False)

        return embed


# ------------------------------
# Discord UI (Restart button / view)
# ------------------------------
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
            await interaction.response.send_message(
                "⛔ Unable to determine your identity.",
                ephemeral=True,
            )
            return

        can_restart, retry_after = self.restart_manager.can_restart(
            self.container_name,
            interaction.user.id,
        )
        if not can_restart:
            if retry_after is None:
                await interaction.response.send_message(
                    "⛔ You are not allowed to restart containers.",
                    ephemeral=True,
                )
            else:
                await interaction.response.send_message(
                    (
                        f"⏳ You’ve hit the restart limit for **{self.alias}**. "
                        f"Try again in {retry_after}s."
                    ),
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
        self.update_message_task = tasks.loop(minutes=1)(
            self._update_message_task
        )
        self.update_external_ip_task = tasks.loop(hours=6)(
            self._update_external_ip_task
        )

    async def _update_message_task(self) -> None:
        if self.message_to_update:
            try:
                embed: discord.Embed = await self.embed_builder.build_embed()
                view = RestartView(self.config, self.restart_manager)
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
            target_channel,
            (discord.TextChannel, discord.Thread, discord.DMChannel),
        ):
            logger.critical("Channel not found or wrong type. Exiting.")
            await self.close()
            sys.exit(1)

        # Try to resume editing an existing message
        if saved_channel_id == self.config.channel_id and saved_message_id:
            try:
                self.message_to_update = await target_channel.fetch_message(  # type: ignore[arg-type]
                    saved_message_id
                )
                logger.info("Resuming updates on message %d", saved_message_id)
            except Exception as fetch_error:
                logger.warning("Failed to fetch saved message: %s", fetch_error)
                self.message_to_update = None

        # If no message to resume, create a new one
        if self.message_to_update is None:
            try:
                initial_embed = await self.embed_builder.build_embed()
                initial_view = RestartView(self.config, self.restart_manager)
                sent_message = await target_channel.send(  # type: ignore[arg-type]
                    embed=initial_embed,
                    view=initial_view,
                )
                self.message_to_update = sent_message
                self.message_state_store.save_message_id(
                    self.config.channel_id, sent_message.id
                )
                logger.info("Created new message %d", sent_message.id)
            except Exception as send_error:
                logger.critical(
                    "Failed to send new message: %s. Exiting.", send_error
                )
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
    stats_service = DockerStatsService(external_ip_service)
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
