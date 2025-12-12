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
# Config directory and env
# ------------------------------
CONFIG_DIR: str = "/config"

dotenv_path: str = os.path.join(CONFIG_DIR, ".env")
logger.debug("Loading environment variables from %s", dotenv_path)
load_dotenv(dotenv_path)


# ------------------------------
# Required environment variables
# ------------------------------
TOKEN: Optional[str] = os.getenv("DISCORD_TOKEN")
if not TOKEN:
    logger.critical("DISCORD_TOKEN is not set. Exiting.")
    sys.exit(1)

CHANNEL_ID_RAW: Optional[str] = os.getenv("DISCORD_CHANNEL_ID")
if not CHANNEL_ID_RAW:
    logger.critical("DISCORD_CHANNEL_ID is not set. Exiting.")
    sys.exit(1)

try:
    CHANNEL_ID: int = int(CHANNEL_ID_RAW)
except ValueError:
    logger.critical("DISCORD_CHANNEL_ID must be an integer. Exiting.")
    sys.exit(1)


# ------------------------------
# Parse containers: CONTAINER_1, CONTAINER_2, ...
# ------------------------------
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

# Sort by numeric suffix
container_env_entries.sort(
    key=lambda entry: int(container_pattern.match(entry[0]).group(1))  # type: ignore[call-arg]
)

CONTAINERS: List[ContainerConfig] = []
for env_var_name, env_var_value in container_env_entries:
    # alias:docker_name:restart_allowed:description
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
    CONTAINERS.append(container_config)
    logger.debug("Parsed container '%s': %s", env_var_name, container_config)

logger.info("Total containers configured: %d", len(CONTAINERS))


# ------------------------------
# Optional settings
# ------------------------------
ALLOWED_USERS_RAW: str = os.getenv("RESTART_ALLOWED_USERS", "")
EMBED_TITLE: str = os.getenv("EMBED_TITLE", "Docker Status")
EMBED_COLOR_RAW: str = os.getenv("EMBED_COLOR", "0x3498DB")
MESSAGE_STATE_FILE: str = os.path.join(
    CONFIG_DIR,
    os.getenv("MESSAGE_STATE_FILE", "message_state.json"),
)

# Max restarts (per period) and period in seconds
RESTART_RATE_LIMIT_COUNT: int = int(os.getenv("RESTART_RATE_LIMIT_COUNT", "2"))
RESTART_RATE_LIMIT_PERIOD: float = float(
    os.getenv("RESTART_RATE_LIMIT_PERIOD", "300")
)

# Field templates
FIELD_TEMPLATE: str = os.getenv(
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

FIELD_NAME_TEMPLATE: str = os.getenv(
    "CONTAINER_FIELD_NAME_TEMPLATE",
    "{alias} (`{name}`)",
).replace("\\n", "\n")

logger.debug("RESTART_ALLOWED_USERS: %s", ALLOWED_USERS_RAW)
logger.debug("EMBED_TITLE: %s", EMBED_TITLE)
logger.debug("FIELD_TEMPLATE: %s", FIELD_TEMPLATE)
logger.debug("FIELD_NAME_TEMPLATE: %s", FIELD_NAME_TEMPLATE)
logger.debug("EMBED_COLOR: %s", EMBED_COLOR_RAW)
logger.debug("MESSAGE_STATE_FILE: %s", MESSAGE_STATE_FILE)


# ------------------------------
# Template validation
# ------------------------------
VALID_PLACEHOLDERS = {
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
    ("FIELD_TEMPLATE", FIELD_TEMPLATE),
    ("FIELD_NAME_TEMPLATE", FIELD_NAME_TEMPLATE),
]:
    used_placeholders = {
        placeholder_name
        for _, placeholder_name, _, _ in template_formatter.parse(template_value)
        if placeholder_name
    }
    invalid_placeholders = used_placeholders - VALID_PLACEHOLDERS
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
# Embed color
# ------------------------------
try:
    EMBED_COLOR: int = int(EMBED_COLOR_RAW, 16)
except ValueError:
    logger.warning(
        "Invalid EMBED_COLOR '%s', defaulting to 0x3498DB",
        EMBED_COLOR_RAW,
    )
    EMBED_COLOR = 0x3498DB


# ------------------------------
# Allowed users for restart
# ------------------------------
ALLOWED_USERS: set[int] = set()
if ALLOWED_USERS_RAW:
    for raw_user_id in ALLOWED_USERS_RAW.split(","):
        stripped_user_id: str = raw_user_id.strip()
        if stripped_user_id.isdigit():
            ALLOWED_USERS.add(int(stripped_user_id))
        else:
            logger.warning(
                "'%s' in RESTART_ALLOWED_USERS is not a valid Discord ID "
                "and will be ignored.",
                stripped_user_id,
            )
logger.debug("ALLOWED_USERS set: %s", ALLOWED_USERS)


# ------------------------------
# Docker client
# ------------------------------
try:
    docker_client: docker.DockerClient = docker.from_env()
    logger.info("Docker client initialized successfully")
except Exception as docker_error:
    logger.critical("Failed to connect to Docker: %s. Exiting.", docker_error)
    sys.exit(1)


# ------------------------------
# Utility helpers
# ------------------------------
def format_size(value_mb: float) -> str:
    if value_mb < 1024:
        return f"{value_mb:.2f} MB"
    value_gb: float = value_mb / 1024
    if value_gb < 1024:
        return f"{value_gb:.2f} GB"
    value_tb: float = value_gb / 1024
    return f"{value_tb:.2f} TB"


def save_message_id(channel_id: int, message_id: int) -> None:
    state: MessageState = {"channel_id": channel_id, "message_id": message_id}
    try:
        with open(MESSAGE_STATE_FILE, "w", encoding="utf-8") as state_file:
            json.dump(state, state_file)
        logger.debug(
            "Saved message state: channel=%d, message=%d",
            channel_id,
            message_id,
        )
    except Exception as save_error:
        logger.warning("Failed to save message state: %s", save_error)


def load_message_id() -> Tuple[Optional[int], Optional[int]]:
    if os.path.exists(MESSAGE_STATE_FILE):
        try:
            with open(MESSAGE_STATE_FILE, "r", encoding="utf-8") as state_file:
                loaded_data_raw: Any = json.load(state_file)
            loaded_state: MessageState = dict(loaded_data_raw)
            logger.debug("Loaded message state: %s", loaded_state)
            return loaded_state.get("channel_id"), loaded_state.get("message_id")
        except Exception as load_error:
            logger.warning("Failed to load message state: %s", load_error)
    return None, None


# ------------------------------
# External IP (blocking -> wrapped in async)
# ------------------------------
EXTERNAL_IP: str = "N/A"


def _fetch_external_ip_sync() -> None:
    global EXTERNAL_IP
    try:
        response = requests.get("https://icanhazip.com", timeout=5)
        if response.status_code == 200:
            EXTERNAL_IP = response.text.strip()
            logger.info("External IP updated: %s", EXTERNAL_IP)
    except Exception as ip_error:
        logger.warning("Failed to fetch external IP: %s", ip_error)


async def fetch_external_ip_async() -> None:
    """Async wrapper for external IP fetch."""
    await asyncio.to_thread(_fetch_external_ip_sync)


# ------------------------------
# Container helpers
# ------------------------------
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


DISK_UPDATE_INTERVAL: float = 6 * 60 * 60  # 6 hours
LAST_DISK_UPDATE: Dict[str, float] = {}  # {container_name: timestamp}
DISK_USAGE_CACHE: Dict[str, float] = {}  # {container_name: mb}


def _get_container_disk_usage_sync(
    container_obj: docker.models.containers.Container,
) -> float:
    """Blocking disk usage computation (to be called from a thread)."""
    total_mb: float = 0.0
    logger.debug("[disk] Calculating disk usage for %s", container_obj.name)

    # 1. Writable layer
    try:
        docker_df: Dict[str, Any] = docker_client.api.df()
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
                volume_mb: float = get_directory_size_mb(host_path)
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
        image_obj = docker_client.images.get(container_obj.image.id)
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


def _get_container_stats_sync(container_name: str) -> StatsDict:
    """
    Blocking container stats gathering.
    This is executed in a thread via asyncio.to_thread.
    """
    try:
        container_obj: docker.models.containers.Container = (
            docker_client.containers.get(container_name)
        )
        stats_stream = container_obj.stats(decode=True)

        # CPU usage: sample over ~1s
        first_stats: Dict[str, Any] = next(stats_stream)
        time.sleep(1)
        second_stats: Dict[str, Any] = next(stats_stream)
        cpu_percent: float = calculate_cpu_percent_from_stats(
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
            container_name not in LAST_DISK_UPDATE
            or current_time - LAST_DISK_UPDATE[container_name] > DISK_UPDATE_INTERVAL
        ):
            logger.debug(
                "[disk:%s] Refreshing cached disk usage...",
                container_name,
            )
            disk_usage_mb: float = _get_container_disk_usage_sync(container_obj)
            DISK_USAGE_CACHE[container_name] = disk_usage_mb
            LAST_DISK_UPDATE[container_name] = current_time
        else:
            disk_usage_mb = DISK_USAGE_CACHE.get(container_name, 0.0)
            cache_age_seconds: float = (
                current_time - LAST_DISK_UPDATE[container_name]
            )
            logger.debug(
                "[disk:%s] Using cached disk usage (%.0fs old)",
                container_name,
                cache_age_seconds,
            )

        host_port: str = get_first_port(container_obj)
        uptime_str: str = format_uptime(container_obj)
        external_ip: str = EXTERNAL_IP

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
            "ram": format_size(mem_usage_mb),
            "ram_percent": mem_percent,
            "disk": format_size(disk_usage_mb),
            "port": host_port,
            "uptime": uptime_str,
            "external_ip": external_ip,
        }
        return result
    except Exception as stats_error:
        logger.error("Failed to get stats for %s: %s", container_name, stats_error)
        return {"error": str(stats_error)}


async def get_container_stats(container_name: str) -> StatsDict:
    """Async wrapper for container stats using a thread executor."""
    return await asyncio.to_thread(_get_container_stats_sync, container_name)


# ------------------------------
# Embed generation (async)
# ------------------------------
async def generate_embed() -> discord.Embed:
    """Build a Discord embed with container stats gathered asynchronously."""
    embed: discord.Embed = discord.Embed(
        title=EMBED_TITLE,
        color=EMBED_COLOR,
        timestamp=datetime.utcnow(),
    )

    # Gather stats concurrently for all containers
    stats_tasks: List[asyncio.Task[StatsDict]] = [
        asyncio.create_task(get_container_stats(container_cfg["name"]))
        for container_cfg in CONTAINERS
    ]
    stats_results: List[StatsDict] = await asyncio.gather(
        *stats_tasks, return_exceptions=False
    )

    for container_cfg, stats_dict in zip(CONTAINERS, stats_results):
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
            field_name: str = FIELD_NAME_TEMPLATE.format(**placeholder_values)
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
            field_name = FIELD_NAME_TEMPLATE.format(**placeholder_values)
            field_value = FIELD_TEMPLATE.format(**placeholder_values)

        embed.add_field(name=field_name, value=field_value, inline=False)

    return embed


# ------------------------------
# Discord Buttons and Views
# ------------------------------
# { container_name: [list of restart timestamps] }
restart_timestamps: Dict[str, List[float]] = {}


class RestartButton(Button):
    def __init__(self, alias: str, container_name: str, restart_allowed: bool) -> None:
        super().__init__(
            label=f"Restart {alias}",
            style=discord.ButtonStyle.danger,
            disabled=not restart_allowed,
        )
        self.container_name: str = container_name
        self.alias: str = alias
        self.restart_allowed: bool = restart_allowed

    async def callback(self, interaction: discord.Interaction) -> None:  # type: ignore[override]
        if (
            interaction.user is None
            or interaction.user.id not in ALLOWED_USERS  # type: ignore[union-attr]
        ):
            await interaction.response.send_message(
                "⛔ You are not allowed to restart containers.",
                ephemeral=True,
            )
            return

        current_time: float = time.time()
        container_history: List[float] = restart_timestamps.setdefault(
            self.container_name,
            [],
        )

        # Clean up old timestamps
        filtered_history: List[float] = [
            timestamp
            for timestamp in container_history
            if current_time - timestamp < RESTART_RATE_LIMIT_PERIOD
        ]
        restart_timestamps[self.container_name] = filtered_history

        if len(filtered_history) >= RESTART_RATE_LIMIT_COUNT:
            first_timestamp: float = filtered_history[0]
            retry_after: float = RESTART_RATE_LIMIT_PERIOD - (
                current_time - first_timestamp
            )
            await interaction.response.send_message(
                (
                    f"⏳ You’ve hit the restart limit for **{self.alias}**. "
                    f"Try again in {int(retry_after)}s."
                ),
                ephemeral=True,
            )
            return

        # Record this restart attempt
        filtered_history.append(current_time)
        restart_timestamps[self.container_name] = filtered_history

        # Perform the restart in a thread (since docker is blocking)
        await interaction.response.defer(ephemeral=True)
        try:
            def _restart_container() -> None:
                container_obj: docker.models.containers.Container = (
                    docker_client.containers.get(self.container_name)
                )
                container_obj.restart()

            await asyncio.to_thread(_restart_container)
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
    def __init__(self) -> None:
        super().__init__(timeout=None)
        for container_cfg in CONTAINERS:
            self.add_item(
                RestartButton(
                    alias=container_cfg["alias"],
                    container_name=container_cfg["name"],
                    restart_allowed=container_cfg["restart_allowed"],
                )
            )


# ------------------------------
# Discord Bot
# ------------------------------
intents: discord.Intents = discord.Intents.default()
bot: commands.Bot = commands.Bot(command_prefix="!", intents=intents)
message_to_update: Optional[discord.Message] = None


@tasks.loop(minutes=1)
async def update_message() -> None:
    """Periodically refresh the embed with updated stats."""
    global message_to_update
    if message_to_update:
        try:
            embed: discord.Embed = await generate_embed()
            await message_to_update.edit(embed=embed, view=RestartView())
            logger.debug("Updated message %d", message_to_update.id)
        except Exception as update_error:
            logger.warning("Failed to update message: %s", update_error)


@tasks.loop(hours=6)
async def update_external_ip() -> None:
    """Periodically refresh external IP in the background."""
    logger.info("Updating external IP...")
    await fetch_external_ip_async()


@bot.event
async def on_ready() -> None:
    """Initialize the message and start background tasks."""
    global message_to_update
    assert bot.user is not None
    logger.info("Logged in as %s", bot.user)

    saved_channel_id, saved_message_id = load_message_id()
    target_channel = bot.get_channel(CHANNEL_ID)
    if target_channel is None or not isinstance(
        target_channel,
        (discord.TextChannel, discord.Thread, discord.DMChannel),
    ):
        logger.critical("Channel not found or wrong type. Exiting.")
        await bot.close()
        sys.exit(1)

    # Try to resume editing an existing message
    if saved_channel_id == CHANNEL_ID and saved_message_id:
        try:
            message_to_update = await target_channel.fetch_message(  # type: ignore[arg-type]
                saved_message_id
            )
            logger.info("Resuming updates on message %d", saved_message_id)
        except Exception as fetch_error:
            logger.warning("Failed to fetch saved message: %s", fetch_error)
            message_to_update = None

    # If no message to resume, create a new one
    if message_to_update is None:
        try:
            initial_embed = await generate_embed()
            initial_view = RestartView()
            sent_message = await target_channel.send(  # type: ignore[arg-type]
                embed=initial_embed,
                view=initial_view,
            )
            message_to_update = sent_message
            save_message_id(CHANNEL_ID, sent_message.id)
            logger.info("Created new message %d", sent_message.id)
        except Exception as send_error:
            logger.critical("Failed to send new message: %s. Exiting.", send_error)
            await bot.close()
            sys.exit(1)

    update_message.start()
    update_external_ip.start()


bot.run(TOKEN)
