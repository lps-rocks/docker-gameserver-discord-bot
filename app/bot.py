import os
import re
import json
import string
import sys
import logging
import asyncio
import time
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
)

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
    datefmt='%Y-%m-%d %H:%M:%S'
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
logger.debug(f"Loading environment variables from {dotenv_path}")
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
container_entries: List[Tuple[str, str]] = [
    (key, value)
    for key, value in os.environ.items()
    if container_pattern.match(key)
]
logger.debug(
    "Detected container environment variables: %s",
    [key for key, _ in container_entries],
)

if not container_entries:
    logger.critical(
        "No containers configured. Define environment variables like CONTAINER_1, "
        "CONTAINER_2, etc. Exiting."
    )
    sys.exit(1)

# Sort by numeric suffix
container_entries.sort(key=lambda x: int(container_pattern.match(x[0]).group(1)))  # type: ignore[call-arg]

CONTAINERS: List[ContainerConfig] = []
for var_name, entry in container_entries:
    # alias:docker_name:restart_allowed:description
    parts: List[str] = entry.split(":", 3)
    if len(parts) < 2:
        logger.critical(
            "Container entry '%s' must have at least alias and docker_name. Exiting.",
            var_name,
        )
        sys.exit(1)

    alias: str = parts[0].strip()
    container_name: str = parts[1].strip()
    restart_allowed: bool = len(parts) > 2 and parts[2].strip().lower() == "yes"
    description: str = parts[3].strip() if len(parts) > 3 else ""

    container_info: ContainerConfig = {
        "alias": alias,
        "name": container_name,
        "restart_allowed": restart_allowed,
        "description": description,
    }
    CONTAINERS.append(container_info)
    logger.debug("Parsed container '%s': %s", var_name, container_info)

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

formatter: string.Formatter = string.Formatter()
for template_name, template in [
    ("FIELD_TEMPLATE", FIELD_TEMPLATE),
    ("FIELD_NAME_TEMPLATE", FIELD_NAME_TEMPLATE),
]:
    used_placeholders = {
        fname for _, fname, _, _ in formatter.parse(template) if fname
    }
    invalid_placeholders = used_placeholders - VALID_PLACEHOLDERS
    logger.debug("%s placeholders detected: %s", template_name, used_placeholders)
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
        "Invalid EMBED_COLOR '%s', defaulting to 0x3498DB", EMBED_COLOR_RAW
    )
    EMBED_COLOR = 0x3498DB

# ------------------------------
# Allowed users for restart
# ------------------------------
ALLOWED_USERS: set[int] = set()
if ALLOWED_USERS_RAW:
    for u in ALLOWED_USERS_RAW.split(","):
        u = u.strip()
        if u.isdigit():
            ALLOWED_USERS.add(int(u))
        else:
            logger.warning(
                "'%s' in RESTART_ALLOWED_USERS is not a valid Discord ID and will "
                "be ignored.",
                u,
            )
logger.debug("ALLOWED_USERS set: %s", ALLOWED_USERS)

# ------------------------------
# Docker client
# ------------------------------
try:
    client: docker.DockerClient = docker.from_env()
    logger.info("Docker client initialized successfully")
except Exception as e:
    logger.critical("Failed to connect to Docker: %s. Exiting.", e)
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
        with open(MESSAGE_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)
        logger.debug(
            "Saved message state: channel=%d, message=%d",
            channel_id,
            message_id,
        )
    except Exception as e:
        logger.warning("Failed to save message state: %s", e)


def load_message_id() -> Tuple[Optional[int], Optional[int]]:
    if os.path.exists(MESSAGE_STATE_FILE):
        try:
            with open(MESSAGE_STATE_FILE, "r", encoding="utf-8") as f:
                data_raw: Any = json.load(f)
            data: MessageState = dict(data_raw)
            logger.debug("Loaded message state: %s", data)
            return data.get("channel_id"), data.get("message_id")
        except Exception as e:
            logger.warning("Failed to load message state: %s", e)
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
    except Exception as e:
        logger.warning("Failed to fetch external IP: %s", e)


async def fetch_external_ip_async() -> None:
    """Async wrapper for external IP fetch."""
    await asyncio.to_thread(_fetch_external_ip_sync)

# ------------------------------
# Container helpers
# ------------------------------
def format_uptime(container: docker.models.containers.Container) -> str:
    try:
        started_at: str = container.attrs["State"]["StartedAt"]
        start_time: datetime = datetime.fromisoformat(
            started_at.replace("Z", "+00:00")
        )
        delta = datetime.now(timezone.utc) - start_time
        days: int = delta.days
        seconds: int = delta.seconds
        hours: int = seconds // 3600
        minutes: int = (seconds % 3600) // 60
        return f"{days}d {hours}h {minutes}m"
    except Exception:
        return "N/A"


def get_first_port(container: docker.models.containers.Container) -> str:
    try:
        ports: Dict[str, Any] = container.attrs.get(
            "NetworkSettings", {}
        ).get("Ports", {})
        for _, mappings in ports.items():
            if mappings:
                host_port: Optional[str] = mappings[0].get("HostPort")
                if host_port:
                    return host_port
    except Exception:
        return "N/A"
    return "N/A"


def calculate_cpu_percent_from_stats(stats1: Dict[str, Any], stats2: Dict[str, Any]) -> float:
    try:
        cpu_delta: int = (
            stats2["cpu_stats"]["cpu_usage"]["total_usage"]
            - stats1["cpu_stats"]["cpu_usage"]["total_usage"]
        )
        system_delta: int = (
            stats2["cpu_stats"]["system_cpu_usage"]
            - stats1["cpu_stats"].get("system_cpu_usage", 0)
        )
        percpu_count: int = stats1["cpu_stats"].get(
            "online_cpus",
            len(stats2["cpu_stats"]["cpu_usage"].get("percpu_usage", [])),
        )
        if system_delta > 0 and cpu_delta > 0:
            return (cpu_delta / system_delta) * percpu_count * 100.0
        return 0.0
    except Exception:
        return 0.0


def get_directory_size_mb(path: str) -> float:
    """
    Return the *actual* disk usage in MB for a directory tree.

    Uses st_blocks * 512 (POSIX block size) when available, which reflects
    allocated space on disk rather than just logical file size (st_size).
    """
    total_bytes: int = 0

    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            try:
                fp: str = os.path.join(dirpath, f)
                st = os.stat(fp, follow_symlinks=False)

                # Prefer actual allocated size (blocks * 512 bytes) if present.
                if hasattr(st, "st_blocks") and st.st_blocks is not None:
                    total_bytes += st.st_blocks * 512
                else:
                    # Fallback to logical size if st_blocks is missing
                    total_bytes += st.st_size
            except Exception:
                # Skip files that can't be accessed
                pass

    return total_bytes / (1024 ** 2)


DISK_UPDATE_INTERVAL: float = 6 * 60 * 60  # 6 hours
LAST_DISK_UPDATE: Dict[str, float] = {}  # {container_name: timestamp}
DISK_USAGE_CACHE: Dict[str, float] = {}  # {container_name: mb}


def _get_container_disk_usage_sync(
    container: docker.models.containers.Container,
) -> float:
    """Blocking disk usage computation (to be called from a thread)."""
    total_mb: float = 0.0
    logger.debug("[disk] Calculating disk usage for %s", container.name)

    # 1. Writable layer
    try:
        df_containers: List[Dict[str, Any]] = client.api.df()["Containers"]
        for c in df_containers:
            if c["Id"].startswith(container.id):
                layer_mb: float = c.get("SizeRw", 0) / (1024**2)
                logger.debug(
                    "[disk:%s] Writable layer: %.2f MB",
                    container.name,
                    layer_mb,
                )
                total_mb += layer_mb
                break
    except Exception as e:
        logger.warning(
            "[disk:%s] Failed to get writable layer size: %s",
            container.name,
            e,
        )

    # 2. Mounted volumes
    try:
        mounts: List[Dict[str, Any]] = container.attrs.get("Mounts", [])
        for mount in mounts:
            host_path: Optional[str] = mount.get("Source")
            if host_path and os.path.exists(host_path):
                vol_mb: float = get_directory_size_mb(host_path)
                logger.debug(
                    "[disk:%s] Volume %s: %.2f MB",
                    container.name,
                    host_path,
                    vol_mb,
                )
                total_mb += vol_mb
    except Exception as e:
        logger.warning(
            "[disk:%s] Failed to get volumes size: %s", container.name, e
        )

    # 3. Image size
    try:
        image = client.images.get(container.image.id)
        image_mb: float = image.attrs.get("Size", 0) / (1024**2)
        logger.debug(
            "[disk:%s] Image size: %.2f MB",
            container.name,
            image_mb,
        )
        total_mb += image_mb
    except Exception as e:
        logger.warning(
            "[disk:%s] Failed to get image size: %s", container.name, e
        )

    logger.debug(
        "[disk:%s] Total calculated: %.2f MB",
        container.name,
        total_mb,
    )
    return total_mb


def _get_container_stats_sync(container_name: str) -> StatsDict:
    """
    Blocking container stats gathering.
    This is executed in a thread via asyncio.to_thread.
    """
    try:
        container: docker.models.containers.Container = client.containers.get(
            container_name
        )
        stats_stream = container.stats(decode=True)

        # CPU usage: sample over ~1s
        stats1: Dict[str, Any] = next(stats_stream)
        time.sleep(1)
        stats2: Dict[str, Any] = next(stats_stream)
        cpu_percent: float = calculate_cpu_percent_from_stats(stats1, stats2)

        # RAM usage
        mem_usage_mb: float = stats1["memory_stats"]["usage"] / (1024**2)
        mem_limit_mb: float = stats1["memory_stats"]["limit"] / (1024**2)
        mem_percent: float = (
            (mem_usage_mb / mem_limit_mb * 100) if mem_limit_mb > 0 else 0.0
        )

        # Disk usage (refresh every 6 hours)
        now: float = time.time()
        if (
            container_name not in LAST_DISK_UPDATE
            or now - LAST_DISK_UPDATE[container_name] > DISK_UPDATE_INTERVAL
        ):
            logger.debug("[disk:%s] Refreshing cached disk usage...", container_name)
            disk_usage_mb: float = _get_container_disk_usage_sync(container)
            DISK_USAGE_CACHE[container_name] = disk_usage_mb
            LAST_DISK_UPDATE[container_name] = now
        else:
            disk_usage_mb = DISK_USAGE_CACHE.get(container_name, 0.0)
            age: float = now - LAST_DISK_UPDATE[container_name]
            logger.debug(
                "[disk:%s] Using cached disk usage (%.0fs old)",
                container_name,
                age,
            )

        port: str = get_first_port(container)
        uptime: str = format_uptime(container)
        external_ip: str = EXTERNAL_IP

        health: str = container.attrs["State"].get("Health", {}).get("Status", "")
        status: str = container.status

        if status == "running" and health == "healthy":
            status_icon: str = "🟢"
        elif status == "exited" or (status == "running" and health == "unhealthy"):
            status_icon = "🔴"
        elif status in ["starting", "restarting"] or (
            status == "running" and health == "starting"
        ):
            status_icon = "🟠"
        elif status == "paused":
            status_icon = "🟡"
        elif status == "dead":
            status_icon = "❌"
        else:
            status_icon = "❓"

        result: StatsDict = {
            "status": status,
            "health": health,
            "status_icon": status_icon,
            "cpu": cpu_percent,
            "ram": format_size(mem_usage_mb),
            "ram_percent": mem_percent,
            "disk": format_size(disk_usage_mb),
            "port": port,
            "uptime": uptime,
            "external_ip": external_ip,
        }
        return result
    except Exception as e:
        logger.error("Failed to get stats for %s: %s", container_name, e)
        return {"error": str(e)}


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
    stats_tasks: List[asyncio.Future[StatsDict]] = [
        asyncio.ensure_future(get_container_stats(c["name"])) for c in CONTAINERS
    ]
    stats_results: List[StatsDict] = await asyncio.gather(
        *stats_tasks, return_exceptions=False
    )

    for container_cfg, stats in zip(CONTAINERS, stats_results):
        alias: str = container_cfg["alias"]
        name: str = container_cfg["name"]
        description: str = container_cfg["description"]

        if "error" in stats:
            placeholders: Dict[str, Any] = {
                "alias": alias,
                "name": name,
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
            field_name: str = FIELD_NAME_TEMPLATE.format(**placeholders)
            field_value: str = f"❌ Error: `{stats['error']}`"
        else:
            placeholders = {
                "alias": alias,
                "name": name,
                "status": stats.get("status", "N/A"),
                "health": stats.get("health", "N/A"),
                "status_icon": stats.get("status_icon", "❌"),
                "cpu": stats.get("cpu", 0.0),
                "ram": stats.get("ram", "N/A"),
                "ram_percent": stats.get("ram_percent", 0.0),
                "disk": stats.get("disk", "N/A"),
                "port": stats.get("port", "N/A"),
                "uptime": stats.get("uptime", "N/A"),
                "description": description,
                "external_ip": stats.get("external_ip", "N/A"),
            }
            field_name = FIELD_NAME_TEMPLATE.format(**placeholders)
            field_value = FIELD_TEMPLATE.format(**placeholders)

        embed.add_field(name=field_name, value=field_value, inline=False)

    return embed

# ------------------------------
# Discord Buttons and Views
# ------------------------------
# { container_name: [list of restart timestamps] }
restart_timestamps: Dict[str, List[float]] = {}


class RestartButton(Button):
    def __init__(self, alias: str, container: str, restart_allowed: bool) -> None:
        super().__init__(
            label=f"Restart {alias}",
            style=discord.ButtonStyle.danger,
            disabled=not restart_allowed,
        )
        self.container: str = container
        self.alias: str = alias
        self.restart_allowed: bool = restart_allowed

    async def callback(self, interaction: discord.Interaction) -> None:  # type: ignore[override]
        if interaction.user is None or interaction.user.id not in ALLOWED_USERS:  # type: ignore[union-attr]
            await interaction.response.send_message(
                "⛔ You are not allowed to restart containers.",
                ephemeral=True,
            )
            return

        now: float = time.time()
        history: List[float] = restart_timestamps.setdefault(self.container, [])

        # Clean up old timestamps
        history = [t for t in history if now - t < RESTART_RATE_LIMIT_PERIOD]
        restart_timestamps[self.container] = history

        if len(history) >= RESTART_RATE_LIMIT_COUNT:
            first: float = history[0]
            retry_after: float = RESTART_RATE_LIMIT_PERIOD - (now - first)
            await interaction.response.send_message(
                (
                    f"⏳ You’ve hit the restart limit for **{self.alias}**. "
                    f"Try again in {int(retry_after)}s."
                ),
                ephemeral=True,
            )
            return

        # Record this restart attempt
        history.append(now)
        restart_timestamps[self.container] = history

        # Perform the restart in a thread (since docker is blocking)
        await interaction.response.defer(ephemeral=True)
        try:
            def _restart_container() -> None:
                cont: docker.models.containers.Container = client.containers.get(
                    self.container
                )
                cont.restart()

            await asyncio.to_thread(_restart_container)
            await interaction.followup.send(
                f"✅ Restarted **{self.alias}** (`{self.container}`).",
                ephemeral=True,
            )
        except Exception as e:
            await interaction.followup.send(
                f"❌ Failed to restart **{self.alias}**:\n`{e}`",
                ephemeral=True,
            )


class RestartView(View):
    def __init__(self) -> None:
        super().__init__(timeout=None)
        for c in CONTAINERS:
            self.add_item(
                RestartButton(
                    alias=c["alias"],
                    container=c["name"],
                    restart_allowed=c["restart_allowed"],
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
        except Exception as e:
            logger.warning("Failed to update message: %s", e)


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
    channel = bot.get_channel(CHANNEL_ID)
    if channel is None or not isinstance(channel, (discord.TextChannel, discord.Thread, discord.DMChannel)):
        logger.critical("Channel not found or wrong type. Exiting.")
        await bot.close()
        sys.exit(1)

    # Try to resume editing an existing message
    if saved_channel_id == CHANNEL_ID and saved_message_id:
        try:
            message_to_update = await channel.fetch_message(saved_message_id)  # type: ignore[arg-type]
            logger.info("Resuming updates on message %d", saved_message_id)
        except Exception as e:
            logger.warning("Failed to fetch saved message: %s", e)
            message_to_update = None

    # If no message to resume, create a new one
    if message_to_update is None:
        try:
            embed = await generate_embed()
            view = RestartView()
            message_to_update = await channel.send(embed=embed, view=view)  # type: ignore[arg-type]
            save_message_id(CHANNEL_ID, message_to_update.id)
            logger.info("Created new message %d", message_to_update.id)
        except Exception as e:
            logger.critical("Failed to send new message: %s. Exiting.", e)
            await bot.close()
            sys.exit(1)

    update_message.start()
    update_external_ip.start()


bot.run(TOKEN)
