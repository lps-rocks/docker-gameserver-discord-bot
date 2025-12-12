import os
import re
import json
import string
import sys
import requests
import logging
import discord
from discord.ext import commands, tasks
from discord.ui import View, Button
from dotenv import load_dotenv
import docker
from datetime import datetime, timezone
import time

# ------------------------------
# Logging configuration
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
if os.getenv("DEBUG", "0") == "1":
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled")

# ------------------------------
# Config directory
# ------------------------------
CONFIG_DIR = "/config"

# Load .env from /config
dotenv_path = os.path.join(CONFIG_DIR, ".env")
logger.debug(f"Loading environment variables from {dotenv_path}")
load_dotenv(dotenv_path)

# ------------------------------
# Required environment variables
# ------------------------------
TOKEN = os.getenv("DISCORD_TOKEN")
if not TOKEN:
    logger.critical("DISCORD_TOKEN is not set. Exiting.")
    sys.exit(1)

CHANNEL_ID_RAW = os.getenv("DISCORD_CHANNEL_ID")
if not CHANNEL_ID_RAW:
    logger.critical("DISCORD_CHANNEL_ID is not set. Exiting.")
    sys.exit(1)
try:
    CHANNEL_ID = int(CHANNEL_ID_RAW)
except ValueError:
    logger.critical("DISCORD_CHANNEL_ID must be an integer. Exiting.")
    sys.exit(1)

# ------------------------------
# Parse containers using regex CONTAINER_1, CONTAINER_2, etc.
# ------------------------------
container_pattern = re.compile(r"^CONTAINER_(\d+)$")
container_entries = [(key, value) for key, value in os.environ.items() if container_pattern.match(key)]
logger.debug(f"Detected container environment variables: {[key for key, _ in container_entries]}")

if not container_entries:
    logger.critical("No containers configured. Define environment variables like CONTAINER_1, CONTAINER_2, etc. Exiting.")
    sys.exit(1)

# Sort by numeric suffix
container_entries.sort(key=lambda x: int(container_pattern.match(x[0]).group(1)))

CONTAINERS = []
for var_name, entry in container_entries:
    # Split into max 4 parts: alias, docker_name, restart_allowed, description
    parts = entry.split(":", 3)
    
    if len(parts) < 2:
        logger.critical(f"Container entry '{var_name}' must have at least alias and docker_name. Exiting.")
        sys.exit(1)
    
    alias = parts[0].strip()
    container_name = parts[1].strip()
    restart_allowed = len(parts) > 2 and parts[2].strip().lower() == "yes"
    description = parts[3].strip() if len(parts) > 3 else ""

    container_info = {
        "alias": alias,
        "name": container_name,
        "restart_allowed": restart_allowed,
        "description": description
    }
    CONTAINERS.append(container_info)
    logger.debug(f"Parsed container '{var_name}': {container_info}")

logger.info(f"Total containers configured: {len(CONTAINERS)}")

# ------------------------------
# Optional settings
# ------------------------------
ALLOWED_USERS_RAW = os.getenv("RESTART_ALLOWED_USERS", "")
EMBED_TITLE = os.getenv("EMBED_TITLE", "Docker Status")
EMBED_COLOR_RAW = os.getenv("EMBED_COLOR", "0x3498DB")
MESSAGE_STATE_FILE = os.path.join(CONFIG_DIR, os.getenv("MESSAGE_STATE_FILE", "message_state.json"))

# Max restarts (per period) and period in seconds
RESTART_RATE_LIMIT_COUNT = int(os.getenv("RESTART_RATE_LIMIT_COUNT", "2"))
RESTART_RATE_LIMIT_PERIOD = float(os.getenv("RESTART_RATE_LIMIT_PERIOD", "300"))  # default: 1 hr

# Load templates and replace literal \n with actual newlines
FIELD_TEMPLATE = os.getenv(
    "CONTAINER_FIELD_TEMPLATE",
    "**Description:** {description}\\n**Status:** {status}\\n**CPU:** {cpu:.2f}%\\n**RAM:** {ram} ({ram_percent:.2f}%)\\n**Disk:** {disk}\\n**Port:** {port}\\n**Uptime:** {uptime}\\n**Host IP:** {external_ip}"
).replace("\\n", "\n")

FIELD_NAME_TEMPLATE = os.getenv(
    "CONTAINER_FIELD_NAME_TEMPLATE",
    "{alias} (`{name}`)"
).replace("\\n", "\n")

logger.debug(f"RESTART_ALLOWED_USERS: {ALLOWED_USERS_RAW}")
logger.debug(f"EMBED_TITLE: {EMBED_TITLE}")
logger.debug(f"FIELD_TEMPLATE: {FIELD_TEMPLATE}")
logger.debug(f"FIELD_NAME_TEMPLATE: {FIELD_NAME_TEMPLATE}")
logger.debug(f"EMBED_COLOR: {EMBED_COLOR_RAW}")
logger.debug(f"MESSAGE_STATE_FILE: {MESSAGE_STATE_FILE}")

# ------------------------------
# Validate templates
# ------------------------------
VALID_PLACEHOLDERS = {"alias", "name", "status", "cpu", "ram", "ram_percent", "disk",
                      "description", "external_ip", "port", "uptime", "health", "status_icon"}

formatter = string.Formatter()
for template_name, template in [("FIELD_TEMPLATE", FIELD_TEMPLATE), ("FIELD_NAME_TEMPLATE", FIELD_NAME_TEMPLATE)]:
    used_placeholders = {fname for _, fname, _, _ in formatter.parse(template) if fname}
    invalid_placeholders = used_placeholders - VALID_PLACEHOLDERS
    logger.debug(f"{template_name} placeholders detected: {used_placeholders}")
    if invalid_placeholders:
        logger.critical(f"{template_name} contains invalid placeholders: {', '.join(invalid_placeholders)}. Exiting.")
        sys.exit(1)

# ------------------------------
# Embed color
# ------------------------------
try:
    EMBED_COLOR = int(EMBED_COLOR_RAW, 16)
except ValueError:
    logger.warning(f"Invalid EMBED_COLOR '{EMBED_COLOR_RAW}', defaulting to 0x3498DB")
    EMBED_COLOR = 0x3498DB

# ------------------------------
# Allowed users for restart
# ------------------------------
ALLOWED_USERS = set()
if ALLOWED_USERS_RAW:
    for u in ALLOWED_USERS_RAW.split(","):
        u = u.strip()
        if u.isdigit():
            ALLOWED_USERS.add(int(u))
        else:
            logger.warning(f"'{u}' in RESTART_ALLOWED_USERS is not a valid Discord ID and will be ignored.")
logger.debug(f"ALLOWED_USERS set: {ALLOWED_USERS}")

# ------------------------------
# Docker client
# ------------------------------
try:
    client = docker.from_env()
    logger.info("Docker client initialized successfully")
except Exception as e:
    logger.critical(f"Failed to connect to Docker: {e}. Exiting.")
    sys.exit(1)

# ------------------------------
# Helpers
# ------------------------------
def format_size(value_mb: float) -> str:
    if value_mb < 1024:
        return f"{value_mb:.2f} MB"
    value_gb = value_mb / 1024
    if value_gb < 1024:
        return f"{value_gb:.2f} GB"
    value_tb = value_gb / 1024
    return f"{value_tb:.2f} TB"

def save_message_id(channel_id, message_id):
    try:
        with open(MESSAGE_STATE_FILE, "w") as f:
            json.dump({"channel_id": channel_id, "message_id": message_id}, f)
        logger.debug(f"Saved message state: channel={channel_id}, message={message_id}")
    except Exception as e:
        logger.warning(f"Failed to save message state: {e}")

def load_message_id():
    if os.path.exists(MESSAGE_STATE_FILE):
        try:
            with open(MESSAGE_STATE_FILE, "r") as f:
                data = json.load(f)
                logger.debug(f"Loaded message state: {data}")
                return data.get("channel_id"), data.get("message_id")
        except Exception as e:
            logger.warning(f"Failed to load message state: {e}")
    return None, None

# ------------------------------
# External IP
# ------------------------------
EXTERNAL_IP = "N/A"
def fetch_external_ip():
    global EXTERNAL_IP
    try:
        response = requests.get("https://icanhazip.com", timeout=5)
        if response.status_code == 200:
            EXTERNAL_IP = response.text.strip()
            logger.info(f"External IP updated: {EXTERNAL_IP}")
    except Exception as e:
        logger.warning(f"Failed to fetch external IP: {e}")

fetch_external_ip()

# ------------------------------
# Container helpers
# ------------------------------
def format_uptime(container):
    try:
        started_at = container.attrs["State"]["StartedAt"]
        start_time = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - start_time
        days, seconds = delta.days, delta.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{days}d {hours}h {minutes}m"
    except Exception:
        return "N/A"

def get_first_port(container):
    try:
        ports = container.attrs.get("NetworkSettings", {}).get("Ports", {})
        for container_port, mappings in ports.items():
            if mappings:
                host_port = mappings[0].get("HostPort")
                if host_port:
                    return host_port
    except Exception:
        return "N/A"
    return "N/A"

def calculate_cpu_percent_from_stats(stats1, stats2):
    try:
        cpu_delta = stats2["cpu_stats"]["cpu_usage"]["total_usage"] - stats1["cpu_stats"]["cpu_usage"]["total_usage"]
        system_delta = stats2["cpu_stats"]["system_cpu_usage"] - stats1["cpu_stats"].get("system_cpu_usage", 0)
        percpu_count = stats1["cpu_stats"].get('online_cpus' , len(stats2["cpu_stats"]["cpu_usage"].get("percpu_usage", [])))
        if system_delta > 0 and cpu_delta > 0:
            return (cpu_delta / system_delta) * percpu_count * 100.0
        return 0.0
    except Exception:
        return 0.0

def get_directory_size_mb(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            try:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
            except Exception:
                pass
    return total / (1024**2)

DISK_UPDATE_INTERVAL = 6 * 60 * 60  # 6 hours
LAST_DISK_UPDATE = {}  # {container_name: timestamp}
DISK_USAGE_CACHE = {}  # {container_name: mb}

def get_container_disk_usage(container):
    total_mb = 0

    # 1. Writable layer
    try:
        df_containers = client.api.df()["Containers"]
        for c in df_containers:
            if c["Id"].startswith(container.id):
                total_mb += c.get("SizeRw", 0) / (1024**2)
                break
    except Exception as e:
        logger.warning(f"Failed to get writable layer size for {container.name}: {e}")

    # 2. Mounted volumes
    try:
        mounts = container.attrs.get("Mounts", [])
        for mount in mounts:
            host_path = mount.get("Source")
            if host_path and os.path.exists(host_path):
                total_mb += get_directory_size_mb(host_path)
    except Exception as e:
        logger.warning(f"Failed to get volumes size for {container.name}: {e}")

    # 3. Image size
    try:
        image = client.images.get(container.image.id)
        total_mb += image.attrs.get("Size", 0) / (1024**2)
    except Exception as e:
        logger.warning(f"Failed to get image size for {container.name}: {e}")

    return total_mb


def get_container_stats(container_name):
    try:
        container = client.containers.get(container_name)  # high-level object
        stats = container.stats(decode=True)
        # ------------------------------
        # CPU Usage
        # ------------------------------
        stats1 = next(stats)
        time.sleep(1)
        stats2 = next(stats)
        cpu_percent = calculate_cpu_percent_from_stats(stats1, stats2)
        
        # ------------------------------
        # RAM Usage
        # ------------------------------
        mem_usage_mb = stats1["memory_stats"]["usage"] / (1024**2)
        mem_limit_mb = stats1["memory_stats"]["limit"] / (1024**2)
        mem_percent = (mem_usage_mb / mem_limit_mb * 100) if mem_limit_mb > 0 else 0

        # ------------------------------
        # Disk usage (refresh every 6 hours)
        # ------------------------------
        now = time.time()
        if (container_name not in LAST_DISK_UPDATE or 
            now - LAST_DISK_UPDATE[container_name] > DISK_UPDATE_INTERVAL):
            
            disk_usage_mb = get_container_disk_usage(container)
            DISK_USAGE_CACHE[container_name] = disk_usage_mb
            LAST_DISK_UPDATE[container_name] = now
        else:
            disk_usage_mb = DISK_USAGE_CACHE.get(container_name, 0)
        
        port = get_first_port(container)
        uptime = format_uptime(container)
        external_ip = EXTERNAL_IP

        health = container.attrs["State"].get("Health", {}).get("Status", "")
        status = container.status
        
        if status == "running" and health == "healthy":
            status_icon = "🟢"
        elif status == 'exited' or (status == "running" and health == "unhealthy"):
            status_icon = "🔴"
        elif status in ['starting', 'restarting'] or (status == "running" and health == "starting"):
            status_icon = "🟠"
        elif status == 'paused':
            status_icon = "🟡"
        elif status == 'dead':
            status_icon = "❌"
        else:
            status_icon = "❓"

        return {
            "status": status,
            "health": health,
            "status_icon": status_icon,
            "cpu": cpu_percent,
            "ram": format_size(mem_usage_mb),
            "ram_percent": mem_percent,
            "disk": format_size(disk_usage_mb),
            "port": port,
            "uptime": uptime,
            "external_ip": external_ip
        }
    except Exception as e:
        logger.error(f"Failed to get stats for {container_name}: {e}")
        return {"error": str(e)}

# ------------------------------
# Embed generation
# ------------------------------
def generate_embed():
    embed = discord.Embed(title=EMBED_TITLE, color=EMBED_COLOR, timestamp=datetime.utcnow())
    for c in CONTAINERS:
        alias = c["alias"]
        name = c["name"]
        description = c["description"]
        stats = get_container_stats(name)

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
            "external_ip": stats.get("external_ip", "N/A")
        }

        if "error" in stats:
            field_value = f"❌ Error: `{stats['error']}`"
        else:
            field_value = FIELD_TEMPLATE.format(**placeholders)

        field_name = FIELD_NAME_TEMPLATE.format(**placeholders)
        embed.add_field(name=field_name, value=field_value, inline=False)
    return embed

# ------------------------------
# Discord buttons
# ------------------------------
# { container_name: [list of restart timestamps] }
restart_timestamps = {}

class RestartButton(Button):
    def __init__(self, alias, container, restart_allowed):
        super().__init__(label=f"Restart {alias}", style=discord.ButtonStyle.danger, disabled=not restart_allowed)
        self.container = container
        self.alias = alias
        self.restart_allowed = restart_allowed

    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id not in ALLOWED_USERS:
            await interaction.response.send_message("⛔ You are not allowed to restart containers.", ephemeral=True)
            return

        # get history for this container
        now = time.time()
        history = restart_timestamps.setdefault(self.container, [])

        # clean up old timestamps
        history = [t for t in history if now - t < RESTART_RATE_LIMIT_PERIOD]
        restart_timestamps[self.container] = history

        # check if we have hit the limit
        if len(history) >= RESTART_RATE_LIMIT_COUNT:
            # calculate wait remaining
            first = history[0]
            retry_after = RESTART_RATE_LIMIT_PERIOD - (now - first)
            await interaction.response.send_message(
                f"⏳ You’ve hit the restart limit for **{self.alias}**. Try again in {int(retry_after)}s.",
                ephemeral=True
            )
            return

        # record this restart attempt
        history.append(now)
        restart_timestamps[self.container] = history

        # perform the restart
        await interaction.response.defer(ephemeral=True)
        try:
            cont = client.containers.get(self.container)
            cont.restart()
            await interaction.followup.send(f"✅ Restarted **{self.alias}** (`{self.container}`).", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to restart **{self.alias}**:\n`{e}`", ephemeral=True)

class RestartView(View):
    def __init__(self):
        super().__init__(timeout=None)
        for c in CONTAINERS:
            self.add_item(RestartButton(alias=c["alias"], container=c["name"], restart_allowed=c["restart_allowed"]))

# ------------------------------
# Discord Bot
# ------------------------------
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)
message_to_update = None

@tasks.loop(minutes=1)
async def update_message():
    if message_to_update:
        embed = generate_embed()
        try:
            await message_to_update.edit(embed=embed, view=RestartView())
            logger.debug(f"Updated message {message_to_update.id}")
        except Exception as e:
            logger.warning(f"Failed to update message: {e}")

@tasks.loop(hours=6)
async def update_external_ip():
    logger.info("Updating external IP...")
    fetch_external_ip()

@bot.event
async def on_ready():
    global message_to_update
    logger.info(f"Logged in as {bot.user}")

    saved_channel_id, saved_message_id = load_message_id()
    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        logger.critical("Channel not found. Exiting.")
        sys.exit(1)

    try:
        if saved_channel_id == CHANNEL_ID and saved_message_id:
            message_to_update = await channel.fetch_message(saved_message_id)
            logger.info(f"Resuming updates on message {saved_message_id}")
    except Exception as e:
        logger.warning(f"Failed to fetch saved message: {e}")
        message_to_update = None

    if not message_to_update:
        embed = generate_embed()
        view = RestartView()
        try:
            message_to_update = await channel.send(embed=embed, view=view)
            save_message_id(CHANNEL_ID, message_to_update.id)
            logger.info(f"Created new message {message_to_update.id}")
        except Exception as e:
            logger.critical(f"Failed to send new message: {e}. Exiting.")
            sys.exit(1)

    update_message.start()
    update_external_ip.start()

bot.run(TOKEN)
