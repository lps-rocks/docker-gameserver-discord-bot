import os
import re
import json
import string
import shlex
import discord
from discord.ext import commands, tasks
from discord.ui import View, Button
from dotenv import load_dotenv
import docker
import datetime

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()

# Required variables
TOKEN = os.getenv("DISCORD_TOKEN")
if not TOKEN:
    raise ValueError("DISCORD_TOKEN is not set.")

CHANNEL_ID_RAW = os.getenv("DISCORD_CHANNEL_ID")
if not CHANNEL_ID_RAW:
    raise ValueError("DISCORD_CHANNEL_ID is not set.")
try:
    CHANNEL_ID = int(CHANNEL_ID_RAW)
except ValueError:
    raise ValueError("DISCORD_CHANNEL_ID must be an integer.")

# ------------------------------
# Parse containers from CONTAINER_1, CONTAINER_2, etc. using regex
# ------------------------------
container_pattern = re.compile(r"^CONTAINER_(\d+)$")
container_entries = [(key, value) for key, value in os.environ.items() if container_pattern.match(key)]

if not container_entries:
    raise ValueError("No containers configured. Define environment variables like CONTAINER_1, CONTAINER_2, etc.")

# Sort by numeric order
container_entries.sort(key=lambda x: int(container_pattern.match(x[0]).group(1)))

CONTAINERS = []
for var_name, entry in container_entries:
    try:
        parts = shlex.split(entry)
    except ValueError as e:
        raise ValueError(f"Failed to parse container entry '{var_name}': {e}")

    if len(parts) < 2:
        raise ValueError(f"Container entry '{var_name}' must have at least alias and docker_name")

    alias = parts[0]
    container_name = parts[1]
    restart_allowed = len(parts) > 2 and parts[2].strip().lower() == "yes"
    description = parts[3] if len(parts) > 3 else ""

    CONTAINERS.append({
        "alias": alias,
        "name": container_name,
        "restart_allowed": restart_allowed,
        "description": description
    })

# ------------------------------
# Optional variables
# ------------------------------
ALLOWED_USERS_RAW = os.getenv("RESTART_ALLOWED_USERS", "")
EMBED_TITLE = os.getenv("EMBED_TITLE", "Docker Status")
FIELD_TEMPLATE = os.getenv(
    "CONTAINER_FIELD_TEMPLATE",
    "**Description:** {description}\n**Status:** {status}\n**CPU:** {cpu:.2f}%\n**RAM:** {ram} ({ram_percent:.2f}%)\n**Disk:** {disk}"
)
FIELD_NAME_TEMPLATE = os.getenv("CONTAINER_FIELD_NAME_TEMPLATE", "{alias} (`{name}`)")
EMBED_COLOR_RAW = os.getenv("EMBED_COLOR", "0x3498DB")
MESSAGE_STATE_FILE = os.getenv("MESSAGE_STATE_FILE", "message_state.json")

# ------------------------------
# Validate templates
# ------------------------------
VALID_PLACEHOLDERS = {"alias", "name", "status", "cpu", "ram", "ram_percent", "disk", "description"}
formatter = string.Formatter()

used_placeholders = {fname for _, fname, _, _ in formatter.parse(FIELD_TEMPLATE) if fname}
invalid_placeholders = used_placeholders - VALID_PLACEHOLDERS
if invalid_placeholders:
    raise ValueError(f"FIELD_TEMPLATE contains invalid placeholders: {', '.join(invalid_placeholders)}")

used_name_placeholders = {fname for _, fname, _, _ in formatter.parse(FIELD_NAME_TEMPLATE) if fname}
invalid_name_placeholders = used_name_placeholders - VALID_PLACEHOLDERS
if invalid_name_placeholders:
    raise ValueError(f"FIELD_NAME_TEMPLATE contains invalid placeholders: {', '.join(invalid_name_placeholders)}")

# ------------------------------
# Parse embed color
# ------------------------------
try:
    EMBED_COLOR = int(EMBED_COLOR_RAW, 16)
except ValueError:
    print(f"Invalid EMBED_COLOR '{EMBED_COLOR_RAW}', defaulting to 0x3498DB")
    EMBED_COLOR = 0x3498DB

# ------------------------------
# Parse allowed users
# ------------------------------
ALLOWED_USERS = set()
if ALLOWED_USERS_RAW:
    for u in ALLOWED_USERS_RAW.split(","):
        u = u.strip()
        if not u.isdigit():
            print(f"Warning: '{u}' in RESTART_ALLOWED_USERS is not a valid Discord ID and will be ignored.")
            continue
        ALLOWED_USERS.add(int(u))

# ------------------------------
# Docker client
# ------------------------------
try:
    client = docker.from_env()
except Exception as e:
    raise RuntimeError(f"Failed to connect to Docker: {e}")

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
    except Exception as e:
        print(f"Warning: Failed to save message state: {e}")

def load_message_id():
    if os.path.exists(MESSAGE_STATE_FILE):
        try:
            with open(MESSAGE_STATE_FILE, "r") as f:
                data = json.load(f)
                return data.get("channel_id"), data.get("message_id")
        except Exception as e:
            print(f"Warning: Failed to load message state: {e}")
    return None, None

def get_container_stats(container_name):
    try:
        container = client.containers.get(container_name)
        stats = container.stats(stream=False)
        # CPU
        cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - stats["precpu_stats"]["cpu_usage"]["total_usage"]
        system_delta = stats["cpu_stats"]["system_cpu_usage"] - stats["precpu_stats"]["system_cpu_usage"]
        cpu_percent = 0.0
        if system_delta > 0 and cpu_delta > 0:
            cpu_percent = cpu_delta / system_delta * len(stats["cpu_stats"]["cpu_usage"].get("percpu_usage", [])) * 100
        # RAM
        mem_usage_mb = stats["memory_stats"]["usage"] / (1024**2)
        mem_limit_mb = stats["memory_stats"]["limit"] / (1024**2)
        mem_percent = (mem_usage_mb / mem_limit_mb * 100) if mem_limit_mb > 0 else 0
        # Disk
        disk_usage_mb = container.attrs.get("SizeRw", 0) / (1024**2)
        return {"status": container.status, "cpu": cpu_percent, "ram": format_size(mem_usage_mb), "ram_percent": mem_percent, "disk": format_size(disk_usage_mb)}
    except Exception as e:
        return {"error": str(e)}

def generate_embed():
    embed = discord.Embed(title=EMBED_TITLE, color=EMBED_COLOR, timestamp=datetime.datetime.utcnow())
    for c in CONTAINERS:
        alias = c["alias"]
        name = c["name"]
        description = c["description"]
        stats = get_container_stats(name)
        if "error" in stats:
            field_value = f"❌ Error: `{stats['error']}`"
        else:
            field_value = FIELD_TEMPLATE.format(
                alias=alias, name=name, status=stats["status"], cpu=stats["cpu"],
                ram=stats["ram"], ram_percent=stats["ram_percent"], disk=stats["disk"],
                description=description
            )
        field_name = FIELD_NAME_TEMPLATE.format(alias=alias, name=name, description=description)
        embed.add_field(name=field_name, value=field_value, inline=False)
    return embed

# ------------------------------
# Buttons
# ------------------------------
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
        await interaction.response.defer(ephemeral=True)
        try:
            cont = client.containers.get(self.container)
            cont.restart()
            await interaction.followup.send(f"🔁 Restarted **{self.alias}** (`{self.container}`).", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to restart **{self.alias}**\nError: `{e}`", ephemeral=True)

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

@bot.event
async def on_ready():
    global message_to_update
    print(f"Logged in as {bot.user}")

    saved_channel_id, saved_message_id = load_message_id()
    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        print("Error: Channel not found.")
        return

    try:
        if saved_channel_id == CHANNEL_ID and saved_message_id:
            message_to_update = await channel.fetch_message(saved_message_id)
            print(f"Resuming updates on message {saved_message_id}")
    except Exception as e:
        print(f"Warning: Failed to fetch saved message: {e}")
        message_to_update = None

    if not message_to_update:
        embed = generate_embed()
        view = RestartView()
        try:
            message_to_update = await channel.send(embed=embed, view=view)
            save_message_id(CHANNEL_ID, message_to_update.id)
            print(f"Created new message {message_to_update.id}")
        except Exception as e:
            print(f"Error: Failed to send new message: {e}")

    update_message.start()

@tasks.loop(minutes=1)
async def update_message():
    if message_to_update:
        embed = generate_embed()
        try:
            await message_to_update.edit(embed=embed, view=RestartView())
        except Exception as e:
            print(f"Warning: Failed to update message: {e}")

bot.run(TOKEN)
