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
# Logging
# ------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------
# Dotty template rendering
# ------------------------------

class DottyFormatter(string.Formatter):
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__()
        self._dot = dotty(context)

    def get_field(self, field_name, args, kwargs):
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
    def __init__(self, config_dir: str = "/config") -> None:
        load_dotenv(os.path.join(config_dir, ".env"))

        self.token = os.getenv("DISCORD_TOKEN")
        self.channel_id = int(os.getenv("DISCORD_CHANNEL_ID", "0"))

        self.embed_title = os.getenv("EMBED_TITLE", "Docker Status")
        self.embed_color = int(os.getenv("EMBED_COLOR", "0x3498DB"), 16)

        self.restart_rate_limit_count = int(os.getenv("RESTART_RATE_LIMIT_COUNT", "2"))
        self.restart_rate_limit_period = float(os.getenv("RESTART_RATE_LIMIT_PERIOD", "300"))

        self.allowed_users = {
            int(u.strip())
            for u in os.getenv("RESTART_ALLOWED_USERS", "").split(",")
            if u.strip().isdigit()
        }

        self.message_state_file = os.path.join(config_dir, "message_state.json")

        self.field_template = os.getenv(
            "CONTAINER_FIELD_TEMPLATE",
            "**Status:** {status_icon} {status}\n"
            "**CPU:** {cpu:.2f}%\n"
            "**RAM:** {ram} ({ram_percent:.2f}%)\n"
            "**Disk:** {disk}\n"
            "**Uptime:** {uptime}\n"
            "**Host IP:** {external_ip}",
        )

        self.field_name_template = "{alias} (`{name}`)"

        self.containers: List[ContainerConfig] = []
        pattern = re.compile(r"^CONTAINER_(\d+)$")
        for key, val in sorted(os.environ.items()):
            if pattern.match(key):
                a, n, p, qp, r, a2s_e, d = val.split(":", 6)
                self.containers.append({
                    "alias": a,
                    "name": n,
                    "port": p,
                    "query_port": qp,
                    "restart_allowed": r.lower() == "yes",
                    "a2s_enabled": a2s_e.lower() == "yes",
                    "description": d,
                })

# ------------------------------
# Message state
# ------------------------------

class MessageStateStore:
    def __init__(self, path: str) -> None:
        self.path = path

    def save(self, channel_id: int, message_id: int) -> None:
        with open(self.path, "w") as f:
            json.dump({"channel_id": channel_id, "message_id": message_id}, f)

    def load(self) -> Tuple[Optional[int], Optional[int]]:
        if not os.path.exists(self.path):
            return None, None
        with open(self.path) as f:
            data = json.load(f)
            return data.get("channel_id"), data.get("message_id")

# ------------------------------
# External IP
# ------------------------------

class ExternalIPService:
    def __init__(self):
        self.ip = "N/A"

    async def update(self):
        try:
            r = await asyncio.to_thread(requests.get, "https://icanhazip.com", timeout=5)
            if r.status_code == 200:
                self.ip = r.text.strip()
        except Exception:
            pass

# ------------------------------
# Docker stats
# ------------------------------

class DockerStatsService:
    def __init__(self, external_ip: ExternalIPService):
        self.client = docker.from_env()
        self.external_ip = external_ip

    async def get_container_status(self, name: str) -> str:
        try:
            c = await asyncio.to_thread(self.client.containers.get, name)
            return c.status
        except Exception:
            return "unknown"

    async def get_stats(self, name: str) -> StatsDict:
        try:
            c = await asyncio.to_thread(self.client.containers.get, name)
            status = c.status
            icon = "✅" if status == "running" else "⏹️"
            return {
                "status": status,
                "status_icon": icon,
                "cpu": 0.0,
                "ram": "N/A",
                "ram_percent": 0.0,
                "disk": "N/A",
                "uptime": "N/A",
                "external_ip": self.external_ip.ip,
            }
        except Exception as e:
            return {"error": str(e)}

# ------------------------------
# Restart manager
# ------------------------------

class RestartManager:
    def __init__(self, client, allowed, count, period):
        self.client = client
        self.allowed = allowed
        self.count = count
        self.period = period
        self.history: Dict[str, List[float]] = {}

    def can_act(self, name: str, user_id: int):
        if user_id not in self.allowed:
            return False, None
        now = time.time()
        h = [t for t in self.history.get(name, []) if now - t < self.period]
        if len(h) >= self.count:
            return False, int(self.period - (now - h[0]))
        h.append(now)
        self.history[name] = h
        return True, None

    async def start(self, name: str):
        await asyncio.to_thread(lambda: self.client.containers.get(name).start())

    async def stop(self, name: str):
        await asyncio.to_thread(lambda: self.client.containers.get(name).stop())

    async def restart(self, name: str):
        await asyncio.to_thread(lambda: self.client.containers.get(name).restart())

# ------------------------------
# Stop confirmation modal
# ------------------------------

class StopConfirmModal(Modal, title="Confirm Stop"):
    confirm = TextInput(label="Type STOP to confirm", required=True)

    def __init__(self, alias, name, manager):
        super().__init__()
        self.alias = alias
        self.name = name
        self.manager = manager

    async def on_submit(self, interaction: discord.Interaction):
        if self.confirm.value.strip().upper() != "STOP":
            await interaction.response.send_message("❌ Cancelled.", ephemeral=True)
            return
        await self.manager.stop(self.name)
        await interaction.response.send_message(f"🛑 **{self.alias}** stopped.", ephemeral=True)

# ------------------------------
# Buttons
# ------------------------------

class ActionButton(Button):
    def __init__(self, label, style, action, enabled, row, alias, name, manager):
        super().__init__(label=label, style=style, disabled=not enabled, row=row)
        self.action = action
        self.alias = alias
        self.name = name
        self.manager = manager

    async def callback(self, interaction: discord.Interaction):
        ok, wait = self.manager.can_act(self.name, interaction.user.id)
        if not ok:
            await interaction.response.send_message("⏳ Rate limited.", ephemeral=True)
            return

        if self.action == "stop":
            await interaction.response.send_modal(
                StopConfirmModal(self.alias, self.name, self.manager)
            )
            return

        fn = getattr(self.manager, self.action)
        await fn(self.name)
        await interaction.response.send_message(
            f"✅ **{self.alias}** {self.action}ed.",
            ephemeral=True,
        )

# ------------------------------
# View
# ------------------------------

class RestartView(View):
    def __init__(self, config, manager, stats):
        super().__init__(timeout=None)
        self.config = config
        self.manager = manager
        self.stats = stats

    async def setup(self):
        for row, c in enumerate(self.config.containers):
            status = await self.stats.get_container_status(c["name"])
            running = status == "running"

            self.add_item(ActionButton(
                "Start", discord.ButtonStyle.success,
                "start", not running and c["restart_allowed"],
                row, c["alias"], c["name"], self.manager
            ))
            self.add_item(ActionButton(
                "Stop", discord.ButtonStyle.secondary,
                "stop", running and c["restart_allowed"],
                row, c["alias"], c["name"], self.manager
            ))
            self.add_item(ActionButton(
                "Restart", discord.ButtonStyle.danger,
                "restart", running and c["restart_allowed"],
                row, c["alias"], c["name"], self.manager
            ))

# ------------------------------
# Embed builder
# ------------------------------

class EmbedBuilder:
    def __init__(self, config, stats):
        self.config = config
        self.stats = stats

    async def build(self):
        e = discord.Embed(
            title=self.config.embed_title,
            color=self.config.embed_color,
            timestamp=datetime.now(UTC),
        )
        for c in self.config.containers:
            s = await self.stats.get_stats(c["name"])
            ctx = {**s, **c}
            e.add_field(
                name=render_template(self.config.field_name_template, ctx),
                value=render_template(self.config.field_template, ctx),
                inline=False,
            )
        return e

# ------------------------------
# Bot
# ------------------------------

class DockerStatusBot(commands.Bot):
    def __init__(self, config):
        super().__init__(command_prefix="!", intents=discord.Intents.default())
        self.config = config
        self.state = MessageStateStore(config.message_state_file)
        self.external_ip = ExternalIPService()
        self.stats = DockerStatsService(self.external_ip)
        self.manager = RestartManager(
            self.stats.client,
            config.allowed_users,
            config.restart_rate_limit_count,
            config.restart_rate_limit_period,
        )
        self.embed = EmbedBuilder(config, self.stats)
        self.message = None

        self.update_loop = tasks.loop(minutes=1)(self.update_message)

    async def on_ready(self):
        channel = self.get_channel(self.config.channel_id)
        cid, mid = self.state.load()
        if mid:
            self.message = await channel.fetch_message(mid)
        else:
            self.message = await channel.send("Loading…")
            self.state.save(channel.id, self.message.id)

        self.update_loop.start()

    async def update_message(self):
        await self.external_ip.update()
        view = RestartView(self.config, self.manager, self.stats)
        await view.setup()
        await self.message.edit(embed=await self.embed.build(), view=view)

# ------------------------------
# Entrypoint
# ------------------------------

def main():
    config = AppConfig()
    bot = DockerStatusBot(config)
    bot.run(config.token)

if __name__ == "__main__":
    main()
