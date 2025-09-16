import logging
import disnake
import os

from disnake.ext import commands
from pint import PintError
from result import Err, Ok, Result

from absolute_unit import parsing

from dotenv import load_dotenv

_ = load_dotenv()

logger = logging.getLogger("disnake")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename="disnake.log", encoding="utf-8", mode="w")
handler.setFormatter(
    logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")
)
logger.addHandler(handler)

bot = commands.InteractionBot(
    test_guilds=[1417274294773223468],
)


@bot.slash_command()
async def convert(
    interaction: disnake.GuildCommandInteraction[commands.Bot], input: str, target: str
):
    embed = disnake.Embed()
    embed.footer.text = (
        "If you are getting unexpected results, try using explicit operations."
    )
    parsing_result = parsing.parse(input)
    if isinstance(parsing_result, Err):
        errors = parsing_result.err_value
        errors_formatted = parsing.format_errors(errors, len(input))


@bot.event
async def on_ready() -> None:
    print(f"Logged in as {bot.user} (ID: {bot.user.id})\n")


if __name__ == "__main__":
    bot.run(os.getenv("DISCORD_APPLICATION_TOKEN"))
