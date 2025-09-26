import logging
import os

import disnake
from disnake.ext import commands
from dotenv import load_dotenv
from result import Err

from absolute_unit.conversion import try_convert_expression
from absolute_unit.parsing import ureg


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
    interaction: disnake.GuildCommandInteraction[commands.InteractionBot],
    input: str,
    # TODO: converters can be used here
    target: str | None = None,
    verbose: bool = False,
) -> None:
    """
    Convert the input expression.

    Parameters
    ----------
    input:
        The input expression to evaluate and convert.
    target:
        The output unit, infered if not specified.
    verbose:
        Print the intepretation of the parsed expression. Use this if output is unexpected.
    """
    # raise ValueError("a")
    converted_result = try_convert_expression(input, target)
    if isinstance(converted_result, Err):
        return await interaction.send(converted_result.err_value, ephemeral=True)
    (expression, converted) = converted_result.ok_value

    # TODO: move allodis to a bigh "post-process" function
    if converted.units == ureg.foot:
        magnitude = converted.magnitude
        whole = int(magnitude)
        quantity_foot = whole * ureg.foot  # pyright: ignore[reportUnknownVariableType]
        decimal = magnitude - whole
        quantity_inch = decimal * 12 * ureg.inch  # pyright: ignore[reportUnknownVariableType]
        converted_str = f"{quantity_foot:~P} {quantity_inch:.3g~P}"
    else:
        if converted.units == ureg.kph:
            converted = converted.to("km/h")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        converted_str = f"{converted:.3g~P}"

    if verbose:
        output = f"```\n{input}\n```interpreting as\n```\n{expression}\n=\n{converted_str}\n```"
    else:
        output = f"`{input}` = `{converted_str}`"
    _ = await interaction.send(output)


# disnake has incorrect type hints for slash command error callbacks
@convert.error  # pyright: ignore[reportArgumentType, reportUntypedFunctionDecorator]
async def convert_error(
    interaction: disnake.ApplicationCommandInteraction[commands.InteractionBot],
    error: commands.CommandInvokeError,
):
    original = error.original
    original_type = type(original)
    original_message = str(original)
    if original_message:
        msg = f"Error when attempting command:\n`{original_type}: {original_message}`\nThis is a bug."
    else:
        msg = f"Error when attempting command:\n`{original_type}`\nThis is a bug."
    await interaction.send(msg, ephemeral=True)


@bot.event
async def on_ready() -> None:
    print(f"Logged in as {bot.user} (ID: {bot.user.id})\n")


if __name__ == "__main__":
    bot.run(os.getenv("DISCORD_APPLICATION_TOKEN"))
