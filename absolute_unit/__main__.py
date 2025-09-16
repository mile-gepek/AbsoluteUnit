import logging
import disnake
import os

from disnake.ext import commands
from pint.facets.plain import PlainQuantity
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


def convert_unit(
    quantity: PlainQuantity[float], target: str | None = None
) -> PlainQuantity[float]:
    if target is None:
        raise NotImplementedError
    return quantity.to(target)


@bot.slash_command()
async def convert(
    interaction: disnake.GuildCommandInteraction[commands.Bot],
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
        Output the intepretation of the parsed expression. Use this if output is unexpected.
    """
    # TODO: test whether defering is needed.
    await interaction.response.defer()

    # embed = disnake.Embed()
    # embed.footer.text = (
    #     "If you are getting unexpected results, try using explicit operations."
    # )

    parsing_result = parsing.parse(input)
    if isinstance(parsing_result, Err):
        errors = parsing_result.err_value
        errors_formatted = parsing.format_errors(errors, len(input))
        output = f"```\n{input}\n{errors_formatted}\n```"

        _ = await interaction.edit_original_response(output)
        return
    expression = parsing_result.ok_value

    eval_result = expression.evaluate()
    if isinstance(eval_result, Err):
        errors = eval_result.err_value
        errors_formatted = parsing.format_errors(errors, len(input))
        output = f"```\n{input}\n{errors_formatted}\n```"
        _ = await interaction.edit_original_response(output)
        return
    evaluated = eval_result.ok_value
    converted = convert_unit(evaluated, target)

    output = f"```\n{input}\n```"
    if verbose:
        output += f"interpreting as\n```\n{expression}\n```"
    output += f"=```\n{converted:~P}\n```"
    _ = await interaction.edit_original_response(output)


@bot.event
async def on_ready() -> None:
    print(f"Logged in as {bot.user} (ID: {bot.user.id})\n")


if __name__ == "__main__":
    bot.run(os.getenv("DISCORD_APPLICATION_TOKEN"))
