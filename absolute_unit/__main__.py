import logging
import os

import disnake
import pint
from disnake.ext import commands
from dotenv import load_dotenv
from pint.facets.plain import PlainQuantity
from pint.util import UnitsContainer
from result import Err, Ok, Result

from absolute_unit import parsing

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


metric_to_imperial = {
    "kilometer": "mile",
    "meter": "ft",
    "decimeter": "ft",
    "centimeter": "inch",
    "kilogram": "pound",
    "gram": "pound",
    "kilometer_per_hour": "mile_per_hour",
}


imperial_to_metric = {
    "mile": "kilometer",
    "foot": "meter",
    "inch": "centimeter",
    "pound": "kilogram",
    "mile_per_hour": "kilometer / hour",
}


class ConversionError(Exception):
    pass


class UnitInferError(ConversionError):
    def __init__(self, *args: object) -> None:
        super().__init__("Can not infer target unit from expression.")


def find_target_unit(
    quantity: PlainQuantity[float],
) -> Result[PlainQuantity[float], UnitInferError]:
    if quantity.units == pint.Unit("cm"):
        if quantity > pint.Quantity("foot"):
            return Ok(pint.Quantity("foot"))
        return Ok(pint.Quantity("inch"))

    new_units: PlainQuantity[float] = pint.Quantity(1)
    has_metric = False
    has_imperial = False
    for unit, power in quantity.unit_items():
        if unit in metric_to_imperial:
            if has_imperial:
                return Err(UnitInferError())
            has_metric = True
            new_unit = metric_to_imperial[unit]

        elif unit in imperial_to_metric:
            if has_metric:
                return Err(UnitInferError())
            has_imperial = True
            new_unit = imperial_to_metric[unit]

        else:
            new_unit = unit
        new_units *= pint.Quantity(new_unit) ** power

    return Ok(new_units)


def convert_unit(
    quantity: PlainQuantity[float],
    target: str | None = None,
) -> Result[PlainQuantity[float], ConversionError]:
    if target is None:
        target_result = find_target_unit(quantity)
        if isinstance(target_result, Err):
            return target_result
        target_unit = target_result.ok_value
    else:
        try:
            target_unit = pint.Unit(target)
        except pint.errors.UndefinedUnitError as e:
            units = ", ".join(e.unit_names)
            return Err(ConversionError(f"Undefined target unit(s): {units}."))
    try:
        return Ok(quantity.to(target_unit).to_reduced_units())
    except pint.errors.DimensionalityError as e:
        return Err(
            ConversionError(
                f"Can not convert expression of dimension '{e.dim1}' to dimension '{e.dim2}'."
            )
        )


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
        Print the intepretation of the parsed expression. Use this if output is unexpected.
    """
    # TODO: test whether defering is needed.
    await interaction.response.defer()

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

    converted_result = convert_unit(evaluated, target)
    if isinstance(converted_result, Err):
        error_str = str(converted_result.err_value)
        output = f"```\n{input}\n{error_str}\n```"
        _ = await interaction.edit_original_response(output)
        return
    converted = converted_result.ok_value

    if converted.units == pint.Unit("foot"):
        magnitude = converted.magnitude
        whole = int(magnitude)
        quantity_foot = whole * pint.Unit("foot")
        decimal = magnitude - whole
        quantity_inch = decimal * 12 * pint.Unit("inch")
        converted_str = f"{quantity_foot:~P} {quantity_inch:.3g~P}"
    else:
        converted_str = f"{converted:.3g~P}"

    if verbose:
        output = f"```\n{input}\n```interpreting as\n```\n{expression}\n=\n{converted_str}\n```"
    else:
        output = f"`{input}` = `{converted_str}`"
    _ = await interaction.edit_original_response(output)


@bot.event
async def on_ready() -> None:
    print(f"Logged in as {bot.user} (ID: {bot.user.id})\n")


if __name__ == "__main__":
    bot.run(os.getenv("DISCORD_APPLICATION_TOKEN"))
