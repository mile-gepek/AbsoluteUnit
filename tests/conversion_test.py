import pytest
from pint import UnitRegistry
from pint.util import UnitsContainer
from result import Err, Ok

from absolute_unit.conversion import (
    DimensionalityError,
    UnitInferError,
    convert,
    imperial_to_metric,
    infer_target_unit,
    metric_to_imperial,
)


def str_to_units_container(unit_registry: UnitRegistry, units: str) -> UnitsContainer:
    quantity = unit_registry.Quantity(units)
    unit_container = UnitsContainer(quantity.unit_items())
    return unit_container


@pytest.mark.parametrize("metric, imperial", list(metric_to_imperial.items()))
def test_infer_target_unit_metric_to_imperial(
    unit_registry: UnitRegistry, metric: str, imperial: str
):
    qty = unit_registry(metric)
    result = infer_target_unit(qty, unit_registry)

    assert isinstance(result, Ok)
    units = result.ok()
    assert imperial in units


@pytest.mark.parametrize("imperial, metric", list(imperial_to_metric.items()))
def test_infer_target_unit_imperial_to_metric(
    unit_registry: UnitRegistry, imperial: str, metric: str
):
    qty = unit_registry(imperial)
    result = infer_target_unit(qty, unit_registry)

    assert isinstance(result, Ok)
    units = result.ok()
    assert metric in units


def test_infer_target_unit_mixed_metric_and_imperial_length(
    unit_registry: UnitRegistry,
):
    qty = unit_registry("5 mile") / unit_registry("2 meter")
    result = infer_target_unit(qty, unit_registry)

    assert isinstance(result, Err)
    assert isinstance(result.err(), UnitInferError)


def test_infer_target_unit_mixed_metric_and_imperial_weight(
    unit_registry: UnitRegistry,
):
    qty = unit_registry("kg") * unit_registry("lbs")
    result = infer_target_unit(qty, unit_registry)

    assert isinstance(result, Err)
    assert isinstance(result.err(), UnitInferError)


def test_infer_target_unit_mixed_metric_and_imperial_speed(
    unit_registry: UnitRegistry,
):
    qty = unit_registry("mph") * unit_registry("meter")
    result = infer_target_unit(qty, unit_registry)

    assert isinstance(result, Err)
    assert isinstance(result.err(), UnitInferError)


@pytest.mark.parametrize(
    "src_unit, expected_unit",
    [
        ("kilometer", "mile"),
        ("meter", "foot"),
        ("decimeter", "foot"),
        ("centimeter", "inch"),
        ("kilogram", "pound"),
        ("gram", "ounce"),
        ("kilometer / hour", "mile / hour"),
        ("mile", "kilometer"),
        ("foot", "meter"),
        ("inch", "centimeter"),
        ("pound", "kilogram"),
        ("mile / hour", "kilometer / hour"),
    ],
)
def test_convert(unit_registry: UnitRegistry, src_unit: str, expected_unit: str):
    qty = unit_registry(src_unit)
    target = str_to_units_container(unit_registry, expected_unit)
    result = convert(qty, target)

    assert isinstance(result, Ok)
    converted = result.ok()
    assert expected_unit in str(converted.units)


@pytest.mark.parametrize(
    "src_unit, target_unit",
    [
        ("meter", "second"),  # length -> time
        ("kilogram", "meter"),  # mass -> length
        ("second", "pound"),  # time -> mass
    ],
)
def test_convert_expression_dimensionality_mismatch(
    unit_registry: UnitRegistry, src_unit: str, target_unit: str
):
    qty = unit_registry(src_unit)
    target = str_to_units_container(unit_registry, target_unit)
    result = convert(qty, target)

    assert isinstance(result, Err)
    error = result.err()
    assert isinstance(error, DimensionalityError)


def test_currency_conversion(currency_unit_registry: UnitRegistry):
    quantity = currency_unit_registry("EUR")
    target = str_to_units_container(currency_unit_registry, "USD")
    result = convert(quantity, target)
    assert isinstance(result, Ok)
    converted = result.ok()
    assert "USD" in str(converted.units)
