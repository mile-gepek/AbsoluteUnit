from pint import UnitRegistry

from absolute_unit.bot import format_magnitude, format_quantity


def test_magnitude_format():
    value = 12345.007885
    formatted = format_magnitude(value, 3)
    assert formatted == "12,345.00789"


def test_quantity_format(unit_registry: UnitRegistry):
    quantity = unit_registry.Quantity("12345.007885 km")
    formatted = format_quantity(quantity, 3)
    assert formatted == "12,345.00789km"
