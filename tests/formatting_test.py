from absolute_unit.conversion import get_unit_registry
from absolute_unit.bot import format_magnitude, format_quantity


ureg = get_unit_registry()


def test_magnitude_format():
    value = 12345.007885
    formatted = format_magnitude(value, 3)
    assert formatted == "12,345.00789"


def test_quantity_format():
    quantity = ureg.Quantity("12345.007885 km")
    formatted = format_quantity(quantity, 3)
    assert formatted == "12,345.00789km"
