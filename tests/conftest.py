from pint import UnitRegistry
from pytest import fixture

from absolute_unit.conversion import get_unit_registry


@fixture
def unit_registry() -> UnitRegistry:
    return get_unit_registry()
