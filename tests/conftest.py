from collections.abc import Generator

import pytest
from pint import UnitRegistry
from pytest import fixture

from absolute_unit.conversion import get_unit_registry
from absolute_unit.currencies import (
    CurrencyApiResponse,
    clear_currencies,
    clear_ureg_cached_currencies,
    define_exchange_rates,
)


@fixture
def unit_registry() -> UnitRegistry:
    return get_unit_registry()


@pytest.fixture(name="currency_unit_registry")
def currency_ureg_fixture(unit_registry: UnitRegistry) -> Generator[UnitRegistry]:
    with open("tests/mock_currency_data.json", "r") as mock_currency_data_file:
        mock_currency_data = mock_currency_data_file.read()
        mock_currency_data_model = CurrencyApiResponse.model_validate_json(
            mock_currency_data
        )
        define_exchange_rates(
            unit_registry,
            mock_currency_data_model.base_currency,
            mock_currency_data_model.exchange_rates_to_base,
        )
    yield unit_registry
    clear_currencies(unit_registry, mock_currency_data_model.base_currency)
    currency_symbols = tuple(mock_currency_data_model.exchange_rates_to_base.keys())
    clear_ureg_cached_currencies(unit_registry, currency_symbols)
