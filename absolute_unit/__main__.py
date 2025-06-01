import abc
import enum
from typing import Any, Generator, Literal

import pint
from pint import Quantity
from pint.facets.plain.quantity import PlainQuantity

ureg = pint.UnitRegistry(system="mks")


class Token(abc.ABC):
    def __init__(self, token: str, start: int) -> None:
        self._token = token
        self._start = start
        self._end = None

    @property
    def token(self) -> str:
        return self._token

    @token.setter
    def token(self, new_value: str) -> None:
        self._token = new_value

    @property
    def start(self) -> int:
        return self._start

    @property
    def end(self) -> int | None:
        return self._end

    @end.setter
    def end(self, new_end: int) -> None:
        self._end = new_end

    # TODO: should this be a property instead
    # e.g. if we have a Value('3') return '0123456789.'
    # but if we have a Value('3.4') return only digits (no dot since it was already used)
    @abc.abstractmethod
    @staticmethod
    def alphabet() -> str | None: ...


class Value(Token):



class Unit(Token):
    pass


class PairType(enum.Enum):
    PAREN = "()"
    BRACKET = "[]"
    BRACE = "{}"


class Pair(Token):
    def __init__(self, token: PairType) -> None:
        self._token
        self._pair_type = PairType(token)

    @property
    def pair_type(self) -> PairType:
        return self._pair_type


class OperatorType(enum.Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    EXP = "**"


class OperatorToken(Token):
    def __init__(self, type: OperatorType, start: int) -> None:
        super().__init__(type.value, start)
        self._type = type


class Unknown(Token):
    pass


class CharStream:
    def __init__(self, string: str) -> None:
        self._string = string
        self._i = 0

    def __next__(self) -> str:
        if self._i >= len(self._string):
            raise StopIteration

        char = self._string[self._i]
        self._i += 1
        return char


# TODO: rewrite this entire tokenizer owo
def tokenize(s: str) -> Generator[Token, None, None]:
    token = None


if __name__ == "__main__":
    while True:
        inp = input("Unit:")
        quantity = parse(inp)
        print(f"{quantity:~P.2}")
