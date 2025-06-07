"""
This module is used for parsing user input from commands
"""

from __future__ import annotations
import abc
import enum
import string
from typing import ClassVar, Self, override
from collections.abc import Generator

__all__ = [
    "tokenize",
]


class CharStream:
    def __init__(self, string: str) -> None:
        self._string: str = string
        self._i: int = 0

    @property
    def position(self) -> int:
        return self._i

    def peek(self) -> str | None:
        if self._i >= len(self._string):
            return None
        return self._string[self._i]

    def advance(self) -> None:
        if self._i < len(self._string):
            self._i += 1

    def __next__(self) -> str:
        char = self.peek()
        self.advance()
        if char is None:
            raise StopIteration
        return char

    def __iter__(self) -> Self:
        return self

    def __bool__(self) -> bool:
        return bool(self._string)


class Token(abc.ABC):
    total_alphabet: ClassVar[str] = ""
    token_types: ClassVar[list[type[Self]]] = []

    def __init__(self, stream: CharStream) -> None:
        self._token: str = ""
        self._start: int = stream.position
        self.consume(stream)
        self._end: int = stream.position

    @classmethod
    def from_stream(cls, stream: CharStream) -> Token | None:
        char = stream.peek()
        if char is None:
            return None
        for token_type in cls.token_types:
            alphabet = token_type.default_alphabet()
            if alphabet is not None and char in alphabet:
                return token_type(stream)
        return UnknownToken(stream)

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
    def end(self) -> int:
        return self._end

    @property
    def span(self) -> tuple[int, int]:
        return (self._start, self._end)

    @staticmethod
    @abc.abstractmethod
    def default_alphabet() -> str | None:
        return None

    def alphabet(self) -> str | None:
        return self.default_alphabet()

    def consume(self, stream: CharStream):
        while (char := stream.peek()) is not None:
            alphabet = self.alphabet()
            if alphabet is None or char not in alphabet:
                break
            self._token += char
            stream.advance()

    def __init_subclass__(cls) -> None:
        alphabet = cls.default_alphabet()
        if alphabet is None:
            return
        Token.token_types.append(cls)
        Token.total_alphabet += alphabet


class FloatToken(Token):
    @override
    @staticmethod
    def default_alphabet() -> str:
        return string.digits + "."

    @override
    def alphabet(self) -> str:
        if "." in self._token:
            return string.digits
        return self.default_alphabet()

    def to_float(self) -> float:
        return float(self._token)


class UnitToken(Token):
    @override
    @staticmethod
    def default_alphabet() -> str:
        return string.ascii_letters


class ParenType(enum.Enum):
    L_PAREN = "("
    R_PAREN = ")"

    L_BRACKET = "["
    R_BRACKET = "]"

    L_BRACE = "{"
    R_BRACE = ""


class ParenToken(Token):
    def __init__(self, stream: CharStream) -> None:
        super().__init__(stream)
        self._paren_type: ParenType = ParenType(self._token)

    @override
    @staticmethod
    def default_alphabet() -> str:
        return "()[]{}"

    @override
    def alphabet(self) -> str:
        if self._token:
            return ""
        return self.default_alphabet()

    @property
    def paren_type(self) -> ParenType:
        return self._paren_type


class OperatorType(enum.Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    EXP = "**"


class OperatorToken(Token):
    def __init__(self, stream: CharStream) -> None:
        super().__init__(stream)
        self._op_type: OperatorType = OperatorType(self._token)

    @override
    @staticmethod
    def default_alphabet() -> str:
        return "+-*/"

    @override
    def alphabet(self) -> str:
        if not self._token:
            return self.default_alphabet()
        if self._token == "*":
            return "*"
        return ""

    @property
    def op_type(self) -> OperatorType:
        return self._op_type


class Whitespace(Token):
    """
    Whitespace gets skipped during tokenization
    """

    @override
    @staticmethod
    def default_alphabet() -> str:
        return string.whitespace

    @override
    def consume(self, stream: CharStream):
        while char := stream.peek():
            if not char.isspace():
                break
            stream.advance()


class UnknownToken(Token):
    @override
    @staticmethod
    def default_alphabet() -> str | None:
        return None

    @override
    def consume(self, stream: CharStream):
        while char := stream.peek():
            if char in Token.total_alphabet:
                break
            self._token += char
            stream.advance()


# TODO: rewrite this entire tokenizer owo
def tokenize(s: str) -> Generator[Token, None, None]:
    stream = CharStream(s)
    while stream:
        token: Token | None = Token.from_stream(stream)
        if token is None:
            break
        if not isinstance(token, Whitespace):
            yield token
