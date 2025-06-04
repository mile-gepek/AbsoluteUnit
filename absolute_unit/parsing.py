"""
This module is used for parsing user input from commands
"""

import abc
import enum
import string
from typing import Self, Generator

__all__ = [
    "CharStream",
    "tokenize",
    "Token",
    "FloatToken",
    "UnitToken",
    "ParenType",
    "ParenToken",
    "OperatorType",
    "OperatorToken",
]


class CharStream:
    def __init__(self, string: str) -> None:
        self._string = string
        self._i = 0

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
    total_alphabet = ""

    def __init__(self, stream: CharStream) -> None:
        self._token = ""
        self._start = stream.position
        self.consume(stream)
        self._end = stream.position

    @classmethod
    def from_stream(cls, stream: CharStream) -> Self | None:
        char = stream.peek()
        if char is None:
            return None
        if char in FloatToken.default_alphabet():
            return FloatToken(stream)
        elif char in UnitToken.default_alphabet():
            return UnitToken(stream)
        elif char in ParenToken.default_alphabet():
            return ParenToken(stream)
        elif char in OperatorToken.default_alphabet():
            return OperatorToken(stream)
        elif char in WhitespaceToken.default_alphabet():
            return WhitespaceToken(stream)
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
        Token.total_alphabet += alphabet


class InvalidCharacter(Exception): ...


class FloatToken(Token):
    @staticmethod
    def default_alphabet() -> str:
        return string.digits + "."

    def alphabet(self) -> str:
        if "." in self._token:
            return string.digits
        return self.default_alphabet()

    def to_float(self) -> float:
        return float(self._token)


class UnitToken(Token):
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
        self._paren_type = ParenType(self._token)

    @staticmethod
    def default_alphabet() -> str:
        return "()[]{}"

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
        self._op_type = OperatorType(self._token)

    @staticmethod
    def default_alphabet() -> str:
        return "+-*/"

    def alphabet(self) -> str:
        if not self._token:
            return self.default_alphabet()
        if self._token == "*":
            return "*"
        return ""

    @property
    def op_type(self) -> OperatorType:
        return self._op_type


class WhitespaceToken(Token):
    @staticmethod
    def default_alphabet() -> str:
        return string.whitespace

    def consume(self, stream: CharStream):
        while (char := stream.peek()) is not None:
            if char not in self.default_alphabet():
                break
            stream.advance()


class UnknownToken(Token):
    @staticmethod
    def default_alphabet() -> str | None:
        return None

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
        token = Token.from_stream(stream)
        if token is None:
            break
        if not isinstance(token, WhitespaceToken):
            yield token
