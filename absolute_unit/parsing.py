"""
This module is used for parsing user input from commands
"""

from __future__ import annotations
import abc
from collections import deque
import enum
import string
import time
from typing import ClassVar, Self, override
from collections.abc import Iterator
from collections.abc import Generator
from pint import Quantity
from pint.facets.plain import PlainQuantity

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

    def __init__(self, token: str, start: int, end: int) -> None:
        self._token: str = token
        self._start: int = start
        self._end: int = end

    @classmethod
    def from_stream(cls, stream: CharStream) -> Token | None:
        char = stream.peek()
        if char is None:
            return None
        token_type: type[Token] = UnknownToken
        for t_type in cls.token_types:
            alphabet = t_type.default_alphabet()
            if alphabet is not None and char in alphabet:
                token_type = t_type
                break
        start = stream.position
        token = token_type.consume(stream)
        return token_type(token, start, stream.position)

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

    @classmethod
    def alphabet(cls, curr_token: str) -> str | None:
        return cls.default_alphabet()

    @classmethod
    def consume(cls, stream: CharStream) -> str:
        token = ""
        while (char := stream.peek()) is not None:
            alphabet = cls.alphabet(token)
            if alphabet is None or char not in alphabet:
                break
            token += char
            stream.advance()
        return token

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
    @classmethod
    def alphabet(cls, curr_token: str) -> str:
        if "." in curr_token:
            return string.digits
        return cls.default_alphabet()

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
    R_BRACE = "}"

    def is_opening(self) -> bool:
        return self in [ParenType.L_PAREN, ParenType.L_BRACKET, ParenType.L_BRACE]

    def paren_name(self) -> str:
        match self:
            case ParenType.L_PAREN:
                return "opening parenthesis"
            case ParenType.R_PAREN:
                return "closing parenthesis"
            case ParenType.L_BRACKET:
                return "opening bracket"
            case ParenType.R_BRACKET:
                return "closing bracket"
            case ParenType.L_BRACE:
                return "opening brace"
            case ParenType.R_BRACE:
                return "closing brace"

    def is_pair(self, other: ParenType) -> bool:
        match self:
            case ParenType.L_PAREN:
                return other == ParenType.R_PAREN
            case ParenType.R_PAREN:
                return other == ParenType.L_PAREN
            case ParenType.L_BRACKET:
                return other == ParenType.R_BRACKET
            case ParenType.R_BRACKET:
                return other == ParenType.L_BRACKET
            case ParenType.L_BRACE:
                return other == ParenType.R_BRACE
            case ParenType.R_BRACE:
                return other == ParenType.L_BRACE


class ParenToken(Token):
    def __init__(self, token: str, start: int, end: int) -> None:
        super().__init__(token, start, end)
        self._paren_type: ParenType = ParenType(token)

    @override
    @staticmethod
    def default_alphabet() -> str:
        return "()[]{}"

    @override
    @classmethod
    def alphabet(cls, curr_token: str) -> str:
        if curr_token:
            return ""
        return cls.default_alphabet()

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
    def __init__(self, token: str, start: int, end: int) -> None:
        super().__init__(token, start, end)
        self._op_type: OperatorType = OperatorType(self._token)

    @override
    @staticmethod
    def default_alphabet() -> str:
        return "+-*/"

    @override
    @classmethod
    def alphabet(cls, curr_token: str) -> str:
        if not curr_token:
            return cls.default_alphabet()
        if curr_token == "*":
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
    @classmethod
    def consume(cls, stream: CharStream) -> str:
        while char := stream.peek():
            if not char.isspace():
                break
            stream.advance()
        return ""


class UnknownToken(Token):
    @override
    @staticmethod
    def default_alphabet() -> str | None:
        return None

    @override
    @classmethod
    def consume(cls, stream: CharStream) -> str:
        token = ""
        while char := stream.peek():
            if char in Token.total_alphabet:
                break
            token += char
            stream.advance()
        return token


def tokenize(s: str) -> Generator[Token, None, None]:
    stream = CharStream(s)
    while stream:
        token: Token | None = Token.from_stream(stream)
        if token is None:
            break
        if not isinstance(token, Whitespace):
            yield token


class ParsingError(Exception):
    def __init__(self, message: str, token: Token):
        super().__init__(f"Parsing error: {message}")
        self.token: Token = token


class UnknownTokenError(ParsingError):
    def __init__(self, token: UnknownToken) -> None:
        super().__init__(f"Unknown syntax: {token}", token)


class UnmatchedParenError(ParsingError):
    def __init__(self, paren_token: ParenToken):
        name = paren_token.paren_type.paren_name
        super().__init__(f"Unmatched {name}", paren_token)


class UnknownPrimaryError(ParsingError):
    def __init__(self, token: Token):
        super().__init__(f"Expected : {token}", token)


class InvalidUnaryError(ParsingError):
    def __init__(self, operator: OperatorToken) -> None:
        super().__init__(f"Invalid unary operator: {operator}", operator)


class ParsingErrorGroup(Exception):
    def __init__(self, errors: list[ParsingError]) -> None:
        if not errors:
            raise ValueError("ParsingError group can not be empty")
        if len(errors) == 1:
            message = f"Got error while parsing: {errors[1]}"
        else:
            message = f"Got multiple errors while parsing: {errors}"
        super().__init__(message)
        self._errors: list[ParsingError] = errors


def parse(input: str) -> PlainQuantity[float]:
    tokens = list(tokenize(input))
    errors: list[ParsingError] = []
    unknown_tokens = [t for t in tokens if isinstance(t, UnknownToken)]
    if unknown_tokens:
        raise ParsingErrorGroup([UnknownTokenError(t) for t in unknown_tokens])
    return parse_expr(tokens)


def parse_unary(tokens: deque[Token]) -> PlainQuantity[float]:
    negative = False
    token = tokens[0]
    while isinstance(token, OperatorToken):
        if token.op_type not in (OperatorType.ADD, OperatorType.SUB):
            raise InvalidUnaryError(token)
        if token.op_type == OperatorType.SUB:
            negative = not negative
        _ = tokens.popleft()
        token = tokens[0]

    result = parse_primary(tokens)
    return -result if negative else result


def parse_primary(tokens: deque[Token]) -> PlainQuantity[float]:
    token = tokens.popleft()

    if isinstance(token, FloatToken):
        return Quantity(float(token.token))

    elif isinstance(token, UnitToken):
        return Quantity(token.token)

    elif isinstance(token, ParenToken):
        if not token.paren_type.is_opening():
            raise UnmatchedParenError(token)
        group_tokens: deque[Token] = deque()
        pairs_open = 1
        t = tokens[0]
        while tokens:
            if isinstance(t, ParenToken):
                if t.paren_type == token.paren_type:
                    pairs_open += 1
                elif t.paren_type.is_pair(token.paren_type):
                    pairs_open -= 1
            if pairs_open == 0:
                break
            group_tokens.append(t)
            t = tokens.popleft()
        if pairs_open:
            raise UnmatchedParenError(token)
        return parse_expr(group_tokens)

    raise UnknownPrimaryError(token)
