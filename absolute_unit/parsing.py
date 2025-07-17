"""
This module is used for parsing user input from commands
"""

from __future__ import annotations
import abc
from collections import deque
import enum
import string
from typing import ClassVar, Self, override
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

    @override
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._token}, {self.span})"

    @override
    def __repr__(self) -> str:
        return str(self)


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
        return other in self.to_pair()

    def to_pair(self) -> tuple[ParenType, ParenType]:
        match self:
            case ParenType.L_PAREN | ParenType.R_PAREN:
                return (ParenType.L_PAREN, ParenType.R_PAREN)
            case ParenType.L_BRACKET | ParenType.R_BRACKET:
                return (ParenType.L_BRACKET, ParenType.R_BRACKET)
            case ParenType.L_BRACE | ParenType.R_BRACE:
                return (ParenType.L_BRACE, ParenType.R_BRACE)


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


class Expression(abc.ABC):
    @override
    def __str__(self) -> str:
        raise NotImplementedError(f"{self.__class__.__qualname__} does not implement str")

    def evaluate(self) -> PlainQuantity[float]:
        raise NotImplementedError(f"{self.__class__.__qualname__} does not implement 'evaluate'")

    def human_readable(self) -> str:
        raise NotImplementedError(f"{self.__class__.__qualname__} does not implement 'human_readable'")


class Binary(Expression):
    def __init__(
        self,
        left: Expression | None,
        right: Expression | None = None,
        op: OperatorType | None = None,
    ) -> None:
        self._left: Expression | None = left
        self._right: Expression | None = right
        self._op_type: OperatorType | None = op

    @property
    def left(self) -> Expression | None:
        return self._left

    @left.setter
    def left(self, new_left: Expression | None) -> None:
        self._left = new_left

    @property
    def right(self) -> Expression | None:
        return self._right

    @right.setter
    def right(self, new_right: Expression | None) -> None:
        self._right = new_right

    @property
    def op(self) -> OperatorType | None:
        return self._op_type

    @op.setter
    def op(self, new_op: OperatorType | None) -> None:
        self._op_type = new_op

    @override
    def __str__(self) -> str:
        repr = f"Binary({self._left}"
        if self._op_type is not None:
            repr += f" {self._op_type.value}"
        if self._right is not None:
            repr += f" {self._right}"
        repr += ")"
        return repr

    @override
    def __repr__(self) -> str:
        return str(self)


class Unary(Expression):
    def __init__(
        self, value: Expression | None, op: OperatorType | None = None
    ) -> None:
        self._value: Expression | None = value
        self._op_type: OperatorType | None = op

    @property
    def value(self) -> Expression | None:
        return self._value

    @value.setter
    def value(self, new_value: Expression | None) -> None:
        self._value = new_value

    @property
    def op(self) -> OperatorType | None:
        return self._op_type

    @op.setter
    def op(self, new_op: OperatorType | None) -> None:
        self._op_type = new_op

    @override
    def __str__(self) -> str:
        repr = "Unary("
        if self._op_type is not None:
            repr += f"{self._op_type.value}"
        repr += f"{self._value})"
        return repr

    @override
    def __repr__(self) -> str:
        return str(self)


class Primary(Expression, abc.ABC):
    def __neg__(self) -> Primary:
        return -self


class Float(Primary):
    def __init__(self, value: float) -> None:
        self._value: float = value

    @override
    def __neg__(self) -> Float:
        return Float(-self._value)

    @property
    def value(self) -> float:
        return self._value

    @override
    def __str__(self) -> str:
        return f"Float({self._value})"

    @override
    def __repr__(self) -> str:
        return str(self)


class Unit(Primary):
    def __init__(self, unit: str) -> None:
        self._unit: str = unit

    @override
    def __neg__(self) -> Unit:
        return Unit(f"-{self._unit}")

    def unit_str(self) -> str:
        return self._unit

    @override
    def __str__(self) -> str:
        return f"Unit({self._unit})"


class PrimaryChain(Primary):
    def __init__(self, terms: list[Float | Unit]) -> None:
        self._terms: list[Float | Unit] = terms

    @override
    def __str__(self) -> str:
        terms_as_str = ",".join(str(t) for t in self._terms)
        return f"[{terms_as_str}]"

    @override
    def human_readable(self) -> str:
        return " ".join

    @override
    def evaluate(self) -> PlainQuantity[float]:
        ...


class Group(Expression):
    def __init__(self, expr: Expression | None, paren_type: ParenType):
        self._paren_type: ParenType = paren_type
        self._expr: Expression | None = expr

    @override
    def __str__(self) -> str:
        opening, closing = self._paren_type.to_pair()
        return f"Group({opening.value}{self._expr}{closing.value})"

    @override
    def evaluate(self) -> PlainQuantity[float]:
        if self._expr is None:
            raise EvaluationError("Tried to evaluate None", None)
        return self._expr.evaluate()


class ParsingError(Exception):
    def __init__(self, message: str, token: Token | None = None):
        super().__init__(f"Parsing error: {message}")
        self.token: Token | None = token


class UnknownTokenError(ParsingError):
    def __init__(self, token: UnknownToken) -> None:
        super().__init__(f"Unknown syntax: {token}", token)


class UnmatchedParenError(ParsingError):
    def __init__(self, paren_token: ParenToken):
        name = paren_token.paren_type.paren_name
        super().__init__(f"Unmatched {name}", paren_token)


class UnknownPrimaryError(ParsingError):
    def __init__(self, token: Token):
        super().__init__(f"Expected number, unit or group expression, got: {token}", token)


class InvalidUnaryError(ParsingError):
    def __init__(self, operator: OperatorToken) -> None:
        super().__init__(f"Invalid unary operator: {operator}", operator)


class ExpectedPrimaryError(ParsingError):
    def __init__(self) -> None:
        super().__init__("Expected number, unit or group expression, got EOF")


class ParsingErrorGroup(Exception):
    def __init__(self, errors: list[ParsingError] | None = None) -> None:
        if errors is None:
            errors = []
        if len(errors) == 1:
            message = f"Got error while parsing:\n{errors[0]}"
        else:
            message = f"Got multiple errors while parsing:\n{errors}"
        super().__init__(message)
        self._errors: list[ParsingError] = errors

    @property
    def errors(self) -> list[ParsingError]:
        return self._errors


class EvaluationError(Exception):
    def __init__(self, message: str, expression: Expression | None) -> None:
        if expression is None:
            expr_format = "None"
        else:
            expr_format = expression.human_readable()
        super().__init__(f"Error evaluating expression {expr_format}")
        self._expr = expression


def parse(input: str) -> Expression | None:
    tokens = list(tokenize(input))
    unknown_tokens = [t for t in tokens if isinstance(t, UnknownToken)]
    if unknown_tokens:
        errors: list[ParsingError] = [UnknownTokenError(t) for t in unknown_tokens]
        raise ParsingErrorGroup(errors)
    result = _parse_sum(deque(tokens))
    return result

def _parse_binary(
    tokens: deque[Token], ops: tuple[OperatorType, ...]
) -> Expression | None:
    errors: list[ParsingError] = []
    match ops[0]:
        case OperatorType.ADD | OperatorType.SUB:
            parse_term = _parse_mul
        case OperatorType.MUL | OperatorType.DIV:
            parse_term = _parse_exp
        case OperatorType.EXP:
            parse_term = _parse_unary
    term = None
    try:
        term = parse_term(tokens)
    except ParsingErrorGroup as error_group:
        errors.extend(error_group.errors)
    except ParsingError as e:
        errors.append(e)
    bin = Binary(left=term)
    curr_bin = bin
    prev_bin = bin
    while tokens:
        token = tokens[0]
        if not isinstance(token, OperatorToken):
            break
        if token.op_type not in ops:
            break
        _ = tokens.popleft()
        term = None
        try:
            term = parse_term(tokens)
        except ParsingErrorGroup as error_group:
            errors.extend(error_group.errors)
        except ParsingError as e:
            errors.append(e)                
        curr_bin.right = Binary(term)
        curr_bin.op = token.op_type
        prev_bin = curr_bin
        curr_bin = curr_bin.right
    if errors:
        raise ParsingErrorGroup(errors)
    if bin.op is None:
        return bin.left
    if curr_bin.op is None:
        prev_bin.right = curr_bin.left
    return bin

def _parse_sum(
    tokens: deque[Token]
) -> Expression | None:
    return _parse_binary(tokens, (OperatorType.ADD, OperatorType.SUB))

def _parse_mul(
    tokens: deque[Token]
) -> Expression | None:
    return _parse_binary(tokens, (OperatorType.MUL, OperatorType.DIV))

def _parse_exp(
    tokens: deque[Token]
) -> Expression | None:
    return _parse_binary(tokens, (OperatorType.EXP,))

def _parse_unary(tokens: deque[Token]) -> Expression | None:
    if not tokens:
        raise ExpectedPrimaryError()
    unary = Unary(None)
    curr_unary = prev = unary
    while tokens:
        token = tokens[0]
        if not isinstance(token, OperatorToken):
            primary = _parse_primary(tokens)
            prev.value = primary
            break
        if token.op_type not in (OperatorType.ADD, OperatorType.SUB):
            raise InvalidUnaryError(token)
        curr_unary.op = token.op_type
        curr_unary.value = Unary(None)
        prev = curr_unary
        curr_unary = curr_unary.value
        _ = tokens.popleft()
    if unary.op is None:
        return unary.value
    return unary

def _parse_primary(tokens: deque[Token]) -> Primary | Group:
    if not tokens:
        raise ExpectedPrimaryError()
    token = tokens[0]

    if isinstance(token, ParenToken):
        paren_token = token
        if not paren_token.paren_type.is_opening():
            raise UnmatchedParenError(paren_token)
        _ = tokens.popleft()
        group_tokens: deque[Token] = deque()
        pairs_open = 1
        while tokens:
            token = tokens.popleft()
            if isinstance(token, ParenToken):
                if token.paren_type == paren_token.paren_type:
                    pairs_open += 1
                elif token.paren_type.is_pair(paren_token.paren_type):
                    pairs_open -= 1
            if pairs_open == 0:
                break
            group_tokens.append(token)
        if pairs_open:
            raise UnmatchedParenError(paren_token)
        expr = _parse_sum(group_tokens)
        return Group(expr, paren_token.paren_type)

    primaries: list[Float | Unit] = []
    while tokens:
        token = tokens[0]
        if isinstance(token, FloatToken):
            primaries.append(Float(token.to_float()))
        elif isinstance(token, UnitToken):
            primaries.append(Unit(token.token))
        else:
            break
        _ = tokens.popleft()
    if not primaries:
        raise UnknownPrimaryError(token)
    if len(primaries) == 1:
        return primaries[0]
    else:
        return PrimaryChain(primaries)

