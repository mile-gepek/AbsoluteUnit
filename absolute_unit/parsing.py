"""
This module is used for parsing user input from commands
"""

from __future__ import annotations

import abc
import enum
import operator
import pint
import string
import rich.repr
from collections import deque
from collections.abc import Callable, Generator
from typing import ClassVar, Self, override

from pint import Quantity
from pint.facets.plain import PlainQuantity
from pint.util import UnitsContainer
from result import Result, Ok, Err

__all__ = [
    "tokenize",
]


class _EOF:
    """
    Marker for token/expression spans.
    Used when an expression is expected but there are no more tokens.
    """

    @override
    def __repr__(self) -> str:
        return "EOF"


EOF = _EOF()


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
    """
    Base class for all Token types.

    Tokenization is implemented via the `consume` method (overriden if certain Tokens want to).
    Tokens are "registered" using the `__init_subclass__` hook, which stores all token types and a total alphabet (used for discovering unknown tokens).
    """

    total_alphabet: ClassVar[str] = ""
    """
    A string containing all possible expression characters.
    When a token is created with `Tokem.from_stream` and the first character from the stream isn't recognized, it returns an UnknownToken.
    """
    token_types: ClassVar[list[type[Self]]] = []

    def __init__(self, token: str, start: int, end: int) -> None:
        self._token: str = token
        self._start: int = start
        self._end: int = end

    @classmethod
    def from_stream(cls, stream: CharStream) -> Token | None:
        """
        Peek into the stream and return a Token depending on the character.
        The token type is decided based on it's `default_alphabet`, or UnknownToken if none of the match.
        """
        char = stream.peek()
        if char is None:
            return None
        if char not in cls.total_alphabet:
            start = stream.position
            token_str = UnknownToken.consume(stream)
            return UnknownToken(token_str, start, stream.position)
        for token_type in cls.token_types:
            alphabet = token_type.default_alphabet()
            if alphabet is not None and char in alphabet:
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

    def span(self) -> tuple[int, int]:
        return (self._start, self._end)

    @staticmethod
    @abc.abstractmethod
    def default_alphabet() -> str | None:
        return None

    @classmethod
    def alphabet(cls, curr_token: str) -> str | None:  # pyright: ignore [reportUnusedParameter]
        """
        Context-dependant alphabet.
        Certain Tokens, such as `OperatorToken`s want to change their alphabet depending on the characters they've already consumed
        """
        return cls.default_alphabet()

    @classmethod
    def consume(cls, stream: CharStream) -> str:
        """
        The standard way of grabbing a token from a stream, used by most Token types.
        Consumes stream characters one by one, stopping when it finds a character which isn't in the Token's `alphabet`
        """
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
        return f"{self.__class__.__name__}({self._token}, {self.span()})"

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
        """
        Used to check if the float token already contains a dot.
        """
        if "." in curr_token:
            return string.digits
        return cls.default_alphabet()

    def to_float(self) -> float:
        return float(self._token)


class UnitToken(Token):
    """
    Units to be used for the pint library.

    NOTE: The units are not checked on token creation or when creating the syntax tree, these represent any ascii string
    """

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
        """
        Return True if `self` and `other` are pairs e.g. "[" forms a pair with "]", but not itself
        """
        match self:
            case ParenType.L_PAREN:
                return ParenType.R_PAREN == other
            case ParenType.R_PAREN:
                return ParenType.L_PAREN == other
            case ParenType.L_BRACKET:
                return ParenType.R_BRACKET == other
            case ParenType.R_BRACKET:
                return ParenType.L_BRACKET == other
            case ParenType.L_BRACE:
                return ParenType.R_BRACE == other
            case ParenType.R_BRACE:
                return ParenType.L_BRACE == other

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


_BINARY_OP_MAP: dict[
    OperatorType,
    Callable[[PlainQuantity[float], PlainQuantity[float]], PlainQuantity[float]],
] = {
    OperatorType.ADD: operator.add,
    OperatorType.SUB: operator.sub,
    OperatorType.MUL: operator.mul,
    OperatorType.DIV: operator.truediv,
    OperatorType.EXP: operator.pow,
}

_UNARY_OP_MAP: dict[
    OperatorType, Callable[[PlainQuantity[float]], PlainQuantity[float]]
] = {
    OperatorType.ADD: lambda x: x,
    OperatorType.SUB: lambda x: -x,
}


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
    @abc.abstractmethod
    def start(self) -> int: ...

    @abc.abstractmethod
    def end(self) -> int: ...

    def span(self) -> tuple[int, int]:
        return (self.start(), self.end())

    @abc.abstractmethod
    def dimensionality(self) -> pint.util.UnitsContainer: ...

    @abc.abstractmethod
    def is_unit(self) -> bool: ...

    @abc.abstractmethod
    def evaluate(self) -> PlainQuantity[float]: ...

    @override
    def __eq__(self, other: object) -> bool: ...


class Binary(Expression):
    def __init__(
        self,
        left: Expression,
        op: OperatorType,
        right: Expression,
        *,
        implicit: bool = False,
    ) -> None:
        self.left: Expression = left
        self.right: Expression = right
        self.op: OperatorType = op
        self.implicit: bool = implicit

    @override
    def start(self) -> int:
        return self.left.start()

    @override
    def end(self) -> int:
        return self.right.end()

    @override
    def dimensionality(self) -> pint.util.UnitsContainer:
        match self.op:
            case OperatorType.MUL:
                return self.left.dimensionality() * self.right.dimensionality()
            case OperatorType.DIV:
                return self.left.dimensionality() / self.right.dimensionality()
            case OperatorType.EXP:
                return self.left.dimensionality() * self.right.dimensionality()
            case _:
                return self.left.dimensionality()

    @override
    def is_unit(self) -> bool:
        return self.left.is_unit() or self.right.is_unit()

    @override
    def evaluate(self) -> PlainQuantity[float]:
        op = _BINARY_OP_MAP[self.op]
        return op(self.left.evaluate(), self.right.evaluate())

    def _as_str(self) -> str:
        if isinstance(self.left, Binary) and self.implicit:
            left = self.left._as_str()
        else:
            left = str(self.left)
        if isinstance(self.right, Binary) and self.implicit:
            right = self.right._as_str()
        else:
            right = str(self.right)

        if (
            isinstance(self.left, Float)
            and self.right.is_unit()
            and self.op == OperatorType.MUL
        ):
            s = f"{left}{right}"
        else:
            s = f"{left} {self.op.value} {right}"

        return s

    @override
    def __str__(self) -> str:
        s = self._as_str()
        if self.implicit:
            return f"({s})"
        return s

    @override
    def __repr__(self) -> str:
        return f"Binary({self.op.value} ({self.left!r}, {self.right!r}))"

    def __rich_repr__(self) -> rich.repr.Result:
        yield "op", self.op.value
        yield "left", self.left
        yield "right", self.right
        yield "implicit", self.implicit, False

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Expression):
            raise TypeError(
                f"Can not compare {self.__class__.__qualname__} and {other.__class__.__qualname__}"
            )
        if not isinstance(other, Binary):
            return False
        return (
            self.left == other.left
            and self.op == other.op
            and self.right == other.right
        )


class Unary(Expression):
    def __init__(self, op: OperatorType, value: Expression, start: int) -> None:
        self.value: Expression = value
        self.op: OperatorType = op
        self._start: int = start

    @override
    def start(self) -> int:
        return self._start

    @override
    def end(self) -> int:
        return self.value.end()

    @override
    def dimensionality(self) -> pint.util.UnitsContainer:
        return self.value.dimensionality()

    @override
    def is_unit(self) -> bool:
        return self.value.is_unit()

    @override
    def evaluate(self) -> PlainQuantity[float]:
        op = _UNARY_OP_MAP[self.op]
        return op(self.value.evaluate())

    @override
    def __str__(self) -> str:
        return f"{self.op.value}{self.value}"

    @override
    def __repr__(self) -> str:
        return f"Unary({self.op}{self.value!r})"

    def __rich_repr__(self) -> rich.repr.Result:
        yield "op", self.op.value
        yield "value", self.value

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Expression):
            raise TypeError(
                f"Can not compare {self.__class__.__qualname__} and {other.__class__.__qualname__}"
            )
        if not isinstance(other, Unary):
            return False
        return self.value == other.value and self.op == other.op


class Primary(Expression, abc.ABC):
    def __init__(self, start: int, end: int) -> None:
        self._start: int = start
        self._end: int = end

    @override
    def start(self) -> int:
        return self._start

    @override
    def end(self) -> int:
        return self._end

    def __neg__(self) -> Primary:
        return -self

    @staticmethod
    def from_token(
        token: UnitToken | FloatToken,
    ) -> Result[Unit | Float, UndefinedUnitError]:
        if isinstance(token, UnitToken):
            return Unit.try_new(token)
        return Ok(Float(token.to_float(), token.start, token.end))


class Float(Primary):
    def __init__(self, value: float, start: int, end: int) -> None:
        super().__init__(start, end)
        self._value: float = value

    @property
    def value(self) -> float:
        return self._value

    @override
    def dimensionality(self) -> pint.util.UnitsContainer:
        return UnitsContainer()

    @override
    def is_unit(self) -> bool:
        return False

    @override
    def evaluate(self) -> PlainQuantity[float]:
        return Quantity(self._value)

    @override
    def __str__(self) -> str:
        return str(self._value)

    @override
    def __repr__(self) -> str:
        return f"Float({self._value})"

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.value

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Expression):
            raise TypeError(
                f"Can not compare {self.__class__.__qualname__} and {other.__class__.__qualname__}"
            )
        if not isinstance(other, Float):
            return False
        return self._value == other._value


class Unit(Primary):
    def __init__(
        self, unit: PlainQuantity[float], as_str: str, start: int, end: int
    ) -> None:
        super().__init__(start, end)
        self.unit: PlainQuantity[float] = unit
        self._as_str: str = as_str

    @classmethod
    def try_new(cls, unit_token: UnitToken) -> Result[Self, UndefinedUnitError]:
        try:
            unit = Quantity(unit_token.token)
        except pint.UndefinedUnitError:
            return Err(UndefinedUnitError(unit_token))
        return Ok(cls(unit, unit_token.token, unit_token.start, unit_token.end))

    @override
    def dimensionality(self) -> pint.util.UnitsContainer:
        return self.unit.dimensionality

    @override
    def is_unit(self) -> bool:
        return True

    @override
    def evaluate(self) -> PlainQuantity[float]:
        return self.unit

    @override
    def __str__(self) -> str:
        return self._as_str

    @override
    def __repr__(self) -> str:
        return f"Unit({self.unit})"

    def __rich_repr__(self) -> rich.repr.Result:
        yield self._as_str

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Expression):
            raise TypeError(
                f"Can not compare {self.__class__.__qualname__} and {other.__class__.__qualname__}"
            )
        if not isinstance(other, Unit):
            return False
        return self.unit == other.unit  # pyright: ignore [reportReturnType, reportUnknownVariableType]


class Group(Expression):
    def __init__(self, expr: Expression, paren_type: ParenType, start: int, end: int):
        self.paren_type: ParenType = paren_type
        self.expr: Expression = expr
        self._start: int = start
        self._end: int = end

    @override
    def start(self) -> int:
        return self._start

    @override
    def end(self) -> int:
        return self._end

    @override
    def dimensionality(self) -> pint.util.UnitsContainer:
        return self.expr.dimensionality()

    @override
    def is_unit(self) -> bool:
        return self.expr.is_unit()

    @override
    def evaluate(self) -> PlainQuantity[float]:
        return self.expr.evaluate()

    @override
    def __str__(self) -> str:
        opening, closing = self.paren_type.to_pair()
        if isinstance(self.expr, Binary) and self.expr.implicit:
            return str(self.expr)
        return f"{opening.value}{self.expr}{closing.value}"

    @override
    def __repr__(self) -> str:
        opening, closing = self.paren_type.to_pair()
        return f"Group({opening.value}{self.expr!r}{closing.value})"

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.expr

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Expression):
            raise TypeError(
                f"Can not compare {self.__class__.__qualname__} and {other.__class__.__qualname__}"
            )
        if isinstance(other, Group):
            return self.expr == other.expr
        return self.expr == other


class ParsingError(Exception):
    def __init__(self, message: str, span: tuple[int, int] | _EOF = EOF):
        super().__init__(message)
        self.span: tuple[int, int] | _EOF = span


class UnexpectedTokenError(ParsingError):
    def __init__(self, token: Token, *, expected: str) -> None:
        super().__init__(f"Expected {expected}, got '{token.token}'.", token.span())


class UnmatchedParenError(ParsingError):
    def __init__(self, paren_token: ParenToken):
        name = paren_token.paren_type.paren_name()
        super().__init__(f"Unmatched {name}.", paren_token.span())


class EmptyGroupExpression(ParsingError):
    def __init__(self, span: tuple[int, int]) -> None:
        super().__init__("Empty group expression.", span)


class InvalidUnaryError(ParsingError):
    def __init__(self, operator_token: OperatorToken) -> None:
        super().__init__(
            f"Invalid unary operator: {operator_token.token}.", operator_token.span()
        )


class ExpectedPrimaryError(ParsingError):
    def __init__(
        self,
        *,
        message: str | None = None,
        span: tuple[int, int] | _EOF = EOF,
    ) -> None:
        if message is None:
            message = "Expected expression."
        super().__init__(message, span)


class UnexpectedPrimaryError(ParsingError):
    def __init__(self, token: Token):
        super().__init__(f"Expected expression, got: {token.token}.", token.span())


class UndefinedUnitError(ParsingError):
    def __init__(self, unit_token: UnitToken) -> None:
        super().__init__(f"Invalid unit {unit_token.token}", unit_token.span())


class ParsingErrorGroup:
    def __init__(self, errors: list[ParsingError] | None = None) -> None:
        if errors is None:
            errors = []
        self.errors: list[ParsingError] = errors

    def __bool__(self) -> bool:
        return bool(self.errors)

    def add(self, error: ParsingError | ParsingErrorGroup) -> None:
        if isinstance(error, ParsingErrorGroup):
            self.errors.extend(error.errors)
        else:
            self.errors.append(error)

    @override
    def __repr__(self) -> str:
        messages = "\n".join(str(e) for e in self.errors)
        return f"ParsingErrorGroup(\n{messages}\n)"


class EvaluationError(Exception):
    def __init__(self, message: str, expression: Expression) -> None:
        super().__init__(message)
        self._expr: Expression | None = expression


def parse(input: str) -> Result[Expression, ParsingError | ParsingErrorGroup]:
    tokens = list(tokenize(input))
    result = _parse_expr(deque(tokens))
    return result


def _parse_expr(
    tokens: deque[Token],
) -> Result[Expression, ParsingError | ParsingErrorGroup]:
    return _parse_sum(tokens)

    error_group = ParsingErrorGroup()
    expressions: deque[Expression] = deque()

    while tokens:
        expr_result = _parse_sum(tokens)
        if isinstance(expr_result, Err):
            error_group.add(expr_result.err_value)
        else:
            expressions.append(expr_result.ok_value)
    if error_group:
        return Err(error_group)

    while len(expressions) > 1:
        left = expressions.popleft()
        right = expressions.popleft()
        if left.dimensionality() == right.dimensionality():
            new_expr = Binary(left, OperatorType.ADD, right, implicit=True)
        else:
            new_expr = Binary(left, OperatorType.MUL, right, implicit=True)
        expressions.appendleft(new_expr)
    return Ok(expressions[0])


def _parse_binary(
    tokens: deque[Token], ops: tuple[OperatorType, ...]
) -> Result[Expression, ParsingError | ParsingErrorGroup]:
    """
    A generic algorithm for parsing binary operations.

    The expression gets parsed in the following way:
    1. Decide what the term is.
        - If we're parsing a sum, we look for unary operations (see `_parse_unary`).
            - This way `-3 + 4` doesn't turn into `-(3 + 4)`, but we guarantee multiplication and exponentiation have higher precedence.
        - If we're parsing factors, look for smaller exponent expressions.
        - If we're parsing an exponentiation, look for "primary" terms (see `_parse_primary`).
    2. Parse one term (the starting left term).
    3. Repeatedly pop an operator and try to parse the `right` term. Then construct a Binary with the current terms.

    # Raises
    - `ParsingErrorGroup`
        - Any errors propagated from parsing terms (`_parse_unary` and `_parse_primary`) are propagated up.
        - Errors get reported for all terms, hopefully this makes it easier to debug and use.
    """

    error_group = ParsingErrorGroup()

    match ops[0]:
        case OperatorType.ADD | OperatorType.SUB:
            parse_term = _parse_unary
        case OperatorType.MUL | OperatorType.DIV:
            parse_term = _parse_exp
        case OperatorType.EXP:
            parse_term = _parse_primary

    term = parse_term(tokens)
    if isinstance(term, Err):
        error_group.add(term.err_value)

    while tokens:
        token = tokens[0]
        op_type = None
        if isinstance(token, UnknownToken):
            expected = "operator or group expression"
            error_group.add(UnexpectedTokenError(token, expected=expected))
            _ = tokens.popleft()
        elif isinstance(token, OperatorToken):
            if token.op_type not in ops:
                break
            op_type = token.op_type
            _ = tokens.popleft()
        elif OperatorType.MUL in ops:
            op_type = OperatorType.MUL
        else:
            break

        right = parse_term(tokens)
        if isinstance(right, Err):
            error_group.add(right.err_value)

        elif op_type is not None:
            if isinstance(term, Ok):
                term = Ok(Binary(term.ok_value, op_type, right.ok_value))

    if error_group:
        return Err(error_group)
    return term


def _parse_sum(
    tokens: deque[Token],
) -> Result[Expression, ParsingError | ParsingErrorGroup]:
    return _parse_binary(tokens, (OperatorType.ADD, OperatorType.SUB))


def _parse_mul(
    tokens: deque[Token],
) -> Result[Expression, ParsingError | ParsingErrorGroup]:
    return _parse_binary(tokens, (OperatorType.MUL, OperatorType.DIV))


def _parse_exp(
    tokens: deque[Token],
) -> Result[Expression, ParsingError | ParsingErrorGroup]:
    return _parse_binary(tokens, (OperatorType.EXP,))


def _parse_unary(
    tokens: deque[Token],
) -> Result[Expression, ParsingError | ParsingErrorGroup]:
    """
    Parse a sequence of unary operations, the tree is created from the innermost operation.

    # Example
    - `+-3` turns into `Unary(ADD, Unary(SUB, Float(3)))`.

    # Errors
    - `InvalidUnaryError`
        - Any operator not found in the unary operator map `_UNARY_OP_MAP` is considered invalid.
    - `ExpectedPrimaryError`
        - No term or "value" was found for the operation, which means there are no more tokens.
    - `ParsingErrorGroup`
        - Any errors propagated from parsing higher precedence terms (multiplication `_parse_mul`, and exponentiation `_parse_exp`)
    """
    op_list: list[OperatorToken] = []
    while tokens:
        token = tokens[0]
        if isinstance(token, UnknownToken):
            _ = tokens.popleft()
            return Err(UnexpectedTokenError(token, expected="expression"))
        if not isinstance(token, OperatorToken):
            value = _parse_mul(tokens)
            if isinstance(value, Err):
                return value
            break
        if token.op_type not in _UNARY_OP_MAP:
            return Err(InvalidUnaryError(token))
        _ = tokens.popleft()
        op_list.append(token)
    else:
        return Err(ExpectedPrimaryError())

    if not op_list:
        return value

    value = value.unwrap()
    while op_list:
        op = op_list.pop()
        value = Unary(op.op_type, value, op.start)
    return Ok(value)


def _parse_primary(
    tokens: deque[Token],
) -> Result[Expression, ParsingError | ParsingErrorGroup]:
    """
    Parses a:
    - `Group`: any expression in parenthesis, brackets or braces gets parsed recursively
    - `Float`: a floating point number
    - `Unit`: a string/identifier, units are not checked while parsing, only while evaluating the tree
    - "Primary chain": Series of Floats and Units which get parsed with implicit binary operations
        - See `_parse_primary_chain` for details.

    # Errors
    - `ExpectedPrimaryError`
        - We reached the end of the token stream, but we're expecting a primary or a group term.
    - `UnmatchedParenError`
        - Unmatched opening or closing group (parenthesis, brackets or braces).
        - Expressions inside a group can only be parsed if every paren has a pair.
    - `UnexpectedPrimaryError`
        - Raised from `_parse_primary_chain`.
        - Raised when we're expecting a Float, Unit or primary chain but we encounter an unexpected token.
        - Raised when a primary chain does not follow the format `Float Unit Float Unit ...`.
    """
    if not tokens:
        return Err(ExpectedPrimaryError())
    token = tokens[0]

    if isinstance(token, ParenToken):
        return _parse_group(tokens)
    else:
        return _parse_primary_chain(tokens)


def _parse_group(
    tokens: deque[Token],
) -> Result[Group, ParsingError | ParsingErrorGroup]:
    opening_pair = tokens.popleft()
    if not isinstance(opening_pair, ParenToken):
        raise ValueError(f"Expected a paren token, got {opening_pair} instead")

    if not opening_pair.paren_type.is_opening():
        return Err(UnmatchedParenError(opening_pair))

    group_tokens: deque[Token] = deque()
    pairs_open = 1
    while tokens:
        token = tokens.popleft()
        if isinstance(token, ParenToken):
            if token.paren_type == opening_pair.paren_type:
                pairs_open += 1
            elif token.paren_type.is_pair(opening_pair.paren_type):
                pairs_open -= 1
        if pairs_open == 0:
            break
        group_tokens.append(token)
    else:
        return Err(UnmatchedParenError(opening_pair))

    closing_pair = token
    if not group_tokens:
        start = opening_pair.start
        end = closing_pair.end
        if start == end:
            end += 1
        return Err(EmptyGroupExpression(span=(start, end)))

    # Since groups get parsed without any context of  the outer expression,
    # they can raise errors EOL errors even when it's not actually the end of the expression.
    # e.g. `(9 / ) + (4 * )`
    #           ^        ^ No tokens left after the operators so they report EOL
    # We solve this by changing every EOL to a span between the last token and paren
    last_group_token = group_tokens[-1]
    expr = _parse_expr(group_tokens)
    if isinstance(expr, Err):
        error = expr.err_value
        start = last_group_token.end
        end = closing_pair.start
        if end == start:
            end += 1
        if isinstance(error, ParsingError):
            if error.span == EOF:
                error.span = (start, end)
            return Err(error)
        for err in error.errors:
            if err.span == EOF:
                err.span = (start, end)
        return expr

    return Ok(
        Group(
            expr.unwrap(),
            opening_pair.paren_type,
            opening_pair.start,
            closing_pair.end,
        )
    )


def _parse_primary_chain(
    tokens: deque[Token],
) -> Result[Binary | Primary, ParsingError | ParsingError | ParsingErrorGroup]:
    """
    Parses a Float, Unit or chain of the form `Float Unit Float Unit ...`, for implicit operations (rules explained below).

    # Parsing rules:
    - If the algorithm only finds one primary element (a FloatToken or UnitToken) it returns Float or Unit
    - Otherwise, it tries to parse a "primary chain" of the form `Float Unit Float Unit ...`.
    - The chain has to start with a Float.
    - Floats and Units must come in pairs so `3m 14` or `5ft in` are not valid chains.
    - Units must be "simple" e.g. `km`, `Newton`.. Compound units such as `km/h` are currently not parsed this way.
    - Only units of the same dimension e.g. `5km 3cm` will get parsed.
        - Sequences like `5km 3h` will only parse into `5*km`, while `3h` will be left in the deque.
    # Implicit operations:
    - When a `Float` comes before a `Unit` they are multiplied. (`3 km` -> `3 * km`).
    - `Float Unit` pairs get added (`3 ft 4 in` -> `3*ft 4*in` -> `3*ft + 4*in`).

    # Errors
    - `ExpectedPrimaryError`
        - We reached the end of the expression, but are still expecting a primary.
    - `UnexpectedPrimaryError`
        - First token encountered was not a primary.
    - `ParsingErrorGroup`
        - Group of `ExpectedPrimaryError` because a primary chain must follow the format `Float Unit Float Unit`, "out of place" tokens are reported.
    """
    token = tokens[0]
    if not isinstance(token, (UnitToken, FloatToken)):
        return Err(UnexpectedPrimaryError(token))
    if len(tokens) == 1 or not isinstance(
        tokens[1], (FloatToken, UnitToken, ParenToken)
    ):
        _ = tokens.popleft()
        return Primary.from_token(token)

    error_group = ParsingErrorGroup()
    primaries: deque[Binary] = deque()
    previous_float_error = False
    previous_unit_error = False

    while tokens:
        first = tokens[0]
        if isinstance(first, UnitToken):
            if not previous_float_error:
                message = f"Expected a number before unit '{first.token}'"
                error_group.add(
                    ExpectedPrimaryError(message=message, span=first.span())
                )
            _ = tokens.popleft()
            previous_float_error = True
            continue
        elif not isinstance(first, FloatToken):
            break
        number = Float(first.to_float(), *first.span())
        _ = tokens.popleft()

        if not isinstance(tokens[0], UnitToken):
            break
        unit_res = _parse_unit(tokens, first)
        if isinstance(unit_res, Err):
            if not previous_unit_error:
                error_group.add(unit_res.err_value)
            previous_unit_error = True
            continue
        unit = unit_res.ok_value

        while tokens:
            token = tokens[0]
            if isinstance(token, UnitToken):
                unit_res = _parse_unit(tokens, first)
                if isinstance(unit_res, Err):
                    if not previous_unit_error:
                        error_group.add(unit_res.err_value)
                    previous_unit_error = True
                    continue
                curr_unit = unit_res.ok_value

                if unit.dimensionality() == curr_unit.dimensionality():
                    message = "Expected a number between units of the same dimension."
                    error_group.add(
                        ExpectedPrimaryError(message=message, span=curr_unit.span())
                    )
                    previous_unit_error = True
                else:
                    unit = Binary(unit, OperatorType.MUL, curr_unit, implicit=True)
            else:
                break
            previous_unit_error = False

        primaries.append(Binary(number, OperatorType.MUL, unit, implicit=True))
        previous_float_error = False
        previous_unit_error = False

    if error_group:
        return Err(error_group)

    while len(primaries) > 1:
        left = primaries.popleft()
        right = primaries.popleft()
        if left.dimensionality() == right.dimensionality():
            op = OperatorType.ADD
        else:
            op = OperatorType.MUL
        new_expr = Binary(left, op, right, implicit=True)
        primaries.appendleft(new_expr)
    return Ok(primaries[0])


def _parse_unit(
    tokens: deque[Token], last_float_token: FloatToken
) -> Result[Unit | Binary, ParsingError | ParsingErrorGroup]:
    if not tokens:
        message = f"Expected a unit after '{last_float_token.token}'"
        return Err(ExpectedPrimaryError(message=message))
    first = tokens[0]
    if not isinstance(first, UnitToken):
        message = f"Expected a unit after the number '{last_float_token.token}', got '{first.token}'."
        return Err(ExpectedPrimaryError(message=message, span=first.span()))

    _ = tokens.popleft()
    unit_res = Unit.try_new(first)
    if isinstance(unit_res, Err):
        return unit_res

    unit = unit_res.ok_value
    if not (
        len(tokens) >= 2
        and isinstance(tokens[0], OperatorToken)
        and isinstance(tokens[1], (FloatToken, UnitToken, ParenToken))
    ):
        return Ok(unit)

    # If we have `Unit ** Float` or `Unit ** (<Expr>)` we want to treat this as Binary(Unit, MUL, ...)
    op = tokens[0].op_type
    if op not in (OperatorType.EXP, OperatorType.DIV):
        return Ok(unit)

    right_token = tokens[1]

    if isinstance(right_token, ParenToken):
        _ = tokens.popleft()
        right_res = _parse_group(tokens)
        if isinstance(right_res, Err):
            return right_res
        right = right_res.ok_value
    else:
        if op == OperatorType.DIV and isinstance(right_token, FloatToken):
            return Ok(unit)
        elif op == OperatorType.EXP and isinstance(right_token, UnitToken):
            message = f"Expected a number or group expression as exponent, got {right_token.token}"
            return Err(ExpectedPrimaryError(message=message, span=right_token.span()))
        _ = tokens.popleft()
        _ = tokens.popleft()
        right_res = Primary.from_token(right_token)
        if isinstance(right_res, Err):
            return right_res
        right = right_res.ok_value

    return Ok(Binary(unit, op, right))
