"""
This module is used for parsing user input from commands
"""

from __future__ import annotations

import abc
import enum
import operator
import pint
import string
from collections import deque
from collections.abc import Callable, Generator
from typing import ClassVar, Self, override

from pint import Quantity
from pint.facets.plain import PlainQuantity

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
    def evaluate(self) -> PlainQuantity[float]: ...

    @override
    def __eq__(self, other: object) -> bool: ...


class Binary(Expression):
    def __init__(
        self,
        left: Expression,
        op: OperatorType,
        right: Expression,
    ) -> None:
        self._left: Expression = left
        self._right: Expression = right
        self._op_type: OperatorType = op

    @property
    def left(self) -> Expression:
        return self._left

    @left.setter
    def left(self, new_left: Expression) -> None:
        self._left = new_left

    @property
    def right(self) -> Expression:
        return self._right

    @right.setter
    def right(self, new_right: Expression) -> None:
        self._right = new_right

    @property
    def op(self) -> OperatorType:
        return self._op_type

    @op.setter
    def op(self, new_op: OperatorType) -> None:
        self._op_type = new_op

    @override
    def evaluate(self) -> PlainQuantity[float]:
        op = _BINARY_OP_MAP[self.op]
        return op(self._left.evaluate(), self._right.evaluate())

    @override
    def __str__(self) -> str:
        if isinstance(self._left, Float) and isinstance(self._right, Unit):
            return f"{self._left}{self._right}"
        return f"{self._left} {self._op_type.value} {self._right}"

    @override
    def __repr__(self) -> str:
        return f"Binary({self._op_type} ({self._left} {self._right}))"

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Expression):
            raise TypeError(
                f"Can not compare {self.__class__.__qualname__} and {other.__class__.__qualname__}"
            )
        if not isinstance(other, Binary):
            return False
        return (
            self._left == other._left
            and self._op_type == other._op_type
            and self._right == other._right
        )


class Unary(Expression):
    def __init__(
        self, op: OperatorType, value: Binary | Unary | Primary | Group
    ) -> None:
        self._value: Binary | Unary | Primary | Group = value
        self._op_type: OperatorType = op

    @property
    def value(self) -> Binary | Unary | Primary | Group:
        return self._value

    @value.setter
    def value(self, new_value: Binary | Unary | Primary | Group) -> None:
        self._value = new_value

    @property
    def op(self) -> OperatorType:
        return self._op_type

    @op.setter
    def op(self, new_op: OperatorType) -> None:
        self._op_type = new_op

    @override
    def evaluate(self) -> PlainQuantity[float]:
        op = _UNARY_OP_MAP[self._op_type]
        return op(self._value.evaluate())

    @override
    def __str__(self) -> str:
        return f"{self._op_type.value}{self._value}"

    @override
    def __repr__(self) -> str:
        return f"Unary({self._op_type}{self._value})"

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Expression):
            raise TypeError(
                f"Can not compare {self.__class__.__qualname__} and {other.__class__.__qualname__}"
            )
        if not isinstance(other, Unary):
            return False
        return self._value == other._value and self._op_type == other._op_type


class Primary(Expression, abc.ABC):
    def __init__(self, span: tuple[int, int] = (0, 0)) -> None:
        self._span: tuple[int, int] = span

    def __neg__(self) -> Primary:
        return -self

    def span(self) -> tuple[int, int]:
        return self._span

    @staticmethod
    def from_token(token: UnitToken | FloatToken) -> Unit | Float:
        if isinstance(token, UnitToken):
            return Unit(token.token, token.span())
        return Float(token.to_float(), token.span())


class Float(Primary):
    def __init__(self, value: float, span: tuple[int, int] = (0, 0)) -> None:
        super().__init__(span)
        self._value: float = value

    @property
    def value(self) -> float:
        return self._value

    @override
    def evaluate(self) -> PlainQuantity[float]:
        return Quantity(self._value)

    @override
    def __str__(self) -> str:
        return str(self._value)

    @override
    def __repr__(self) -> str:
        return f"Float({self._value})"

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
    def __init__(self, unit: str, span: tuple[int, int] = (0, 0)) -> None:
        super().__init__(span)
        self.unit: str = unit

    @override
    def evaluate(self) -> PlainQuantity[float]:
        try:
            return Quantity(self.unit)
        except pint.UndefinedUnitError:
            raise InvalidUnitError(self)

    @override
    def __str__(self) -> str:
        return self.unit

    @override
    def __repr__(self) -> str:
        return f"Unit({self.unit})"

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Expression):
            raise TypeError(
                f"Can not compare {self.__class__.__qualname__} and {other.__class__.__qualname__}"
            )
        if not isinstance(other, Unit):
            return False
        return self.unit == other.unit


class Group(Expression):
    def __init__(self, expr: Expression, paren_type: ParenType):
        self._paren_type: ParenType = paren_type
        self._expr: Expression = expr

    @override
    def __str__(self) -> str:
        opening, closing = self._paren_type.to_pair()
        return f"{opening.value}{self._expr}{closing.value}"

    @override
    def __repr__(self) -> str:
        opening, closing = self._paren_type.to_pair()
        return f"Group({opening.value}{self._expr}{closing.value})"

    @override
    def evaluate(self) -> PlainQuantity[float]:
        return self._expr.evaluate()

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Expression):
            raise TypeError(
                f"Can not compare {self.__class__.__qualname__} and {other.__class__.__qualname__}"
            )
        if isinstance(other, Group):
            return self._expr == other._expr
        return self._expr == other


class ParsingError(Exception):
    def __init__(self, message: str, span: tuple[int, int] | _EOF = EOF):
        super().__init__(message)
        self.span: tuple[int, int] | _EOF = span


class UnknownTokenError(ParsingError):
    def __init__(self, token: UnknownToken) -> None:
        super().__init__(f"Unknown syntax: {token}.", token.span())


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
            f"Invalid unary operator: {operator_token}.", operator_token.span()
        )


class ExpectedPrimaryError(ParsingError):
    def __init__(
        self,
        *,
        message: str | None = None,
        expected: type[Float | Unit] | None = None,
        span: tuple[int, int] | _EOF = EOF,
    ) -> None:
        if message is None:
            if expected is None:
                message = "Expected float, unit or group expression."
            else:
                name = expected.__name__.lower()
                message = f"Expected {name}."
        super().__init__(message, span)


class UnexpectedPrimaryError(ParsingError):
    def __init__(self, token: Token, expected: type[Primary | Group] | None = None):
        if expected is None:
            name = "float, unit or group expression."
        else:
            name = expected.__name__.lower()
        super().__init__(f"Expected {name}, got: {token.token}", token.span())


class ParsingErrorGroup(ExceptionGroup):  # noqa: F821
    def __init__(self, errors: list[ParsingError]) -> None:
        self.errors: list[ParsingError] = errors
        super().__init__("Parsing errors:", errors)

    def __new__(cls, errors: list[ParsingError]) -> ParsingErrorGroup:
        self = super().__new__(cls, "Parsing errors:", errors)
        return self


class EvaluationError(Exception):
    def __init__(self, message: str, expression: Expression) -> None:
        super().__init__(message)
        self._expr: Expression | None = expression


class InvalidUnitError(EvaluationError):
    def __init__(self, unit: Unit) -> None:
        super().__init__(f"Invalid unit {unit}", unit)


def parse(input: str) -> Expression:
    tokens = list(tokenize(input))
    unknown_tokens = [t for t in tokens if isinstance(t, UnknownToken)]
    if unknown_tokens:
        errors: list[ParsingError] = [UnknownTokenError(t) for t in unknown_tokens]
        raise ParsingErrorGroup(errors)
    result = _parse_sum(deque(tokens))
    return result


def _parse_binary(
    tokens: deque[Token], ops: tuple[OperatorType, ...]
) -> Binary | Unary | Primary | Group:
    """
    A generic algorithm for parsing binary operations.

    The expression gets parsed in the following way:
    1. Decide what the term is.
        - If we're parsing a sum, we look for unary operations (see `_parse_unary`).
            - This way `-3 + 4` doesn't turn into `-(3 + 4)`, but we guarantee multiplication and exponentiation have higher precedence.
        - If we're parsing factors, look for smaller exponent expressions.
        - If we're parsing an exponentiation, look for "primary" terms (see `_parse_primary`).
    2. Collect a list of terms and operators.
        - For the expression `3 + 4 - 5` the lists gathered are `terms = [3, 4, 5]` and `operators = [+, -]`.
    4. Pop 2 terms, and one operator to combine into a Binary expression, then add that expression back to the terms list.
    5. Repeat step 4. until there is only one term.

    # Raises
    - `ParsingErrorGroup`
        - Any errors propagated from parsing terms (`_parse_unary` and `_parse_primary`) are propagated up.
        - Errors get reported for all terms, hopefully this makes it easier to debug and use.
    """

    errors: list[ParsingError] = []

    match ops[0]:
        case OperatorType.ADD | OperatorType.SUB:
            parse_term = _parse_unary
        case OperatorType.MUL | OperatorType.DIV:
            parse_term = _parse_exp
        case OperatorType.EXP:
            parse_term = _parse_primary

    term = None
    try:
        term = parse_term(tokens)
    except ParsingErrorGroup as error_group:
        errors.extend(error_group.errors)
    except ParsingError as e:
        errors.append(e)

    while tokens:
        token = tokens[0]
        if not isinstance(token, OperatorToken) or token.op_type not in ops:
            break
        _ = tokens.popleft()
        op_type = token.op_type

        right = None
        try:
            right = parse_term(tokens)
        except ParsingErrorGroup as error_group:
            errors.extend(error_group.errors)
        except ParsingError as e:
            errors.append(e)

        if term is not None and right is not None:
            term = Binary(term, op_type, right)

    if errors:
        raise ParsingErrorGroup(errors)
    assert term is not None
    return term


def _parse_sum(tokens: deque[Token]) -> Binary | Unary | Primary | Group:
    return _parse_binary(tokens, (OperatorType.ADD, OperatorType.SUB))


def _parse_mul(tokens: deque[Token]) -> Binary | Unary | Primary | Group:
    return _parse_binary(tokens, (OperatorType.MUL, OperatorType.DIV))


def _parse_exp(tokens: deque[Token]) -> Binary | Unary | Primary | Group:
    return _parse_binary(tokens, (OperatorType.EXP,))


def _parse_unary(tokens: deque[Token]) -> Binary | Unary | Primary | Group:
    """
    Parse a sequence of unary operations, the tree is created from the innermost operation.

    # Example
    - `+-3` turns into `Unary(ADD, Unary(SUB, Float(3)))`.

    # Raises
    - `InvalidUnaryError`
        - Any operator not found in the unary operator map `_UNARY_OP_MAP` is considered invalid.
    - `ExpectedPrimaryError`
        - No term or "value" was found for the operation, which means there are no more tokens.
    - `ParsingErrorGroup`
        - Any errors propagated from parsing higher precedence terms (multiplication `_parse_mul`, and exponentiation `_parse_exp`)
    """
    op_list: list[OperatorType] = []
    while tokens:
        token = tokens[0]
        if not isinstance(token, OperatorToken):
            value = _parse_mul(tokens)
            break
        if token.op_type not in _UNARY_OP_MAP:
            raise InvalidUnaryError(token)
        _ = tokens.popleft()
        op_list.append(token.op_type)
    else:
        raise ExpectedPrimaryError()

    if not op_list:
        return value

    while op_list:
        op = op_list.pop()
        value = Unary(op, value)
    return value


def _parse_primary(tokens: deque[Token]) -> Binary | Primary | Group:
    """
    Parses a:
    - `Group`: any expression in parenthesis, brackets or braces gets parsed recursively
    - `Float`: a floating point number
    - `Unit`: a string/identifier, units are not checked while parsing, only while evaluating the tree
    - "Primary chain": Series of Floats and Units which get parsed with implicit binary operations
        - See `_parse_primary_chain` for details.

    # Raises
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
        raise ExpectedPrimaryError()
    token = tokens[0]

    if isinstance(token, ParenToken):
        opening_pair = token

        if not opening_pair.paren_type.is_opening():
            raise UnmatchedParenError(opening_pair)

        _ = tokens.popleft()

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

        if pairs_open:
            raise UnmatchedParenError(opening_pair)

        closing_pair = token
        if not group_tokens:
            start = opening_pair.start
            end = closing_pair.end
            if start == end:
                end += 1
            raise EmptyGroupExpression(span=(start, end))

        # Since groups get parsed without any context of  the outer expression,
        # they can raise errors EOL errors even when it's not the end of the expression.
        # e.g. `(9 / ) + (4 * )`
        #           ^        ^ No tokens left after the operators so they report EOL
        # We solve this by changing every EOL to a span between the last token and paren
        last_group_token = group_tokens[-1]
        try:
            expr = _parse_sum(group_tokens)
        except ParsingErrorGroup as err_group:
            start = last_group_token.end
            end = closing_pair.start
            if end == start:
                end += 1
            for exc in err_group.errors:
                if exc.span == EOF:
                    exc.span = (start, end)
            raise
        return Group(expr, opening_pair.paren_type)

    return _parse_primary_chain(tokens)


def _parse_primary_chain(tokens: deque[Token]) -> Binary | Primary:
    """
    Parses a Float, Unit or chain of the form `Float Unit Float Unit ...`, for implicit operations (rules explained below).

    # Parsing rules:
    - If the algorithm only finds one primary element (a FloatToken or UnitToken) it returns Float or Unit
    - Otherwise, it tries to parse a "primary chain" of the form `Float Unit Float Unit ...`.
    - Floats and Units must come in pairs so `3m 14` or `5ft in` are not valid chains.
    - Any subsequence like `Float Float` or `Unit Unit` will produce an error.
    - The chain has to start with a Float.
    Implicit operations:
    - When a `Float` comes before a `Unit` they are multiplied. (`3 km` -> `3 * km`).
    - `Float Unit` pairs get added (`3 ft 4 in` -> `3 ft + 4 in` -> `3 * ft + 4 * in`).

    # Raises
    - `ExpectedPrimaryError`
        - We reached the end of the expression, but are still expecting a primary.
    - `UnexpectedPrimaryError`
        - First token encountered was not a primary.
    - `ParsingErrorGroup`
        - List of `ExpectedPrimaryError(expected=Float)` and `ExpectedPrimaryError(expected=Unit)`:
        - A primary chain must follow the format `Float Unit Float Unit`, any token which is "out of place" is reported.
    """
    errors: list[ParsingError] = []
    pairs: list[Binary] = []

    # Collect all the Floats and Units from the start
    # last_token_span is used to report where a missing primary was expected but not found.
    # e.g. a sequence like `Float Unit Float Operator` would report the operator as an unexpected primary, since a Float must be followed by a unit.
    primaries: deque[Primary] = deque()
    if len(tokens) == 1 or not isinstance(tokens[1], (UnitToken, FloatToken)):
        token = tokens[0]
        if not isinstance(token, (UnitToken, FloatToken)):
            raise UnexpectedPrimaryError(token)
        _ = tokens.popleft()
        return Primary.from_token(token)

    previous_float_error = False
    previous_unit_error = False
    while tokens:
        first = tokens[0]
        if not isinstance(first, (FloatToken, UnitToken)):
            break

        _ = tokens.popleft()
        if not isinstance(first, FloatToken):
            if not previous_float_error:
                errors.append(
                    ExpectedPrimaryError(
                        message=f"Expected float before unit '{first.token}'.",
                        span=first.span(),
                    )
                )
            previous_float_error = True
            continue

        second = None
        second_span = EOF
        if tokens:
            second = tokens[0]
            second_span = second.span()
        if not isinstance(second, UnitToken):
            if not previous_unit_error:
                message = "Expected a unit after a float"
                if second is not None:
                    message += f", got '{second.token}'"
                message += "."
                errors.append(ExpectedPrimaryError(message=message, span=second_span))
            previous_unit_error = True
            continue
        _ = tokens.popleft()

        primaries.append(Primary.from_token(first))
        primaries.append(Primary.from_token(second))
        previous_float_error = False
        previous_unit_error = False

    if errors:
        raise ParsingErrorGroup(errors)

    # Turn collected terms into implicit operations
    while primaries:
        left = primaries.popleft()
        right = primaries.popleft()
        pairs.append(Binary(left, OperatorType.MUL, right))
    while len(pairs) >= 2:
        right_binary = pairs.pop()
        left_binary = pairs.pop()
        pairs.append(Binary(left_binary, OperatorType.ADD, right_binary))
    return pairs[0]
