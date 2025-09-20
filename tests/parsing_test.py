# pyright: reportPrivateUsage=false

from collections import deque

from result import Err, Ok

from pint import Quantity
from absolute_unit.parsing import (
    Binary,
    CharStream,
    ExpectedPrimaryError,
    Expression,
    Float,
    FloatToken,
    Group,
    InvalidUnaryError,
    OperatorToken,
    OperatorType,
    ParenToken,
    ParenType,
    Token,
    Unary,
    UndefinedUnitError,
    Unit,
    UnitToken,
    UnexpectedPrimaryError,
    UnknownToken,
    UnmatchedParenError,
    Whitespace,
    _parse_primary,
    _parse_unary,
    _parse_primary_chain,
    _parse_unit,
    _parse_expr,
    tokenize,
)


def float_token(value: float) -> FloatToken:
    return FloatToken(str(value), 0, 0)


def float_mock(value: float) -> Float:
    return Float(value, 0, 0)


def unit_token(unit: str) -> UnitToken:
    return UnitToken(unit, 0, 0)


def unit_mock(unit: str) -> Unit:
    return Unit(Quantity(unit), unit, 0, 0)


def unary_mock(op_type: OperatorType, expr: Expression) -> Unary:
    return Unary(op_type, expr, 0)


op_plus = OperatorToken("+", 0, 0)
op_minus = OperatorToken("-", 0, 0)
op_mul = OperatorToken("*", 0, 0)
op_div = OperatorToken("/", 0, 0)
op_exp = OperatorToken("**", 0, 0)


def group_mock(paren_type: ParenType, expr: Expression) -> Group:
    return Group(expr, paren_type, 0, 0)


left_paren = ParenToken("(", 0, 0)
right_paren = ParenToken(")", 0, 0)
left_bracket = ParenToken("[", 0, 0)
right_bracket = ParenToken("]", 0, 0)
left_brace = ParenToken("{", 0, 0)
right_brace = ParenToken("}", 0, 0)


def test_char_stream() -> None:
    """Test whether the CharStream iteration works properly."""
    stream = CharStream(" 1.2345 big   string 3.13")
    string = "".join(stream)
    assert string == " 1.2345 big   string 3.13"


def test_float_token() -> None:
    float_token = FloatToken("3.393", 0, 0)
    assert float_token.to_float() == 3.393


def test_float_token_consume() -> None:
    """Test whether the FloatToken.consume method works as intended."""
    token = Token.from_stream(CharStream("3.393"))
    assert isinstance(token, FloatToken) and token.token == "3.393"


def test_unit_token() -> None:
    unit_token = UnitToken("feet", 0, 0)
    assert unit_token.token == "feet"


def test_unit_token_consume() -> None:
    token = Token.from_stream(CharStream("km"))
    assert isinstance(token, UnitToken) and token.token == "km"


def test_paren_token() -> None:
    paren_token = ParenToken("(", 0, 0)
    assert paren_token.paren_type == ParenType.L_PAREN
    paren_token = ParenToken(")", 0, 0)
    assert paren_token.paren_type == ParenType.R_PAREN


def test_paren_token_consume() -> None:
    stream = CharStream("()")
    token = Token.from_stream(stream)
    assert isinstance(token, ParenToken) and token.token == "("
    token = Token.from_stream(stream)
    assert isinstance(token, ParenToken) and token.token == ")"


def test_operator_token() -> None:
    op_token = OperatorToken("+", 0, 0)
    assert op_token.op_type == OperatorType.ADD
    op_token = OperatorToken("*", 0, 0)
    assert op_token.op_type == OperatorType.MUL
    op_token = OperatorToken("-", 0, 0)
    assert op_token.op_type == OperatorType.SUB
    op_token = OperatorToken("**", 0, 0)
    assert op_token.op_type == OperatorType.EXP
    op_token = OperatorToken("/", 0, 0)
    assert op_token.op_type == OperatorType.DIV


def test_operator_token_consume() -> None:
    """Primarily intended to check whether ** gets tokenized to OperatorType.MUL."""
    stream = CharStream("*-**/")
    token = Token.from_stream(stream)
    assert isinstance(token, OperatorToken) and token.op_type == OperatorType.MUL
    token = Token.from_stream(stream)
    assert isinstance(token, OperatorToken) and token.op_type == OperatorType.SUB
    token = Token.from_stream(stream)
    assert isinstance(token, OperatorToken) and token.op_type == OperatorType.EXP
    token = Token.from_stream(stream)
    assert isinstance(token, OperatorToken) and token.op_type == OperatorType.DIV


def test_whitespace_token() -> None:
    """All whitespace should get ignored."""
    whitespace = Whitespace("", 0, 0)
    assert whitespace.token == ""


def test_whitespace_consume() -> None:
    stream = CharStream("   bla   \n\n\r")
    token = Token.from_stream(stream)
    assert isinstance(token, Whitespace) and token.token == ""
    token = Token.from_stream(stream)
    token = Token.from_stream(stream)
    assert isinstance(token, Whitespace) and token.token == ""


def test_unknown_token() -> None:
    """
    Anything not in the full Token alphabet is considered unknown.

    These are non-ascii, non-digit and non-operator (+, -, *, /) characters
    """
    unknown_token = UnknownToken("@#$;<><:", 0, 0)
    assert unknown_token.token == "@#$;<><:"


def test_tokenize() -> None:
    token_stream = tokenize("6 ft 1 in /   (4.3s * 13J)")
    token_strings = [t.token for t in token_stream]
    assert token_strings == [
        "6",
        "ft",
        "1",
        "in",
        "/",
        "(",
        "4.3",
        "s",
        "*",
        "13",
        "J",
        ")",
    ]


def test_token_span() -> None:
    """
    Test whether the token span matches up with the location in the input string.

    This is important for errors when parsing.
    """
    token_stream = tokenize("6 kilometer / 3 hour")
    token = next(token_stream)
    assert token is not None and token.span() == (0, 1)
    token = next(token_stream)
    assert token is not None and token.span() == (2, 11)
    token = next(token_stream)
    assert token is not None and token.span() == (12, 13)


def test_primary_single() -> None:
    tokens: deque[Token] = deque([float_token(6.68)])
    parsed = _parse_primary(tokens)
    mock_result = float_mock(6.68)
    assert isinstance(parsed, Ok)
    assert parsed.ok_value == mock_result
    assert not tokens


def test_primary_group() -> None:
    tokens: deque[Token] = deque(
        [
            left_paren,
            left_brace,
            float_token(6.68),
            right_brace,
            right_paren,
        ]
    )
    parsed = _parse_primary(tokens)
    mock_result = float_mock(6.68)
    assert isinstance(parsed, Ok)
    assert parsed.ok_value == mock_result
    assert not tokens


def test_primary_chain() -> None:
    tokens: deque[Token] = deque(
        [
            float_token(3),
            unit_token("m"),
            float_token(14),
            unit_token("cm"),
        ]
    )
    mock_result = Binary(
        Binary(
            float_mock(3),
            OperatorType.MUL,
            unit_mock("m"),
        ),
        OperatorType.ADD,
        Binary(
            float_mock(14),
            OperatorType.MUL,
            unit_mock("cm"),
        ),
    )
    parsed = _parse_primary_chain(tokens)
    assert isinstance(parsed, Ok)
    assert parsed.ok_value == mock_result
    assert not tokens


def test_group_unmatched_closing_paren_error() -> None:
    tokens: deque[Token] = deque(tokenize(")(())"))
    result = _parse_primary(tokens)
    assert isinstance(result, Err)
    errors = result.err_value
    assert isinstance(errors[0], UnmatchedParenError)


def test_group_unmatched_opening_paren_error() -> None:
    tokens: deque[Token] = deque(
        [
            left_paren,
            unit_token("m"),
        ]
    )
    result = _parse_primary(tokens)
    assert isinstance(result, Err)
    errors = result.err_value
    assert isinstance(errors[0], UnmatchedParenError)
    assert not tokens


def test_primary_unknown_primary_error() -> None:
    tokens: deque[Token] = deque([op_mul])
    result = _parse_primary(tokens)
    assert isinstance(result, Err)
    errors = result.err_value
    assert isinstance(errors[0], UnexpectedPrimaryError)


def test_primary_expected_float_error() -> None:
    """
    The chain "m 6 ft" is invalid because the first unit is missing a number in front.
    """
    tokens: deque[Token] = deque(
        [
            unit_token("m"),
            float_token(6),
            unit_token("ft"),
        ]
    )
    result = _parse_primary(tokens)
    assert isinstance(result, Err)
    errors = result.err_value
    assert isinstance(errors[0], ExpectedPrimaryError)
    assert "number before the unit" in str(errors)
    assert not tokens


def test_primary_chain_format_error() -> None:
    """
    The chain "6 3 ft m" is invalid because we're expecting a unit after the first '6', and a float after 'ft'.
    """
    tokens: deque[Token] = deque(
        [
            float_token(6),
            float_token(3),
            unit_token("ft"),
            unit_token("m"),
        ]
    )
    result = _parse_primary(tokens)
    assert isinstance(result, Err)
    errors = result.err_value
    error_0 = errors[0]
    assert isinstance(error_0, ExpectedPrimaryError) and "between numbers" in str(
        error_0
    )
    error_1 = errors[1]
    assert isinstance(error_1, ExpectedPrimaryError) and "number between units" in str(
        error_1
    )
    assert not tokens


def test_unary_parse() -> None:
    tokens: deque[Token] = deque(
        [
            op_minus,
            op_minus,
            op_plus,
            float_token(6.3),
        ]
    )
    parsed = _parse_unary(tokens)
    mock_result = unary_mock(
        OperatorType.SUB,
        unary_mock(
            OperatorType.SUB,
            unary_mock(
                OperatorType.ADD,
                float_mock(6.3),
            ),
        ),
    )
    assert isinstance(parsed, Ok)
    assert parsed.ok_value == mock_result
    assert not tokens


def test_unary_invalid_unary_error() -> None:
    tokens: deque[Token] = deque(
        [
            op_mul,
            float_token(6.68),
        ]
    )
    result = _parse_unary(tokens)
    assert isinstance(result, Err)
    errors = result.err_value
    assert isinstance(errors[0], InvalidUnaryError)


def test_binary_parse() -> None:
    tokens: deque[Token] = deque(
        [
            float_token(4.5),
            op_plus,
            float_token(3.6),
        ]
    )
    parsed = _parse_expr(tokens)
    mock_result = Binary(
        float_mock(4.5),
        OperatorType.ADD,
        float_mock(3.6),
    )
    assert isinstance(parsed, Ok)
    assert parsed.ok_value == mock_result
    assert not tokens


def test_primary_chain_simple() -> None:
    tokens: deque[Token] = deque(
        [
            float_token(30),
            unit_token("km"),
            op_div,
            float_token(2),
            unit_token("h"),
        ]
    )
    parsed = _parse_expr(tokens)
    mock_result = Binary(
        Binary(
            float_mock(30),
            OperatorType.MUL,
            unit_mock("km"),
        ),
        OperatorType.DIV,
        Binary(
            float_mock(2),
            OperatorType.MUL,
            unit_mock("h"),
        ),
    )
    assert isinstance(parsed, Ok)
    assert parsed.ok_value == mock_result
    assert not tokens


def test_primary_chain_complex() -> None:
    tokens: deque[Token] = deque(
        [
            float_token(1),
            unit_token("km"),
            left_paren,
            float_token(5),
            op_plus,
            float_token(3),
            right_paren,
            unit_token("m"),
            op_div,
            float_token(2),
            unit_token("h"),
            float_token(13),
            unit_token("min"),
        ]
    )
    parsed = _parse_expr(tokens)
    mock_result = Binary(
        Binary(
            Binary(
                Binary(
                    float_mock(1),
                    OperatorType.MUL,
                    unit_mock("km"),
                ),
                OperatorType.MUL,
                group_mock(
                    ParenType.L_PAREN,
                    Binary(
                        float_mock(5),
                        OperatorType.ADD,
                        float_mock(3),
                    ),
                ),
            ),
            OperatorType.MUL,
            unit_mock("m"),
        ),
        OperatorType.DIV,
        Binary(
            Binary(
                float_mock(2),
                OperatorType.MUL,
                unit_mock("h"),
            ),
            OperatorType.ADD,
            Binary(
                float_mock(13),
                OperatorType.MUL,
                unit_mock("min"),
            ),
        ),
    )
    assert isinstance(parsed, Ok)
    assert parsed.ok_value == mock_result
    assert not tokens


def test_parse_unit_standalone() -> None:
    parsed = _parse_unit(deque(), first=unit_token("km"))
    assert isinstance(parsed, Ok)
    mock_result = unit_mock("km")
    assert parsed.ok_value == mock_result


def test_invalid_unit() -> None:
    result = Unit.try_new(unit_token("dfdasf"))
    assert isinstance(result, Err)
    assert isinstance(result.err_value, UndefinedUnitError)


def test_parse_unit_invalid_unit() -> None:
    tokens: deque[Token] = deque(
        [
            op_div,
            unit_token("def"),
        ]
    )
    result = _parse_unit(tokens, first=unit_token("abc"))
    assert isinstance(result, Err)
    errors = result.err_value
    assert isinstance(errors[0], UndefinedUnitError)
    assert isinstance(errors[1], UndefinedUnitError)


def test_parse_unit_power_float() -> None:
    tokens: deque[Token] = deque([op_exp, float_token(2)])
    parsed = _parse_unit(tokens, first=unit_token("km"))
    assert isinstance(parsed, Ok)
    mock_result = Binary(
        unit_mock("km"),
        OperatorType.EXP,
        float_mock(2),
    )
    assert parsed.ok_value == mock_result
    assert not tokens


def test_parse_unit_power_error() -> None:
    tokens: deque[Token] = deque([op_exp, unit_token("km")])
    result = _parse_unit(tokens, first=unit_token("km"))
    assert isinstance(result, Err)
    errors = result.err_value
    assert isinstance(errors[0], ExpectedPrimaryError)
    assert not tokens


def test_parse_unit_power_groupexpr() -> None:
    tokens: deque[Token] = deque(
        [
            op_exp,
            left_paren,
            float_token(1),
            op_plus,
            float_token(1),
            right_paren,
        ]
    )
    parsed = _parse_unit(tokens, first=unit_token("km"))
    assert isinstance(parsed, Ok)
    # mock_result: km ** (1 + 1)
    mock_result = Binary(
        unit_mock("km"),
        OperatorType.EXP,
        group_mock(
            ParenType.L_PAREN,
            Binary(
                float_mock(1),
                OperatorType.ADD,
                float_mock(1),
            ),
        ),
    )
    assert parsed.ok_value == mock_result
    assert not tokens


def test_binary_dimensionality_error() -> None:
    left = Float(1.0, 0, 0)
    right = Unit(Quantity("km"), "km", 0, 0)
    op = OperatorType.ADD
    result = Binary.try_new(left, op, right)
    assert isinstance(result, Err)


def test_primary_chain_order() -> None:
    tokens: deque[Token] = deque(
        [
            float_token(1),
            op_div,
            float_token(2),
            op_exp,
            float_token(3),
            unit_token("cm"),
            op_exp,
            float_token(2),
        ]
    )
    result = _parse_expr(tokens)
    assert isinstance(result, Ok)
    mock_result = Binary(
        Binary(
            float_mock(1),
            OperatorType.DIV,
            Binary(
                float_mock(2),
                OperatorType.EXP,
                float_mock(3),
            ),
        ),
        OperatorType.MUL,
        Binary(
            unit_mock("cm"),
            OperatorType.EXP,
            float_mock(2),
        ),
    )
    assert result.ok_value == mock_result
