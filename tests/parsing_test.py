# pyright: reportPrivateUsage=false

from collections import deque

from result import Err, Ok

from pint import Quantity
from absolute_unit.parsing import (
    Binary,
    CharStream,
    ExpectedPrimaryError,
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


def test_char_stream() -> None:
    stream = CharStream(" 1.2345 big   string 3.13")
    string = "".join(stream)
    assert string == " 1.2345 big   string 3.13"


def test_float_token() -> None:
    float_token = FloatToken("3.393", 0, 0)
    assert float_token.to_float() == 3.393


def test_float_token_consume() -> None:
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
    token_stream = tokenize("6 kilometer / 3 hour")
    token = next(token_stream)
    assert token is not None and token.span() == (0, 1)
    token = next(token_stream)
    assert token is not None and token.span() == (2, 11)
    token = next(token_stream)
    assert token is not None and token.span() == (12, 13)


def test_primary_single() -> None:
    tokens: deque[Token] = deque([FloatToken("6.68", 0, 0)])
    parsed = _parse_primary(tokens)
    mock_result = Float(6.68, 0, 0)
    assert isinstance(parsed, Ok)
    assert parsed.ok_value == mock_result
    assert not tokens


def test_primary_group() -> None:
    tokens: deque[Token] = deque(
        [
            ParenToken("(", 0, 0),
            ParenToken("{", 0, 0),
            FloatToken("6.68", 0, 0),
            ParenToken("}", 0, 0),
            ParenToken(")", 0, 0),
        ]
    )
    parsed = _parse_primary(tokens)
    mock_result = Float(6.68, 0, 0)
    assert isinstance(parsed, Ok)
    assert parsed.ok_value == mock_result
    assert not tokens


def test_primary_chain() -> None:
    tokens: deque[Token] = deque(
        [
            FloatToken("3", 0, 0),
            UnitToken("m", 0, 0),
            FloatToken("14", 0, 0),
            UnitToken("cm", 0, 0),
        ]
    )
    mock_result = Binary(
        Binary(
            Float(3, 0, 0),
            OperatorType.MUL,
            Unit(Quantity("m"), "m", 0, 0),
        ),
        OperatorType.ADD,
        Binary(
            Float(14, 0, 0),
            OperatorType.MUL,
            Unit(Quantity("cm"), "cm", 0, 0),
        ),
    )
    parsed = _parse_primary_chain(tokens)
    assert isinstance(parsed, Ok)
    assert parsed.ok_value == mock_result
    assert not tokens


def test_primary_unmatched_closing_paren_error() -> None:
    tokens: deque[Token] = deque(tokenize(")(())"))
    result = _parse_primary(tokens)
    assert isinstance(result, Err)
    errors = result.err_value
    assert isinstance(errors[0], UnmatchedParenError)
    assert not tokens


def test_primary_unmatched_opening_paren_error() -> None:
    tokens: deque[Token] = deque([ParenToken("(", 0, 0), UnitToken("bla", 0, 0)])
    result = _parse_primary(tokens)
    assert isinstance(result, Err)
    errors = result.err_value
    assert isinstance(errors[0], UnmatchedParenError)
    assert not tokens


def test_primary_unknown_primary_error_error() -> None:
    tokens: deque[Token] = deque([OperatorToken("*", 0, 0)])
    result = _parse_primary(tokens)
    assert isinstance(result, Err)
    errors = result.err_value
    assert isinstance(errors[0], UnexpectedPrimaryError)


def test_primary_expected_float_error() -> None:
    """
    The chain "m 6 ft" is invalid because the first unit is missing a number in front.
    """
    tokens: deque[Token] = deque(
        [UnitToken("m", 0, 0), FloatToken("6", 0, 0), UnitToken("ft", 0, 0)]
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
            FloatToken("6", 0, 0),
            FloatToken("3", 0, 0),
            UnitToken("ft", 0, 0),
            UnitToken("m", 0, 0),
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
            OperatorToken("-", 0, 0),
            OperatorToken("-", 0, 0),
            OperatorToken("+", 0, 0),
            FloatToken("6.3", 0, 0),
        ]
    )
    parsed = _parse_unary(tokens)
    mock_result = Unary(
        OperatorType.SUB,
        Unary(OperatorType.SUB, Unary(OperatorType.ADD, Float(6.3, 0, 0), 0), 0),
        0,
    )
    assert isinstance(parsed, Ok)
    assert parsed.ok_value == mock_result
    assert not tokens


def test_unary_invalid_unary_error() -> None:
    tokens: deque[Token] = deque([OperatorToken("*", 0, 0), FloatToken("6.68", 0, 0)])
    result = _parse_unary(tokens)
    assert isinstance(result, Err)
    errors = result.err_value
    assert isinstance(errors[0], InvalidUnaryError)


def test_binary_parse() -> None:
    tokens: deque[Token] = deque(
        [
            FloatToken("4.5", 0, 0),
            OperatorToken("+", 0, 0),
            OperatorToken("-", 0, 0),
            FloatToken("3.6", 0, 0),
        ]
    )
    parsed = _parse_expr(tokens)
    mock_result = Binary(
        Float(4.5, 0, 0), OperatorType.ADD, Unary(OperatorType.SUB, Float(3.6, 0, 0), 0)
    )
    assert isinstance(parsed, Ok)
    assert parsed.ok_value == mock_result
    assert not tokens


def test_primary_chain_simple() -> None:
    tokens: deque[Token] = deque(
        [
            FloatToken("30", 0, 0),
            UnitToken("km", 0, 0),
            OperatorToken("/", 0, 0),
            FloatToken("2", 0, 0),
            UnitToken("h", 0, 0),
        ]
    )
    parsed = _parse_expr(tokens)
    mock_result = Binary(
        Binary(
            Float(30, 0, 0),
            OperatorType.MUL,
            Unit(Quantity("km"), "km", 0, 0),
        ),
        OperatorType.DIV,
        Binary(
            Float(2, 0, 0),
            OperatorType.MUL,
            Unit(Quantity("h"), "h", 0, 0),
        ),
    )
    assert isinstance(parsed, Ok)
    assert parsed.ok_value == mock_result
    assert not tokens


def test_primary_chain_complex() -> None:
    tokens: deque[Token] = deque(
        [
            FloatToken("1", 0, 0),
            UnitToken("km", 0, 0),
            ParenToken("(", 0, 0),
            FloatToken("5", 0, 0),
            OperatorToken("+", 0, 0),
            FloatToken("3", 0, 0),
            ParenToken(")", 0, 0),
            UnitToken("m", 0, 0),
            OperatorToken("/", 0, 0),
            FloatToken("2", 0, 0),
            UnitToken("h", 0, 0),
            FloatToken("13", 0, 0),
            UnitToken("min", 0, 0),
        ]
    )
    parsed = _parse_expr(tokens)
    mock_result = Binary(
        Binary(
            Binary(
                Float(1, 0, 0),
                OperatorType.MUL,
                Unit(Quantity("km"), "km", 0, 0),
            ),
            OperatorType.ADD,
            Binary(
                Group(
                    Binary(
                        Float(5, 0, 0),
                        OperatorType.ADD,
                        Float(3, 0, 0),
                    ),
                    ParenType.L_PAREN,
                    0,
                    0,
                ),
                OperatorType.MUL,
                Unit(Quantity("m"), "m", 0, 0),
            ),
        ),
        OperatorType.DIV,
        Binary(
            Binary(
                Float(2, 0, 0),
                OperatorType.MUL,
                Unit(Quantity("h"), "h", 0, 0),
            ),
            OperatorType.ADD,
            Binary(
                Float(13, 0, 0),
                OperatorType.MUL,
                Unit(Quantity("min"), "min", 0, 0),
            ),
        ),
    )
    assert isinstance(parsed, Ok)
    assert parsed.ok_value == mock_result
    assert not tokens


def test_parse_unit_standalone() -> None:
    tokens: deque[Token] = deque([UnitToken("km", 0, 0)])
    parsed = _parse_unit(tokens)
    assert isinstance(parsed, Ok)
    mock_result = Unit(Quantity("km"), "km", 0, 0)
    assert parsed.ok_value == mock_result
    assert not tokens


def test_invalid_unit() -> None:
    result = Unit.try_new(UnitToken("fsdjlksdaf", 0, 0))
    assert isinstance(result, Err)
    assert isinstance(result.err_value, UndefinedUnitError)


def test_parse_unit_invalid_unit() -> None:
    tokens: deque[Token] = deque(
        [
            UnitToken("abc", 0, 0),
            OperatorToken("/", 0, 0),
            UnitToken("def", 0, 0),
        ]
    )
    result = _parse_unit(tokens)
    assert isinstance(result, Err)
    errors = result.err_value
    assert isinstance(errors[0], UndefinedUnitError)
    assert isinstance(errors[1], UndefinedUnitError)


def test_parse_unit_power_float() -> None:
    tokens: deque[Token] = deque(
        [UnitToken("km", 0, 0), OperatorToken("**", 0, 0), FloatToken("2", 0, 0)]
    )
    parsed = _parse_unit(tokens)
    assert isinstance(parsed, Ok)
    mock_result = Binary(
        Unit(Quantity("km"), "km", 0, 0), OperatorType.EXP, Float(2, 0, 0)
    )
    assert parsed.ok_value == mock_result
    assert not tokens


def test_parse_unit_power_error() -> None:
    tokens: deque[Token] = deque(
        [UnitToken("km", 0, 0), OperatorToken("**", 0, 0), UnitToken("hr", 0, 0)]
    )
    result = _parse_unit(tokens)
    assert isinstance(result, Err)
    errors = result.err_value
    assert isinstance(errors[0], ExpectedPrimaryError)
    assert not tokens


def test_parse_unit_power_groupexpr() -> None:
    tokens: deque[Token] = deque(
        [
            UnitToken("km", 0, 0),
            OperatorToken("**", 0, 0),
            ParenToken("(", 0, 0),
            FloatToken("1", 0, 0),
            OperatorToken("+", 0, 0),
            FloatToken("1", 0, 0),
            ParenToken(")", 0, 0),
        ]
    )
    parsed = _parse_unit(tokens)
    assert isinstance(parsed, Ok)
    # mock_result: km ** (1 + 1)
    mock_result = Binary(
        Unit(Quantity("km"), "km", 0, 0),
        OperatorType.EXP,
        Group(
            Binary(Float(1, 0, 0), OperatorType.ADD, Float(1, 0, 0)),
            ParenType.L_PAREN,
            0,
            0,
        ),
    )
    assert parsed.ok_value == mock_result
    assert not tokens
