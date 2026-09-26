# pyright: reportPrivateUsage=false

from collections import deque

from pint import UnitRegistry
from result import Err, Ok

from absolute_unit.parsing import (
    Binary,
    CharStream,
    DimensionalityError,
    DivisionByZeroError,
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
    Parser,
    ParserMode,
    Token,
    Unary,
    UndefinedUnitError,
    UnexpectedTokenError,
    Unit,
    UnitToken,
    UnknownToken,
    UnmatchedParenError,
    tokenize,
)


def float_token(value: float) -> FloatToken:
    return FloatToken(str(value), 0)


def float_mock(value: float) -> Float:
    return Float(value, 0, 0)


def unit_token(unit: str) -> UnitToken:
    return UnitToken(unit, 0)


def unit_mock(unit_registry: UnitRegistry, unit: str) -> Unit:
    mock_token = unit_token(unit)
    return Unit.try_new(mock_token, unit_registry).unwrap()


def unary_mock(op_type: OperatorType, expr: Expression) -> Unary:
    return Unary(op_type, expr, 0)


op_plus = OperatorToken(OperatorType.ADD, 0)
op_minus = OperatorToken(OperatorType.SUB, 0)
op_mul = OperatorToken(OperatorType.MUL, 0)
op_div = OperatorToken(OperatorType.DIV, 0)
op_exp = OperatorToken(OperatorType.EXP, 0)


def group_mock(paren_type: ParenType, expr: Expression) -> Group:
    return Group(expr, paren_type, 0, 0)


left_paren = ParenToken(ParenType.L_PAREN, 0)
right_paren = ParenToken(ParenType.R_PAREN, 0)
left_bracket = ParenToken(ParenType.L_BRACKET, 0)
right_bracket = ParenToken(ParenType.R_BRACKET, 0)
left_brace = ParenToken(ParenType.L_BRACE, 0)
right_brace = ParenToken(ParenType.R_BRACE, 0)


def test_preprocess_feet_inch() -> None:
    # 6'3'''
    input = "6' 3''"
    processed = Parser.preprocess_input(input)
    assert processed == "6ft 3in"


def test_preprocess_per_to_div() -> None:
    input = "6m per s per s"
    processed = Parser.preprocess_input(input)
    assert processed == "6m / s / s"


def test_preprocess_common_imperial_length_input() -> None:
    input = "6.3  foot   3.3"
    processed = Parser.preprocess_input(input)
    assert processed == "6.3  foot   3.3 inch"

    input = "6.3  foot   3.3 inch"
    processed = Parser.preprocess_input(input)
    assert processed == "6.3  foot   3.3 inch"


def test_char_stream() -> None:
    """Test whether the CharStream iteration works properly."""
    stream = CharStream(" 1.2345 big   string 3.13")
    string = "".join(stream)
    assert string == " 1.2345 big   string 3.13"


def test_float_token() -> None:
    float_token = FloatToken("3.393", 0)
    assert float_token.to_float() == 3.393


def test_float_token_consume() -> None:
    """Test whether the FloatToken.consume method works as intended."""
    token = next(CharStream("3.393").tokenize())
    assert isinstance(token, FloatToken) and token.token == "3.393"


def test_number_token_exponent() -> None:
    token = next(CharStream("2e-3 km").tokenize())
    assert isinstance(token, FloatToken) and token.token == "2e-3"


def test_number_token_decimal_exponent() -> None:
    token_stream = CharStream("2.3e-4.5").tokenize()
    token = next(token_stream)
    assert isinstance(token, FloatToken) and token.token == "2.3e-4"
    token = next(token_stream)
    assert isinstance(token, UnknownToken) and token.token == "."


def test_unit_token() -> None:
    unit_token = UnitToken("km", 0)
    assert unit_token.token == "km"


def test_unit_token_consume() -> None:
    token = next(CharStream("km").tokenize())
    assert isinstance(token, UnitToken) and token.token == "km"


def test_paren_token_consume() -> None:
    stream = CharStream("()")
    left, right = stream.tokenize()
    assert isinstance(left, ParenToken) and left.token == "("
    assert isinstance(right, ParenToken) and right.token == ")"


def test_operator_token_consume() -> None:
    """Primarily intended to check whether ** gets tokenized to OperatorType.MUL."""
    stream = CharStream("*-**/")
    mul, sub, exp, div = stream.tokenize()
    assert isinstance(mul, OperatorToken) and mul.op_type == OperatorType.MUL
    assert isinstance(sub, OperatorToken) and sub.op_type == OperatorType.SUB
    assert isinstance(exp, OperatorToken) and exp.op_type == OperatorType.EXP
    assert isinstance(div, OperatorToken) and div.op_type == OperatorType.DIV


def test_whitespace_consume() -> None:
    stream = CharStream("   bla   \n\n\r")
    token = next(stream.tokenize())
    assert isinstance(token, UnitToken) and token.token == "bla"


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


def test_unknown_token() -> None:
    stream = CharStream("123.4;%& #@@km")
    token_stream = stream.tokenize()
    next(token_stream)
    token = next(token_stream)
    assert isinstance(token, UnknownToken) and token.token == ";%&"

    token = next(token_stream)
    assert isinstance(token, UnknownToken) and token.token == "#@@"


def test_unary_parse(unit_registry: UnitRegistry) -> None:
    tokens: deque[Token] = deque(
        [
            op_minus,
            op_minus,
            op_plus,
            float_token(6.3),
        ]
    )
    parser = Parser(unit_registry)
    parsed = parser._parse_unary(tokens)
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
    assert parsed.ok() == mock_result
    assert not tokens


def test_unary_invalid_unary_error(unit_registry: UnitRegistry) -> None:
    tokens: deque[Token] = deque(
        [
            op_mul,
            float_token(6.68),
        ]
    )
    parser = Parser(unit_registry)
    result = parser._parse_unary(tokens)
    assert isinstance(result, Err)
    errors = result.err()
    assert isinstance(errors[0], InvalidUnaryError)


def test_binary_dimensionality_error(unit_registry: UnitRegistry) -> None:
    left = Float(1.0, 0, 0)
    right = Unit(unit_registry.Quantity("km"), "km", 0, 0)
    op = OperatorType.ADD
    result = Binary.try_new(left, op, right)
    assert isinstance(result, Err)
    assert isinstance(result.err(), DimensionalityError)


def test_binary_parse(unit_registry: UnitRegistry) -> None:
    tokens: deque[Token] = deque(
        [
            float_token(4.5),
            op_plus,
            float_token(3.6),
        ]
    )
    parser = Parser(unit_registry)
    parsed = parser._parse_expr(tokens)
    mock_result = Binary(
        float_mock(4.5),
        OperatorType.ADD,
        float_mock(3.6),
    )
    assert isinstance(parsed, Ok)
    assert parsed.ok() == mock_result
    assert not tokens


def test_parse_binary_division_by_zero(unit_registry: UnitRegistry) -> None:
    tokens: deque[Token] = deque(
        [
            unit_token("km"),
            op_div,
            float_token(0),
        ]
    )
    parser = Parser(unit_registry)
    result = parser._parse_expr(tokens)
    assert isinstance(result, Err)
    assert isinstance(result.err()[0], DivisionByZeroError)


def test_parse_binary_multiple_errors(unit_registry: UnitRegistry) -> None:
    """The expression "(1 / 0) + (2 / ) should report 2 errors"""
    tokens: deque[Token] = deque(
        [
            left_paren,
            float_token(1),
            op_div,
            float_token(0),
            right_paren,
            op_plus,
            left_paren,
            float_token(2),
            op_div,
            right_paren,
        ]
    )
    parser = Parser(unit_registry)
    parsed = parser._parse_expr(tokens)
    assert isinstance(parsed, Err)
    errors = parsed.err()
    assert len(errors) == 2
    assert isinstance(errors[0], DivisionByZeroError)
    assert isinstance(errors[1], ExpectedPrimaryError)


def test_primary_unknown_primary_error(unit_registry: UnitRegistry) -> None:
    tokens: deque[Token] = deque([op_mul])
    parser = Parser(unit_registry)
    result = parser._parse_primary(tokens)
    assert isinstance(result, Err)
    errors = result.err()
    assert isinstance(errors[0], UnexpectedTokenError)


def test_parse_group(unit_registry: UnitRegistry) -> None:
    tokens: deque[Token] = deque(
        [
            left_paren,
            left_brace,
            float_token(6.68),
            right_brace,
            right_paren,
        ]
    )
    parser = Parser(unit_registry)
    parsed = parser._parse_group(tokens, tokens.popleft())  # ty: ignore[invalid-argument-type]
    mock_result = float_mock(6.68)
    assert isinstance(parsed, Ok)
    assert parsed.ok() == mock_result
    assert not tokens


def test_parse_group_unmatched_closing_paren_error(unit_registry: UnitRegistry) -> None:
    tokens: deque[Token] = deque(tokenize(")(())"))
    parser = Parser(unit_registry)
    result = parser._parse_primary(tokens)
    assert isinstance(result, Err)
    errors = result.err()
    assert isinstance(errors[0], UnmatchedParenError)


def test_parse_group_unmatched_opening_paren_error(unit_registry: UnitRegistry) -> None:
    tokens: deque[Token] = deque(
        [
            left_paren,
            unit_token("m"),
        ]
    )
    parser = Parser(unit_registry)
    result = parser._parse_primary(tokens)
    assert isinstance(result, Err)
    errors = result.err()
    assert isinstance(errors[0], UnmatchedParenError)
    assert not tokens


def test_parse_float_standalone(unit_registry: UnitRegistry) -> None:
    tokens: deque[Token] = deque([float_token(3)])
    parser = Parser(unit_registry)
    parsed = parser._parse_primary_expression(Float, tokens)
    mock_result = float_mock(3)
    assert isinstance(parsed, Ok)
    assert parsed.ok() == mock_result


def test_parse_unit_standalone(unit_registry: UnitRegistry) -> None:
    tokens: deque[Token] = deque([unit_token("km")])
    parser = Parser(unit_registry)
    parsed = parser._parse_primary_expression(Unit, tokens)
    mock_result = unit_mock(unit_registry, "km")
    assert isinstance(parsed, Ok)
    assert parsed.ok() == mock_result


def test_parse_unit_standalone_leftover(unit_registry: UnitRegistry) -> None:
    """_parse_unit should not do implicit operations, so the 2nd token should be leftover"""
    tokens: deque[Token] = deque(
        [
            unit_token("N"),
            unit_token("m"),
        ]
    )
    parser = Parser(unit_registry)
    parsed = parser._parse_primary_expression(Unit, tokens)
    assert isinstance(parsed, Ok)
    mock_result = unit_mock(unit_registry, "N")
    assert parsed.ok() == mock_result
    assert tokens


def test_parse_unit_invalid_unit_simple(unit_registry: UnitRegistry) -> None:
    result = Unit.try_new(unit_token("dfdasf"), unit_registry)
    assert isinstance(result, Err)
    assert isinstance(result.err(), UndefinedUnitError)


def test_parse_unit_invalid_unit_complex(unit_registry: UnitRegistry) -> None:
    tokens: deque[Token] = deque(
        [
            unit_token("abc"),
            op_div,
            unit_token("def"),
        ]
    )
    parser = Parser(unit_registry)
    result = parser._parse_primary_expression(Unit, tokens)
    assert isinstance(result, Err)
    errors = result.err()
    assert isinstance(errors[0], UndefinedUnitError)
    assert isinstance(errors[1], UndefinedUnitError)


def test_parse_float_power_float(unit_registry: UnitRegistry) -> None:
    tokens: deque[Token] = deque(
        [
            float_token(4),
            op_exp,
            float_token(2),
        ]
    )
    parser = Parser(unit_registry)
    parsed = parser._parse_primary_expression(Float, tokens)
    assert isinstance(parsed, Ok)
    mock_result = Binary(
        float_mock(4),
        OperatorType.EXP,
        float_mock(2),
    )
    assert parsed.ok() == mock_result
    assert not tokens


def test_parse_unit_power_float(unit_registry: UnitRegistry) -> None:
    tokens: deque[Token] = deque(
        [
            unit_token("km"),
            op_exp,
            float_token(2),
        ]
    )
    parser = Parser(unit_registry)
    parsed = parser._parse_primary_expression(Unit, tokens)
    assert isinstance(parsed, Ok)
    mock_result = Binary(
        unit_mock(unit_registry, "km"),
        OperatorType.EXP,
        float_mock(2),
    )
    assert parsed.ok() == mock_result
    assert not tokens


def test_parse_unit_power_error(unit_registry: UnitRegistry) -> None:
    tokens: deque[Token] = deque(
        [
            unit_token("km"),
            op_exp,
            unit_token("km"),
        ]
    )
    parser = Parser(unit_registry)
    result = parser._parse_primary_expression(Unit, tokens)
    assert isinstance(result, Err)
    errors = result.err()
    assert isinstance(errors[0], ExpectedPrimaryError)
    assert not tokens


def test_parse_unit_power_groupexpr(unit_registry: UnitRegistry) -> None:
    tokens: deque[Token] = deque(
        [
            unit_token("km"),
            op_exp,
            left_paren,
            float_token(1),
            op_plus,
            float_token(1),
            right_paren,
        ]
    )
    parser = Parser(unit_registry)
    parsed = parser._parse_primary_expression(Unit, tokens)
    assert isinstance(parsed, Ok)
    # mock_result: km ** (1 + 1)
    mock_result = Binary(
        unit_mock(unit_registry, "km"),
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
    assert parsed.ok() == mock_result
    assert not tokens


def test_primary_chain_simple(unit_registry: UnitRegistry) -> None:
    # 30km / 2h
    tokens: deque[Token] = deque(
        [
            float_token(30),
            unit_token("km"),
            op_div,
            float_token(2),
            unit_token("h"),
        ]
    )
    parser = Parser(unit_registry)
    parsed = parser._parse_expr(tokens)
    mock_result = Binary(
        Binary(
            float_mock(30),
            OperatorType.MUL,
            unit_mock(unit_registry, "km"),
        ),
        OperatorType.DIV,
        Binary(
            float_mock(2),
            OperatorType.MUL,
            unit_mock(unit_registry, "h"),
        ),
    )
    assert isinstance(parsed, Ok)
    assert parsed.ok() == mock_result
    assert not tokens


def test_primary_chain_complex(unit_registry: UnitRegistry) -> None:
    # 1km (5+3)m / 2h 13min
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
    parser = Parser(unit_registry)
    parsed = parser._parse_expr(tokens)
    mock_result = Binary(
        Binary(
            Binary(
                Binary(
                    float_mock(1),
                    OperatorType.MUL,
                    unit_mock(unit_registry, "km"),
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
            unit_mock(unit_registry, "m"),
        ),
        OperatorType.DIV,
        Binary(
            Binary(
                float_mock(2),
                OperatorType.MUL,
                unit_mock(unit_registry, "h"),
            ),
            OperatorType.ADD,
            Binary(
                float_mock(13),
                OperatorType.MUL,
                unit_mock(unit_registry, "min"),
            ),
        ),
    )
    assert isinstance(parsed, Ok)
    assert parsed.ok() == mock_result
    assert not tokens


def test_primary_chain_order(unit_registry: UnitRegistry) -> None:
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
    parser = Parser(unit_registry)
    result = parser._parse_expr(tokens)
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
            unit_mock(unit_registry, "cm"),
            OperatorType.EXP,
            float_mock(2),
        ),
    )
    assert result.ok() == mock_result


def test_primary_chain_format_error(unit_registry: UnitRegistry) -> None:
    """The chain "6 3 ft m" is invalid because we're expecting a unit after the first '6', and a float after 'ft'."""
    tokens: deque[Token] = deque(
        [
            float_token(6),
            float_token(3),
            unit_token("ft"),
            unit_token("m"),
        ]
    )
    parser = Parser(unit_registry)
    result = parser._parse_primary(tokens)
    assert isinstance(result, Err)
    errors = result.err()
    error_0 = errors[0]
    assert isinstance(error_0, ExpectedPrimaryError) and "between numbers" in str(
        error_0
    )
    error_1 = errors[1]
    assert isinstance(error_1, ExpectedPrimaryError) and "number between units" in str(
        error_1
    )
    assert not tokens


def test_parse_strict_mode_implicit_multiplication(unit_registry: UnitRegistry) -> None:
    parser = Parser(unit_registry, mode=ParserMode.Strict)
    tokens: deque[Token] = deque(
        [
            float_token(6),
            unit_token("km"),
            unit_token("m"),
        ]
    )
    result = parser._parse_expr(tokens)
    assert isinstance(result, Ok)
    mock_result = Binary(
        Binary(float_mock(6), OperatorType.MUL, unit_mock(unit_registry, "km")),
        OperatorType.MUL,
        unit_mock(unit_registry, "m"),
    )
    assert result.ok() == mock_result


def test_parse_strict_mode_complex(unit_registry: UnitRegistry) -> None:
    parser = Parser(unit_registry, mode=ParserMode.Strict)
    tokens: deque[Token] = deque(
        [
            float_token(6),
            unit_token("km"),
            unit_token("m"),
            op_div,
            left_paren,
            unit_token("m"),
            op_plus,
            float_token(3),
            unit_token("cm"),
            right_paren,
            unit_token("h"),
        ]
    )
    result = parser._parse_expr(tokens)
    assert isinstance(result, Ok)
    mock_result = Binary(
        Binary(
            Binary(
                Binary(float_mock(6), OperatorType.MUL, unit_mock(unit_registry, "km")),
                OperatorType.MUL,
                unit_mock(unit_registry, "m"),
            ),
            OperatorType.DIV,
            group_mock(
                ParenType.L_PAREN,
                Binary(
                    unit_mock(unit_registry, "m"),
                    OperatorType.ADD,
                    Binary(
                        float_mock(3), OperatorType.MUL, unit_mock(unit_registry, "cm")
                    ),
                ),
            ),
        ),
        OperatorType.MUL,
        unit_mock(unit_registry, "h"),
    )
    assert result.ok() == mock_result


def test_constant_in_exponent(unit_registry: UnitRegistry):
    parser = Parser(unit_registry)
    tokens: deque[Token] = deque(
        [
            float_token(2),
            op_exp,
            unit_token("pi"),
        ]
    )
    result = parser._parse_primary_expression(Float, tokens)
    assert isinstance(result, Ok)
    mock_result = Binary(
        float_mock(2),
        OperatorType.EXP,
        unit_mock(unit_registry, "pi"),
    )
    assert result.ok() == mock_result


def test_parsing_currency_symbols(currency_unit_registry: UnitRegistry) -> None:
    parser = Parser(currency_unit_registry)
    input_string = "15$"
    result = parser.parse(input_string)
    assert isinstance(result, Ok)
    mock_result = Binary(
        float_mock(15),
        OperatorType.MUL,
        unit_mock(currency_unit_registry, "$"),
    )
    assert result.ok() == mock_result


def test_currency_symbols_not_defined(unit_registry: UnitRegistry) -> None:
    """Test whether the parser correctly sets the start and length of the currency symbol token when they're not defined as units."""
    parser = Parser(unit_registry)
    input_string = "15$"
    result = parser.parse(input_string)
    assert isinstance(result, Err)
    errors = result.err()
    assert len(errors) == 1
    assert (isinstance(errors[0], UndefinedUnitError)) and errors[0].span == (2, 3)
