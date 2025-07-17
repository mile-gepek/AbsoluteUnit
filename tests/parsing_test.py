import pytest
from collections import deque
from absolute_unit import parsing
from absolute_unit.parsing import (
    Binary,
    CharStream,
    Float,
    Token,
    FloatToken,
    OperatorToken,
    OperatorType,
    ParenToken,
    ParenType,
    Unary,
    Unit,
    UnitToken,
    UnknownToken,
    UnmatchedParenError,
    Whitespace,
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
    assert token is not None and token.span == (0, 1)
    token = next(token_stream)
    assert token is not None and token.span == (2, 11)
    token = next(token_stream)
    assert token is not None and token.span == (12, 13)


def test_primary_parse() -> None:
    float_token: deque[Token] = deque([FloatToken("6.68", 0, 0)])
    parsed = parsing._parse_primary(float_token)
    assert isinstance(parsed, Float) and parsed.value == 6.68
    unit_token: deque[Token] = deque([UnitToken("km", 0, 0)])
    parsed = parsing._parse_primary(unit_token)
    assert isinstance(parsed, Unit) and parsed.unit_str() == "km"


def test_primary_unmatched_closing_paren() -> None:
    tokens: deque[Token] = deque(tokenize(")(())"))
    with pytest.raises(UnmatchedParenError) as excinfo:
        _ = parsing._parse_primary(tokens)
        assert "unmatched closing parenthesis" == str(excinfo.value)


def test_primary_unmatched_opening_paren() -> None:
    tokens: deque[Token] = deque(tokenize("[abc * 2"))
    with pytest.raises(UnmatchedParenError) as excinfo:
        _ = parsing._parse_primary(tokens)
        assert "unmatched opening bracket" == str(excinfo.value)


def test_unary_parse() -> None:
    unary_tokens: deque[Token] = deque(
        [
            OperatorToken("-", 0, 0),
            OperatorToken("-", 0, 0),
            OperatorToken("+", 0, 0),
            FloatToken("6.3", 0, 0),
        ]
    )
    parsed = parsing._parse_unary(unary_tokens)
    assert isinstance(parsed, Unary) and parsed.op == OperatorType.SUB
    value = parsed.value
    assert isinstance(value, Unary) and value.op == OperatorType.SUB
    value = value.value
    assert isinstance(value, Unary) and value.op == OperatorType.ADD
    value = parsed.value


def test_binary_parse() -> None:
    tokens: deque[Token] = deque(
        [
            FloatToken("4.5", 0, 0),
            OperatorToken("+", 0, 0),
            OperatorToken("-", 0, 0),
            FloatToken("3.6", 0, 0),
        ]
    )
    parsed = parsing._parse_sum(tokens)
    assert isinstance(parsed, Binary)
    left = parsed.left
    assert isinstance(left, Float) and left.value == 4.5
    assert parsed.op is not None and parsed.op == OperatorType.ADD
    right = parsed.right
    assert isinstance(right, Unary)
    assert right.op is not None and right.op == OperatorType.SUB
    value = right.value
    assert isinstance(value, Float) and value.value == 3.6
