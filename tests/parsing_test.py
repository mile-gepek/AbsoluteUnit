import pytest
from collections import deque
from pint import Quantity
from pint.facets.plain import PlainQuantity
from absolute_unit.parsing import (
    CharStream,
    Token,
    FloatToken,
    OperatorToken,
    OperatorType,
    ParenToken,
    ParenType,
    UnitToken,
    UnknownToken,
    UnmatchedParenError,
    Whitespace,
    parse_primary,
    parse_unary,
    tokenize,
)


def test_char_stream():
    stream = CharStream(" 1.2345 big   string 3.13")
    string = "".join(stream)
    assert string == " 1.2345 big   string 3.13"


def test_float_token():
    float_token = FloatToken("3.393", 0, 0)
    assert float_token.to_float() == 3.393


def test_float_token_consume():
    token = Token.from_stream(CharStream("3.393"))
    assert isinstance(token, FloatToken) and token.token == "3.393"


def test_unit_token():
    unit_token = UnitToken("feet", 0, 0)
    assert unit_token.token == "feet"


def test_unit_token_consume():
    token = Token.from_stream(CharStream("km"))
    assert isinstance(token, UnitToken) and token.token == "km"


def test_paren_token():
    paren_token = ParenToken("(", 0, 0)
    assert paren_token.paren_type == ParenType.L_PAREN
    paren_token = ParenToken(")", 0, 0)
    assert paren_token.paren_type == ParenType.R_PAREN


def test_paren_token_consume():
    stream = CharStream("()")
    token = Token.from_stream(stream)
    assert isinstance(token, ParenToken) and token.token == "("
    token = Token.from_stream(stream)
    assert isinstance(token, ParenToken) and token.token == ")"


def test_operator_token():
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


def test_operator_token_consume():
    stream = CharStream("*-**/")
    token = Token.from_stream(stream)
    assert isinstance(token, OperatorToken) and token.op_type == OperatorType.MUL
    token = Token.from_stream(stream)
    assert isinstance(token, OperatorToken) and token.op_type == OperatorType.SUB
    token = Token.from_stream(stream)
    assert isinstance(token, OperatorToken) and token.op_type == OperatorType.EXP
    token = Token.from_stream(stream)
    assert isinstance(token, OperatorToken) and token.op_type == OperatorType.DIV


def test_whitespace_token():
    whitespace = Whitespace("", 0, 0)
    assert whitespace.token == ""


def test_whitespace_consume():
    stream = CharStream("   bla   \n\n\r")
    token = Token.from_stream(stream)
    assert isinstance(token, Whitespace) and token.token == ""
    token = Token.from_stream(stream)
    token = Token.from_stream(stream)
    assert isinstance(token, Whitespace) and token.token == ""


def test_unknown_token():
    unknown_token = UnknownToken("@#$;<><:", 0, 0)
    assert unknown_token.token == "@#$;<><:"


def test_tokenize():
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


def test_token_span():
    token_stream = tokenize("6 kilometer / 3 hour")
    token = next(token_stream)
    assert token is not None and token.span == (0, 1)
    token = next(token_stream)
    assert token is not None and token.span == (2, 11)
    token = next(token_stream)
    assert token is not None and token.span == (12, 13)


def test_primary_parse():
    float_token: deque[Token] = deque([FloatToken("6.68", 0, 0)])
    assert parse_primary(float_token) == Quantity(6.68)
    unit_token: deque[Token] = deque([UnitToken("km", 0, 0)])
    assert parse_primary(unit_token) == Quantity("km")


def test_primary_unmatched_closing_paren():
    tokens = deque(tokenize(")(())"))
    with pytest.raises(UnmatchedParenError) as excinfo:
        _ = parse_primary(tokens)
        assert "unmatched closing parenthesis" == str(excinfo.value)


def test_primary_unmatched_opening_paren():
    tokens = deque(tokenize("[abc * 2"))
    with pytest.raises(UnmatchedParenError) as excinfo:
        _ = parse_primary(tokens)
        assert "unmatched opening bracket" == str(excinfo.value)


def test_unary_parse():
    unary_tokens: deque[Token] = deque(
        [
            OperatorToken("-", 0, 0),
            OperatorToken("-", 0, 0),
            OperatorToken("+", 0, 0),
            FloatToken("6.3", 0, 0),
        ]
    )
    assert parse_unary(unary_tokens) == Quantity(6.3)
