# pyright: reportPrivateUsage=false

from collections import deque

import pytest

from absolute_unit.parsing import (
    Binary,
    CharStream,
    Float,
    FloatToken,
    InvalidUnaryError,
    OperatorToken,
    OperatorType,
    ParenToken,
    ParenType,
    Token,
    Unary,
    Unit,
    UnitToken,
    UnexpectedPrimaryError,
    UnknownToken,
    UnmatchedParenError,
    Whitespace,
    _parse_primary,
    _parse_sum,
    _parse_unary,
    _parse_primary_chain,
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


def test_primary_parse() -> None:
    float_token: deque[Token] = deque([FloatToken("6.68", 0, 0)])
    parsed = _parse_primary(float_token)
    mock_result = Float(6.68)
    assert parsed == mock_result


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
    mock_result = Float(6.68)
    assert parsed == mock_result


def test_primary_raises_unmatched_closing_paren() -> None:
    tokens: deque[Token] = deque(tokenize(")(())"))
    with pytest.raises(UnmatchedParenError):
        _ = _parse_primary(tokens)


def test_primary_raises_unmatched_opening_paren() -> None:
    tokens: deque[Token] = deque([ParenToken("(", 0, 0), UnitToken("bla", 0, 0)])
    with pytest.raises(UnmatchedParenError):
        _ = _parse_primary(tokens)


def test_primary_raises_unknown_primary_error() -> None:
    tokens: deque[Token] = deque([OperatorToken("*", 0, 0)])
    with pytest.raises(UnexpectedPrimaryError):
        _ = _parse_primary(tokens)


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
            Float(3),
            OperatorType.MUL,
            Unit("m"),
        ),
        OperatorType.ADD,
        Binary(
            Float(14),
            OperatorType.MUL,
            Unit("cm"),
        ),
    )
    parsed = _parse_primary_chain(tokens)
    assert parsed == mock_result


def test_unary_parse() -> None:
    unary_tokens: deque[Token] = deque(
        [
            OperatorToken("-", 0, 0),
            OperatorToken("-", 0, 0),
            OperatorToken("+", 0, 0),
            FloatToken("6.3", 0, 0),
        ]
    )
    parsed = _parse_unary(unary_tokens)
    mock_result = Unary(
        OperatorType.SUB, Unary(OperatorType.SUB, Unary(OperatorType.ADD, Float(6.3)))
    )
    assert parsed == mock_result


def test_unary_raises_invalid_unary_error() -> None:
    tokens: deque[Token] = deque([OperatorToken("*", 0, 0), FloatToken("6.68", 0, 0)])
    with pytest.raises(InvalidUnaryError):
        _ = _parse_unary(tokens)


def test_binary_parse() -> None:
    tokens: deque[Token] = deque(
        [
            FloatToken("4.5", 0, 0),
            OperatorToken("+", 0, 0),
            OperatorToken("-", 0, 0),
            FloatToken("3.6", 0, 0),
        ]
    )
    parsed = _parse_sum(tokens)
    mock_result = Binary(
        Float(4.5), OperatorType.ADD, Unary(OperatorType.SUB, Float(3.6))
    )
    assert parsed == mock_result
