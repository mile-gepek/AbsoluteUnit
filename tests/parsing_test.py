from absolute_unit import CharStream
from absolute_unit.parsing import (
    FloatToken,
    OperatorToken,
    OperatorType,
    ParenToken,
    ParenType,
    UnitToken,
    UnknownToken,
    WhitespaceToken,
    tokenize,
)


def test_char_stream():
    stream = CharStream(" 1.2345 big   string 3.13")
    string = "".join(stream)
    assert string == " 1.2345 big   string 3.13"


def test_float_token():
    stream = CharStream("3.393.")
    float_token = FloatToken(stream)
    assert float_token.to_float() == 3.393


def test_unit_token():
    stream = CharStream("feet3.4")
    unit_token = UnitToken(stream)
    assert unit_token.token == "feet"


def test_paren_token():
    stream = CharStream("()")
    paren_token = ParenToken(stream)
    assert paren_token.paren_type == ParenType.L_PAREN
    paren_token = ParenToken(stream)
    assert paren_token.paren_type == ParenType.R_PAREN


def test_operator_token():
    stream = CharStream("+*-**/")
    op_token = OperatorToken(stream)
    assert op_token.op_type == OperatorType.ADD
    op_token = OperatorToken(stream)
    assert op_token.op_type == OperatorType.MUL
    op_token = OperatorToken(stream)
    assert op_token.op_type == OperatorType.SUB
    op_token = OperatorToken(stream)
    assert op_token.op_type == OperatorType.EXP
    op_token = OperatorToken(stream)
    assert op_token.op_type == OperatorType.DIV


def test_whitespace_token():
    stream = CharStream("  \t\n\n\r")
    whitespace_token = WhitespaceToken(stream)
    assert whitespace_token.token is None


def test_unknown_token():
    stream = CharStream("@#$;<><:")
    unknown_token = UnknownToken(stream)
    assert unknown_token.token == "@#$;<><:"


def test_batch():
    token_stream = tokenize("6 ft in /   (4.3s * 13J)")
    token_strings = [t.token for t in token_stream]
    print(token_strings)
    assert token_strings == [
        "6",
        "ft",
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
