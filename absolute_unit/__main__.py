from result import Err
from rich.pretty import pprint
from .parsing import parse, _EOL

if __name__ == "__main__":
    inputs = [
        "3.6 * km + 14 * m + [(57 * inch / 27)] ** 1",
        "4.5 + -3.6",
    ]
    while inp := input().rstrip():
        stripped = inp.strip()
        leading_whitespace = len(inp) - len(stripped)
        parsed = parse(stripped)
        if isinstance(parsed, Err):
            errors = parsed.err_value
            for exc in errors:
                print(" " * leading_whitespace, end="")
                span = exc.span
                if not isinstance(span, _EOL):
                    length = span[1] - span[0]
                    print(" " * span[0], end="")
                    print("^" * length, end="")
                    print(" " * (len(stripped) - span[1] + 4), end="")
                    print(exc)
                else:
                    print(" " * len(stripped), end="")
                    print(" ^", end="")
                    print(" ", exc)
            continue
        value = parsed.ok_value
        pprint(value, expand_all=True)
        print(inp)
        print("=")
        print(value)
        print("=")
        evaluated_str = str(value.evaluate())
        print(evaluated_str)
        print("-" * len(evaluated_str))
