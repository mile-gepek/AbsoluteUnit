from .parsing import parse

if __name__ == "__main__":
    text = "3.6 km 14m + [(57 / 27)] ** 3"
    parsed = parse(text)
    print(parsed)
