from parsing import CharStream

if __name__ == "__main__":
    stream = CharStream("big looong-ass sttring")
    while stream:
        print(stream.peek())
        stream.advance()
