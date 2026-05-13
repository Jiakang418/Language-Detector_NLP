def detect_script(text):
    for ch in text:
        code = ord(ch)

        if 0x4E00 <= code <= 0x9FFF:
            return "Chinese/CJK"
        elif 0x3040 <= code <= 0x309F:
            return "Japanese Hiragana"
        elif 0x30A0 <= code <= 0x30FF:
            return "Japanese Katakana"
        elif 0xAC00 <= code <= 0xD7AF:
            return "Korean Hangul"

    return "Latin/Other"


def cjk_segment(text):
    return " ".join(list(text))


def preprocess_text(text):
    script = detect_script(text)

    if script != "Latin/Other":
        return cjk_segment(text)

    return text.lower().strip()


examples = [
    "Hello world",
    "Saya suka makan nasi lemak",
    "Saya suka makan bakso",
    "我喜欢学习人工智能",
    "私は学生です",
    "한국어를 공부합니다"
]

for text in examples:
    print("Original:", text)
    print("Detected:", detect_script(text))
    print("Processed:", preprocess_text(text))
    print()