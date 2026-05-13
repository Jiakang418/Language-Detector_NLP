import re
import unicodedata


def normalize_unicode(text):
    return unicodedata.normalize("NFKC", text)


def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub("", text)


def remove_numbers(text):
    return re.sub(r"\d+", "", text)


def remove_extra_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


def detect_script(text):
    counts = {
        "chinese": 0,
        "japanese": 0,
        "korean": 0,
        "latin": 0,
        "other": 0
    }

    for char in text:
        code = ord(char)

        if 0x3040 <= code <= 0x30FF:
            counts["japanese"] += 1
        elif (
                0xAC00 <= code <= 0xD7AF or
                0x1100 <= code <= 0x11FF or
                0x3130 <= code <= 0x318F
        ):
            counts["korean"] += 1
        elif 0x4E00 <= code <= 0x9FFF:
            counts["chinese"] += 1
        elif char.isalpha():
            counts["latin"] += 1
        elif not char.isspace():
            counts["other"] += 1

    if counts["japanese"] > 0:
        return "japanese"

    if counts["korean"] > 0:
        return "korean"

    if counts["chinese"] > 0:
        return "chinese"

    if counts["latin"] > 0:
        return "latin"

    return "empty_or_symbol"

def segment_cjk(text):
    return " ".join([char for char in text if not char.isspace()])

def clean_latin_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)
    text = remove_extra_spaces(text)
    return text


def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = normalize_unicode(text)
    text = remove_emojis(text)
    text = remove_numbers(text)
    text = remove_extra_spaces(text)

    if text == "":
        return ""

    script = detect_script(text)

    if script in ["chinese", "japanese", "korean"]:
        return segment_cjk(text)

    if script == "latin":
        return clean_latin_text(text)

    return remove_extra_spaces(text)


if __name__ == "__main__":
    test_samples = [
        "Hello!!! I am learning NLP 123 😊",
        "Saya suka makan nasi lemak!!!",
        "Aku suka makan bakso 2024",
        "我喜欢学习人工智能。",
        "私は学生です。",
        "한국어를 공부합니다.",
        "😂😂😂",
        "Hello 我喜欢 NLP"
    ]

    for sample in test_samples:
        print("Original:    ", sample)
        print("Script:      ", detect_script(sample))
        print("Preprocessed:", preprocess_text(sample))
        print("-" * 50)