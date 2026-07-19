from src.text_preprocessing import (
    OOV_TOKEN,
    PAD_TOKEN,
    TokenizerConfig,
    build_vocabulary,
    encode_text,
    tokenize_text,
)


def test_style_tokens_are_preserved():
    tokens = tokenize_text("SHOCKING NEWS!!! Visit https://example.com for 100% proof?")
    assert "<EXCLAMATION>" in tokens
    assert "<URL>" in tokens
    assert "<NUMBER>" in tokens
    assert "<QUESTION>" in tokens
    assert "<ALL_CAPS_STYLE>" in tokens


def test_vocabulary_and_padding():
    config = TokenizerConfig(maximum_vocabulary_size=20, maximum_sequence_length=8)
    vocabulary = build_vocabulary(["one two three", "one two four"], config)
    assert vocabulary.index_to_token[0] == PAD_TOKEN
    assert vocabulary.index_to_token[1] == OOV_TOKEN
    encoded, diagnostics = encode_text("one unseen", vocabulary, config)
    assert len(encoded) == 8
    assert diagnostics["oov_count"] == 1
