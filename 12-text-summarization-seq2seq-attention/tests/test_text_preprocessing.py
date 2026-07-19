from src.text_preprocessing import clean_text, validate_input_text


def test_clean_text_matches_training_policy():
    assert clean_text("<b>Hello, WORLD!</b>  2026") == "hello world 2026"


def test_short_input_is_rejected():
    result = validate_input_text("too short", min_words=8)
    assert not result.is_valid
    assert "too short" in result.message.lower()


def test_valid_input_is_accepted():
    result = validate_input_text(
        "The company announced a policy after a detailed internal review process."
    )
    assert result.is_valid
    assert result.word_count >= 8
