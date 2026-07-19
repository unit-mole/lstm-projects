from src.config import SOURCE_TOKENIZER_PATH, TARGET_TOKENIZER_PATH
from src.tokenizer_utils import PortableTokenizer, fit_portable_tokenizer


def test_saved_tokenizer_dimensions_and_boundary_tokens():
    source = PortableTokenizer.load(SOURCE_TOKENIZER_PATH)
    target = PortableTokenizer.load(TARGET_TOKENIZER_PATH)
    assert source.vocab_size == 88
    assert target.vocab_size == 57
    assert target.word_index["sostok"] == 2
    assert target.word_index["eostok"] == 6


def test_oov_token_is_used_for_unseen_words():
    tokenizer = fit_portable_tokenizer(["alpha beta", "alpha gamma"])
    sequence = tokenizer.text_to_sequence("alpha unseen")
    assert sequence[0] == tokenizer.word_index["alpha"]
    assert sequence[1] == tokenizer.oov_id
