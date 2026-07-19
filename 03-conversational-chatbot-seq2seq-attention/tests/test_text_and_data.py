from __future__ import annotations
import pandas as pd
import pytest
from src.data_preprocessing import prepare_conversation_pairs
from src.text_preprocessing import clean_text, validate_user_text

def test_clean_text_matches_supplied_pipeline():
    assert clean_text("Hello, HOW are you?!") == "hello how are you"

def test_prepare_conversation_pairs_detects_aliases_and_duplicates():
    frame = pd.DataFrame({"question":["Hello","Hello",None],"answer":["Hi there","Hi there","Missing"]})
    prepared = prepare_conversation_pairs(frame)
    assert len(prepared) == 1
    assert prepared.loc[0,"input_clean"] == "hello"
    assert prepared.loc[0,"target_clean"] == "hi there"

def test_empty_user_input_is_rejected():
    with pytest.raises(ValueError):
        validate_user_text("   ")
