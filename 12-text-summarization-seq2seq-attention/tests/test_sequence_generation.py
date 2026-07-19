import numpy as np

from src.sequence_generation import pad_sequences_post, prepare_teacher_forcing_targets


def test_post_padding_and_truncation():
    result = pad_sequences_post([[1, 2], [3, 4, 5, 6]], max_length=3)
    np.testing.assert_array_equal(result, np.array([[1, 2, 0], [3, 4, 5]]))


def test_teacher_forcing_shift():
    values = np.array([[2, 10, 11, 6, 0]])
    decoder_input, decoder_target = prepare_teacher_forcing_targets(values)
    np.testing.assert_array_equal(decoder_input, np.array([[2, 10, 11, 6]]))
    np.testing.assert_array_equal(decoder_target, np.array([[10, 11, 6, 0]]))
