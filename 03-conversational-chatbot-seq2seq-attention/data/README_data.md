# Conversation Data

The included `sample_conversations.csv` contains 20 unique, synthetic, privacy-safe
input-response pairs reconstructed from the supplied notebook.

## Schema

| Column | Description |
|---|---|
| `input_text` | User message supplied to the encoder |
| `target_text` | Reference chatbot response |

The supplied notebook generated 3,500 rows by repeatedly sampling these 20 pairs.
No private messages, customer conversations, or confidential text are included.

## Important evaluation note

The original notebook performed a random row split after repeating the same 20 pairs.
Consequently, every unique pair appeared in training, validation, and test data. The
reported perfect BLEU-like and exact-match values should be interpreted as template
memorization, not unseen-dialogue generalization.

## Using another dataset

A compatible CSV should contain one input column and one response column. Supported
aliases include:

- input: `input_text`, `input`, `question`, `prompt`, `message`, `user_message`
- response: `target_text`, `response`, `answer`, `reply`, `target`, `bot_response`

Use only conversation data that you are permitted to process and redistribute. Never
commit private, sensitive, confidential, or copyrighted dialogue data without permission.
