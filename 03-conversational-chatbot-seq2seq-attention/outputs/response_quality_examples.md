# Response Quality Examples

## Canonical in-domain examples

| User message | Generated response |
|---|---|
| `hello` | `hi there` |
| `how are you` | `i am doing well` |
| `what should i do next` | `you should review the latest update` |
| `can you summarize this` | `yes i can provide a short summary` |
| `goodbye` | `goodbye have a nice day` |

## Out-of-domain behavior

The model was trained on only 20 fixed synthetic templates. Messages dominated by unseen words
trigger a responsible fallback rather than presenting a memorized or nonsensical response as reliable.

## Evaluation warning

The original 3,500-row dataset repeatedly sampled the same 20 pairs, and the random row split put
all 20 exact pairs into training, validation, and test sets. Perfect BLEU-like and exact-match scores
therefore demonstrate memorization of the repeated templates, not broad conversational generalization.
