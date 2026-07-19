# Methodology Improvements

## 1. Prevent exact dialogue-pair leakage

The supplied notebook repeats 20 fixed pairs 3,500 times and performs a random row split.
Every exact pair appears in all three splits. The cleaned retraining pipeline splits by unique
input-response pair so an exact pair cannot cross training, validation, and test data.

## 2. Fit tokenizers on training data only

The cleaned pipeline fits source and target tokenizers on training pairs only and stores explicit
mappings, sequence lengths, start/end IDs, and vocabulary sizes.

## 3. Separate training from inference

The portfolio project includes native Keras artifacts, a reusable NumPy inference engine, and a
`ChatbotService.respond()` interface used by Streamlit and automated tests.

## 4. Handle out-of-domain inputs safely

The application calculates an out-of-vocabulary ratio and decoder token confidence. Strongly
out-of-domain or empty outputs use a documented fallback instead of presenting arbitrary text as reliable.

## 5. Evaluate beyond token accuracy

The project reports training and validation curves, BLEU-like score, exact match, response length,
generated examples, attention, retrieval and most-frequent baselines, and qualitative limitations.

## 6. Communicate scope honestly

Perfect supplied metrics are retained for reproducibility but explicitly qualified. This is a small
synthetic-template chatbot, not an open-domain assistant, production support agent, Transformer, or LLM.
