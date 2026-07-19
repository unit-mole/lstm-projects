from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch

from src.data_preprocessing import load_huggingface_liar, load_local_liar
from src.model import FakeNewsLSTM, ModelConfig
from src.model_evaluation import (
    classification_metrics,
    make_prediction_frame,
    save_json,
    select_threshold,
    train_tfidf_baseline,
)
from src.model_training import (
    TrainingConfig,
    predict_probabilities,
    save_checkpoint,
    train_model,
)
from src.sequence_generation import create_data_loader, create_sequence_bundle
from src.text_preprocessing import TokenizerConfig, build_vocabulary
from src.visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_frequent_words,
    plot_length_distribution,
    plot_model_comparison,
    plot_roc_and_pr,
    plot_training_history,
)

ROOT = Path(__file__).resolve().parent


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the LIAR LSTM pipeline.")
    parser.add_argument("--source", choices=["huggingface", "local"], default="huggingface")
    parser.add_argument("--train-path", type=Path)
    parser.add_argument("--validation-path", type=Path)
    parser.add_argument("--test-path", type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data(args: argparse.Namespace):
    if args.source == "huggingface":
        return load_huggingface_liar()
    required = [args.train_path, args.validation_path, args.test_path]
    if any(path is None for path in required):
        raise ValueError("Local source requires --train-path, --validation-path, and --test-path.")
    return load_local_liar(args.train_path, args.validation_path, args.test_path)


def main() -> None:
    args = parse_arguments()
    models_dir = ROOT / "models"
    outputs_dir = ROOT / "outputs"
    models_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    splits = load_data(args)
    tokenizer_config = TokenizerConfig()
    vocabulary = build_vocabulary(splits.train["statement"], tokenizer_config)

    train_bundle = create_sequence_bundle(
        splits.train["statement"], splits.train["target"], vocabulary, tokenizer_config
    )
    validation_bundle = create_sequence_bundle(
        splits.validation["statement"], splits.validation["target"], vocabulary, tokenizer_config
    )
    test_bundle = create_sequence_bundle(
        splits.test["statement"], splits.test["target"], vocabulary, tokenizer_config
    )

    training_config = TrainingConfig(epochs=args.epochs, seed=args.seed)
    train_loader = create_data_loader(
        train_bundle, training_config.batch_size, shuffle=True, seed=args.seed
    )
    validation_loader = create_data_loader(validation_bundle, 512, shuffle=False)
    test_loader = create_data_loader(test_bundle, 512, shuffle=False)

    model_config = ModelConfig(vocabulary_size=len(vocabulary))
    model = FakeNewsLSTM(model_config)
    model, threshold, validation_metrics, history = train_model(
        model,
        train_loader,
        validation_loader,
        splits.train["target"],
        training_config,
    )

    y_test, lstm_probabilities = predict_probabilities(model, test_loader, torch.device("cpu"))
    lstm_metrics = classification_metrics(y_test, lstm_probabilities, threshold)

    baseline_vectorizer, baseline_model, baseline_validation_probabilities = train_tfidf_baseline(
        splits.train["statement"],
        splits.train["target"],
        splits.validation["statement"],
        random_state=args.seed,
    )
    baseline_threshold, _ = select_threshold(
        splits.validation["target"], baseline_validation_probabilities
    )
    baseline_test_probabilities = baseline_model.predict_proba(
        baseline_vectorizer.transform(splits.test["statement"])
    )[:, 1]
    baseline_metrics = classification_metrics(
        splits.test["target"], baseline_test_probabilities, baseline_threshold
    )

    predictions = make_prediction_frame(splits.test, lstm_probabilities, threshold)
    predictions.to_csv(outputs_dir / "sample_predictions.csv", index=False)
    predictions.loc[~predictions["correct"]].sort_values(
        "confidence", ascending=False
    ).to_csv(outputs_dir / "error_analysis.csv", index=False)

    result_payload = {
        "dataset": "ucsbnlp/liar",
        "task": "binary short-claim classification",
        "positive_class": "Fake",
        "validation_metrics": validation_metrics,
        "test_metrics": lstm_metrics,
        "baseline_test_metrics": baseline_metrics,
        "integrity_report": splits.integrity_report,
        "history": history,
    }
    save_json(result_payload, outputs_dir / "model_metrics.json")

    save_checkpoint(model, model_config, threshold, models_dir / "fake_news_lstm.pt")
    vocabulary.save(models_dir / "vocabulary.json")
    save_json(tokenizer_config.to_dict(), models_dir / "tokenizer_config.json")
    save_json(
        {
            "model_name": "LIAR Bidirectional LSTM",
            "created_on": date.today().isoformat(),
            "dataset": "ucsbnlp/liar",
            "task_scope": "Short political claims; not full-article fact checking",
            "label_mapping": {
                "pants-fire": 1,
                "false": 1,
                "barely-true": 1,
                "half-true": 0,
                "mostly-true": 0,
                "true": 0,
            },
            "class_names": {"0": "Real", "1": "Fake"},
            "prediction_threshold": threshold,
            "model_config": model_config.to_dict(),
            "tokenizer_config": tokenizer_config.to_dict(),
            "training_config": training_config.to_dict(),
            "validation_metrics": validation_metrics,
            "test_metrics": lstm_metrics,
            "baseline_test_metrics": baseline_metrics,
            "responsible_use": (
                "Educational portfolio model only. It detects language patterns and is not a "
                "fact-checking or evidence-verification system."
            ),
        },
        models_dir / "model_metadata.json",
    )

    plot_class_distribution(splits.train, outputs_dir / "class_distribution.png")
    plot_length_distribution(splits.train, outputs_dir / "statement_length_distribution.png")
    plot_frequent_words(splits.train, outputs_dir / "word_frequency_analysis.png")
    plot_training_history(history, outputs_dir / "training_curve.png")
    plot_confusion_matrix(lstm_metrics["confusion_matrix"], outputs_dir / "confusion_matrix.png")
    plot_roc_and_pr(
        y_test,
        lstm_probabilities,
        outputs_dir / "roc_curve.png",
        outputs_dir / "precision_recall_curve.png",
    )
    plot_model_comparison(
        {"TF-IDF Logistic Regression": baseline_metrics, "Bidirectional LSTM": lstm_metrics},
        outputs_dir / "baseline_comparison.png",
    )

    print(json.dumps(result_payload, indent=2))


if __name__ == "__main__":
    main()
