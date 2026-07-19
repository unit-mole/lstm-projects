from pathlib import Path

from src.fake_news_pipeline import FakeNewsPipeline


def test_saved_pipeline_produces_probability():
    model_dir = Path(__file__).resolve().parents[1] / "models"
    pipeline = FakeNewsPipeline.load(model_dir)
    result = pipeline.predict("The agency published the report after a public meeting.")
    assert result.predicted_label in {"Real", "Fake"}
    assert 0.0 <= result.fake_probability <= 1.0
    assert 0.0 <= result.confidence <= 1.0
