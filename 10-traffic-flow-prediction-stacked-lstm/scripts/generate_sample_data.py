"""Regenerate the safe sample CSV."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.synthetic_data import generate_traffic_data  # noqa: E402


def main() -> None:
    frame = generate_traffic_data().tail(24 * 90)
    output = PROJECT_ROOT / "data" / "sample_traffic_flow_data.csv"
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame):,} rows to {output}")


if __name__ == "__main__":
    main()
