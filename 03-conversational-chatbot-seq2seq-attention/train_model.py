from __future__ import annotations
import argparse
from pathlib import Path
from src.data_preprocessing import load_conversation_file
from src.model_training import train_chatbot

def parse_args():
    parser = argparse.ArgumentParser(description="Retrain the Seq2Seq attention chatbot.")
    parser.add_argument("--data", type=Path, default=Path("data/sample_conversations.csv"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    frame = load_conversation_file(args.data)
    print(train_chatbot(frame, Path.cwd(), args.epochs, args.batch_size, args.seed))

if __name__ == "__main__":
    main()
