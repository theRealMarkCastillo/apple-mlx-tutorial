#!/usr/bin/env python3
"""
Dataset Download Script
=======================

Downloads and prepares datasets for NLP training.

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --imdb --snips
    python scripts/download_datasets.py --sentiment

The ``--samples`` flag delegates to ``scripts/generate_synthetic_data.py`` so
that there is exactly one canonical source of synthetic data in this repo.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

# Add repo root to sys.path so we can import the synthetic-data generator.
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' library not installed.")
    print("Please run: pip install datasets")
    sys.exit(1)


# Known HuggingFace mirrors/configs for SNIPS. ``bbalogh/snips_built_in_intents``
# is a stable public mirror of the original SNIPS voice-assistant corpus.
_SNIPS_CANDIDATES = [
    "bbalogh/snips_built_in_intents",
    "snips_built_in_intents",
]


class DatasetDownloader:
    """Download and prepare datasets."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Real datasets
    # ------------------------------------------------------------------

    def download_imdb(self, max_samples: int = 25000) -> Path | None:
        """Download IMDB movie reviews (train/test)."""
        print("\n" + "=" * 60)
        print("DOWNLOADING IMDB DATASET")
        print("=" * 60)

        dataset = load_dataset("imdb")
        output_dir = self.data_dir / "imdb"
        output_dir.mkdir(exist_ok=True)

        def _split(name: str) -> tuple[list[str], list[int]]:
            texts = []
            labels = []
            for i, ex in enumerate(dataset[name]):
                if i >= max_samples:
                    break
                texts.append(ex["text"])
                labels.append(ex["label"])
            return texts, labels

        train_texts, train_labels = _split("train")
        test_texts, test_labels = _split("test")

        with open(output_dir / "train.json", "w", encoding="utf-8") as f:
            json.dump({"texts": train_texts, "labels": train_labels}, f)
        with open(output_dir / "test.json", "w", encoding="utf-8") as f:
            json.dump({"texts": test_texts, "labels": test_labels}, f)

        print(f"✓ Downloaded {len(train_texts)} training examples")
        print(f"✓ Downloaded {len(test_texts)} test examples")
        print(f"✓ Saved to: {output_dir}")
        return output_dir

    def download_snips(self) -> Path | None:
        """Download SNIPS intents from a known HuggingFace mirror."""
        print("\n" + "=" * 60)
        print("DOWNLOADING SNIPS DATASET")
        print("=" * 60)

        dataset = None
        for name in _SNIPS_CANDIDATES:
            try:
                dataset = load_dataset(name)
                print(f"  Loaded from mirror: {name}")
                break
            except Exception as exc:  # noqa: BLE001 — surface every mirror failure
                print(f"  Mirror '{name}' failed: {exc}")

        if dataset is None:
            print("All SNIPS mirrors failed; falling back to the curated offline dataset.")
            return self._create_snips_fallback()

        output_dir = self.data_dir / "snips"
        output_dir.mkdir(exist_ok=True)

        # Some mirrors ship a single 'train' split; handle either case.
        train_texts = [ex["text"] for ex in dataset["train"]]
        train_labels = [ex["label"] for ex in dataset["train"]]

        if "test" in dataset:
            test_texts = [ex["text"] for ex in dataset["test"]]
            test_labels = [ex["label"] for ex in dataset["test"]]
        else:
            # Hold out the last 20% as a real test set with no overlap.
            split_idx = int(len(train_texts) * 0.8)
            test_texts = train_texts[split_idx:]
            test_labels = train_labels[split_idx:]
            train_texts = train_texts[:split_idx]
            train_labels = train_labels[:split_idx]

        with open(output_dir / "train.json", "w", encoding="utf-8") as f:
            json.dump({"texts": train_texts, "labels": train_labels}, f)
        with open(output_dir / "test.json", "w", encoding="utf-8") as f:
            json.dump({"texts": test_texts, "labels": test_labels}, f)

        print(f"✓ Downloaded {len(train_texts)} training examples")
        print(f"✓ Downloaded {len(test_texts)} test examples")
        print(f"✓ Saved to: {output_dir}")
        return output_dir

    def _create_snips_fallback(self) -> Path | None:
        """Offline SNIPS-like fallback used when every HF mirror fails."""
        output_dir = self.data_dir / "snips"
        output_dir.mkdir(exist_ok=True)

        # 6 intents (matches the canonical SNIPS NLU subset commonly used in
        # tutorials — PlayMusic, GetWeather, BookRestaurant, SearchCreativeWork,
        # AddToPlaylist, RateBook).
        intents_data = {
            "PlayMusic": [
                "play some music",
                "play my favorite song",
                "start playing music",
                "can you play a song",
                "play something from the Beatles",
            ],
            "GetWeather": [
                "what's the weather like",
                "how's the weather today",
                "will it rain tomorrow",
                "what's the forecast",
                "is it going to be sunny",
            ],
            "BookRestaurant": [
                "book a table for two",
                "make a reservation at a restaurant",
                "find me a place to eat",
                "reserve a table for dinner",
                "book a restaurant for tonight",
            ],
            "SearchCreativeWork": [
                "find me a good movie",
                "search for books by Stephen King",
                "show me romantic comedies",
                "find songs by Taylor Swift",
                "search for action movies",
            ],
            "AddToPlaylist": [
                "add this to my playlist",
                "save this song to my favorites",
                "add to my workout playlist",
                "put this in my playlist",
                "save to my music collection",
            ],
            "RateBook": [
                "rate this book 5 stars",
                "give this book a good review",
                "I rate this book highly",
                "this book deserves 4 stars",
                "rate this book positively",
            ],
        }

        train_texts: list[str] = []
        train_labels: list[str] = []
        for label, texts in intents_data.items():
            train_texts.extend(texts)
            train_labels.extend([label] * len(texts))

        # Use the LAST 20% of training rows as the held-out test set, and
        # remove them from training. This avoids the train/test overlap that
        # the previous modulo-based sampling produced.
        split_idx = int(len(train_texts) * 0.8)
        test_texts = train_texts[split_idx:]
        test_labels = train_labels[split_idx:]
        train_texts = train_texts[:split_idx]
        train_labels = train_labels[:split_idx]

        with open(output_dir / "train.json", "w", encoding="utf-8") as f:
            json.dump({"texts": train_texts, "labels": train_labels}, f)
        with open(output_dir / "test.json", "w", encoding="utf-8") as f:
            json.dump({"texts": test_texts, "labels": test_labels}, f)

        print(f"✓ Created {len(train_texts)} training examples")
        print(f"✓ Created {len(test_texts)} test examples")
        print(f"✓ Saved to: {output_dir}")
        return output_dir

    def download_banking77(self) -> Path | None:
        """Download Banking77 intents."""
        print("\n" + "=" * 60)
        print("DOWNLOADING BANKING77 DATASET")
        print("=" * 60)

        try:
            dataset = load_dataset("banking77")
        except Exception as exc:
            print(f"ERROR: Could not download Banking77 dataset: {exc}")
            traceback.print_exc()
            return None

        output_dir = self.data_dir / "banking77"
        output_dir.mkdir(exist_ok=True)

        train_texts = [ex["text"] for ex in dataset["train"]]
        train_labels = [ex["label"] for ex in dataset["train"]]
        test_texts = [ex["text"] for ex in dataset["test"]]
        test_labels = [ex["label"] for ex in dataset["test"]]

        with open(output_dir / "train.json", "w", encoding="utf-8") as f:
            json.dump({"texts": train_texts, "labels": train_labels}, f)
        with open(output_dir / "test.json", "w", encoding="utf-8") as f:
            json.dump({"texts": test_texts, "labels": test_labels}, f)

        print(f"✓ Downloaded {len(train_texts)} training examples")
        print(f"✓ Downloaded {len(test_texts)} test examples")
        print(f"✓ Saved to: {output_dir}")
        return output_dir

    def download_wikitext(self, version: str = "wikitext-2-v1") -> Path | None:
        """Download WikiText for text generation."""
        print("\n" + "=" * 60)
        print(f"DOWNLOADING WIKITEXT DATASET ({version})")
        print("=" * 60)

        try:
            dataset = load_dataset("wikitext", version)
        except Exception as exc:
            print(f"ERROR: Could not download WikiText {version}: {exc}")
            traceback.print_exc()
            return None

        output_dir = self.data_dir / "wikitext"
        output_dir.mkdir(exist_ok=True)

        for split, filename in [("train", "train.txt"),
                                ("validation", "validation.txt"),
                                ("test", "test.txt")]:
            with open(output_dir / filename, "w", encoding="utf-8") as f:
                for ex in dataset[split]:
                    text = ex["text"].strip()
                    if text:
                        f.write(text + "\n")

        print(f"✓ Downloaded WikiText {version}")
        print(f"✓ Saved to: {output_dir}")
        return output_dir

    # ------------------------------------------------------------------
    # Synthetic samples (delegated)
    # ------------------------------------------------------------------

    def create_sample_datasets(self) -> None:
        """
        Generate the synthetic sample datasets used by the intro notebooks.

        Delegates to ``generate_synthetic_data`` so there is one canonical
        source of sample data in this repo. The synthetic generator writes to
        the same paths (``data/intent_samples``, ``data/sentiment_samples``,
        ``data/text_gen_samples``, ``data/rag_samples``) and also produces the
        LoRA chat-format files.
        """
        import generate_synthetic_data  # local module, same directory as this script

        print("\n" + "=" * 60)
        print("CREATING SAMPLE DATASETS")
        print("=" * 60)

        generate_synthetic_data.generate_intent_data()
        generate_synthetic_data.generate_sentiment_data()
        generate_synthetic_data.generate_text_corpus()
        generate_synthetic_data.generate_rag_knowledge_base()
        generate_synthetic_data.generate_lora_chat_data()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NLP datasets")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--sentiment", action="store_true", help="Download sentiment datasets")
    parser.add_argument("--intent", action="store_true", help="Download intent datasets")
    parser.add_argument("--generation", action="store_true", help="Download text generation datasets")

    parser.add_argument("--imdb", action="store_true", help="Download IMDB reviews")
    parser.add_argument("--snips", action="store_true", help="Download SNIPS intents")
    parser.add_argument("--banking77", action="store_true", help="Download Banking77 intents")
    parser.add_argument("--wikitext", action="store_true", help="Download WikiText")
    parser.add_argument("--samples", action="store_true", help="Create small sample datasets")

    parser.add_argument("--data-dir", default="data", help="Directory to save datasets (default: data)")
    parser.add_argument("--max-samples", type=int, default=25000,
                        help="Maximum samples for large datasets (default: 25000 — IMDB's full size)")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    downloader = DatasetDownloader(args.data_dir)

    print("=" * 60)
    print("DATASET DOWNLOADER")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Max samples: {args.max_samples}")

    if args.all or args.samples:
        downloader.create_sample_datasets()
    if args.all or args.sentiment or args.imdb:
        downloader.download_imdb(max_samples=args.max_samples)
    if args.all or args.intent or args.snips:
        downloader.download_snips()
    if args.all or args.intent or args.banking77:
        downloader.download_banking77()
    if args.all or args.generation or args.wikitext:
        downloader.download_wikitext()

    print("\n" + "=" * 60)
    print("✓ DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"\nDatasets saved to: {args.data_dir}/")
    print("\nAvailable datasets:")

    data_path = Path(args.data_dir)
    if data_path.exists():
        for subdir in sorted(data_path.iterdir()):
            if subdir.is_dir():
                files = list(subdir.glob("*.json")) + list(subdir.glob("*.txt"))
                print(f"  • {subdir.name}/ ({len(files)} files)")
        for f in sorted(data_path.glob("*.jsonl")):
            print(f"  • {f.name}")

    print("\nNext steps:")
    print("  1. Start notebooks: cd notebooks && jupyter notebook")
    print("  2. Open 00_Overview.ipynb to get started")
    print("  3. See notebooks/README.md for learning paths")


if __name__ == "__main__":
    main()