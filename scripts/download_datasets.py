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
import random
from pathlib import Path

# Add repo root to sys.path so we can import the synthetic-data generator.
sys.path.insert(0, str(Path(__file__).resolve().parent))

def load_dataset(*args, **kwargs):
    """Import the optional network loader only for real dataset downloads."""
    from datasets import load_dataset as hf_load_dataset
    return hf_load_dataset(*args, **kwargs)


def stratified_holdout(texts, labels, seed=0):
    """Hold out examples within each label, never whole class-sorted tails."""
    rng = random.Random(seed)
    groups = {}
    for text, label in dict.fromkeys(zip(texts, labels)):
        groups.setdefault(label, []).append(text)
    train_texts, train_labels, test_texts, test_labels = [], [], [], []
    for label, examples in groups.items():
        rng.shuffle(examples)
        n_test = min(len(examples) - 1, max(1, round(len(examples) * 0.2)))
        test_texts.extend(examples[:n_test])
        test_labels.extend([label] * n_test)
        train_texts.extend(examples[n_test:])
        train_labels.extend([label] * (len(examples) - n_test))
    return train_texts, train_labels, test_texts, test_labels


class DatasetDownloader:
    """Download and prepare datasets."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Real datasets
    # ------------------------------------------------------------------

    def download_imdb(self, max_samples: int = 25000) -> Path | None:
        """Download IMDB movie reviews (train/test)."""
        print("\n" + "=" * 60)
        print("DOWNLOADING IMDB DATASET")
        print("=" * 60)

        dataset = load_dataset("stanfordnlp/imdb")
        output_dir = self.data_dir / "imdb"
        output_dir.mkdir(exist_ok=True)

        def _split(name: str) -> tuple[list[str], list[int]]:
            texts = []
            labels = []
            for i, ex in enumerate(dataset[name].shuffle(seed=0)):
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
        """Download the original seven-intent SNIPS train/validation benchmark."""
        print("\n" + "=" * 60)
        print("DOWNLOADING SNIPS DATASET")
        print("=" * 60)

        # The original SNIPS benchmark is JSON, so no removed HF dataset
        # script API or unverified mirror is needed.
        from urllib.request import urlopen
        intents = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic",
                   "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]
        base = "https://raw.githubusercontent.com/snipsco/nlu-benchmark/master/2017-06-custom-intent-engines"
        train_texts, train_labels, test_texts, test_labels = [], [], [], []
        for intent in intents:
            for split, texts, labels in [("train", train_texts, train_labels),
                                         ("validate", test_texts, test_labels)]:
                filename = f"train_{intent}_full.json" if split == "train" else f"validate_{intent}.json"
                with urlopen(f"{base}/{intent}/{filename}", timeout=60) as response:
                    raw = response.read()
                # The original benchmark includes both UTF-8 and Latin-1 files.
                try:
                    document = raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    document = raw.decode("latin-1")
                records = json.loads(document)[intent]
                for record in records:
                    texts.append("".join(piece["text"] for piece in record["data"]))
                    labels.append(intent)
        output_dir = self.data_dir / "snips"
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / "train.json", "w", encoding="utf-8") as f:
            json.dump({"texts": train_texts, "labels": train_labels}, f)
        with open(output_dir / "test.json", "w", encoding="utf-8") as f:
            json.dump({"texts": test_texts, "labels": test_labels}, f)

        print(f"✓ Downloaded {len(train_texts)} training examples")
        print(f"✓ Downloaded {len(test_texts)} test examples")
        print(f"✓ Saved to: {output_dir}")
        return output_dir

    def _create_snips_fallback(self) -> Path | None:
        """Explicit tiny offline fixture; never silently replaces real downloads."""
        output_dir = self.data_dir / "snips"
        output_dir.mkdir(exist_ok=True)

        # Six toy intent groups for offline exercises, not the seven-class
        # original SNIPS benchmark.
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

        train_texts, train_labels, test_texts, test_labels = stratified_holdout(
            train_texts, train_labels,
        )

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

        # Load the publisher's CSV files with the generic builder. HF's old
        # banking77.py loader is incompatible with datasets 4+.
        base = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
        dataset = load_dataset("csv", data_files={
            "train": f"{base}/train.csv", "test": f"{base}/test.csv",
        })

        output_dir = self.data_dir / "banking77"
        output_dir.mkdir(exist_ok=True)

        train_texts = [ex["text"] for ex in dataset["train"]]
        train_labels = [ex["category"] for ex in dataset["train"]]
        test_texts = [ex["text"] for ex in dataset["test"]]
        test_labels = [ex["category"] for ex in dataset["test"]]

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

        dataset = load_dataset("Salesforce/wikitext", version)

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

        generate_synthetic_data.generate_intent_data(output_dir=str(self.data_dir))
        generate_synthetic_data.generate_sentiment_data(output_dir=str(self.data_dir))
        generate_synthetic_data.generate_text_corpus(output_dir=str(self.data_dir))
        generate_synthetic_data.generate_rag_knowledge_base(output_dir=str(self.data_dir))
        generate_synthetic_data.generate_rag_eval_queries(output_dir=str(self.data_dir))
        generate_synthetic_data.generate_lora_chat_data(output_dir=str(self.data_dir))


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

    if args.max_samples < 1:
        parser.error("--max-samples must be positive")
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