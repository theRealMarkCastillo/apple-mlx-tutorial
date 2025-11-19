#!/usr/bin/env python3
"""
Dataset Download Script
=======================

Downloads and prepares datasets for NLP training.

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --imdb --snips
    python scripts/download_datasets.py --sentiment
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' library not installed.")
    print("Please run: pip install datasets")
    sys.exit(1)


class DatasetDownloader:
    """Download and prepare datasets"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_imdb(self, max_samples=25000):
        """Download IMDB movie reviews"""
        print("\n" + "="*60)
        print("DOWNLOADING IMDB DATASET")
        print("="*60)
        
        dataset = load_dataset("imdb")
        
        output_dir = self.data_dir / "imdb"
        output_dir.mkdir(exist_ok=True)
        
        # Save train split
        train_texts = []
        train_labels = []
        
        for i, example in enumerate(dataset['train']):
            if i >= max_samples:
                break
            train_texts.append(example['text'])
            train_labels.append(example['label'])
        
        with open(output_dir / 'train.json', 'w') as f:
            json.dump({
                'texts': train_texts,
                'labels': train_labels
            }, f)
        
        # Save test split
        test_texts = []
        test_labels = []
        
        for i, example in enumerate(dataset['test']):
            if i >= max_samples:
                break
            test_texts.append(example['text'])
            test_labels.append(example['label'])
        
        with open(output_dir / 'test.json', 'w') as f:
            json.dump({
                'texts': test_texts,
                'labels': test_labels
            }, f)
        
        print(f"✓ Downloaded {len(train_texts)} training examples")
        print(f"✓ Downloaded {len(test_texts)} test examples")
        print(f"✓ Saved to: {output_dir}")
        
        return output_dir
    
    def download_snips(self):
        """Download SNIPS intent dataset"""
        print("\n" + "="*60)
        print("DOWNLOADING SNIPS DATASET")
        print("="*60)
        
        try:
            dataset = load_dataset("snips_built_in_intents")
            
            output_dir = self.data_dir / "snips"
            output_dir.mkdir(exist_ok=True)
            
            # Process train split
            train_texts = [ex['text'] for ex in dataset['train']]
            train_labels = [ex['label'] for ex in dataset['train']]
            
            with open(output_dir / 'train.json', 'w', encoding='utf-8') as f:
                json.dump({
                    'texts': train_texts,
                    'labels': train_labels
                }, f)
            
            # Check if test split exists
            if 'test' in dataset:
                test_texts = [ex['text'] for ex in dataset['test']]
                test_labels = [ex['label'] for ex in dataset['test']]
            else:
                # Create test split from train (20%)
                split_idx = int(len(train_texts) * 0.8)
                test_texts = train_texts[split_idx:]
                test_labels = train_labels[split_idx:]
                train_texts = train_texts[:split_idx]
                train_labels = train_labels[:split_idx]
                
                # Re-save train split
                with open(output_dir / 'train.json', 'w', encoding='utf-8') as f:
                    json.dump({
                        'texts': train_texts,
                        'labels': train_labels
                    }, f)
            
            with open(output_dir / 'test.json', 'w', encoding='utf-8') as f:
                json.dump({
                    'texts': test_texts,
                    'labels': test_labels
                }, f)
            
            print(f"✓ Downloaded {len(train_texts)} training examples")
            print(f"✓ Downloaded {len(test_texts)} test examples")
            print(f"✓ Saved to: {output_dir}")
            
            return output_dir
            
        except Exception as e:
            print(f"Error loading SNIPS dataset: {e}")
            print("Using alternative SNIPS source...")
            # Fallback to manual data
            self._create_snips_fallback()
            return
    
    def _create_snips_fallback(self):
        """Create SNIPS-like dataset fallback"""
        output_dir = self.data_dir / "snips"
        output_dir.mkdir(exist_ok=True)
        
        # Sample data for each intent
        intents_data = {
            'PlayMusic': [
                "play some music",
                "play my favorite song",
                "start playing music",
                "can you play a song",
                "play something from the Beatles",
            ],
            'GetWeather': [
                "what's the weather like",
                "how's the weather today",
                "will it rain tomorrow",
                "what's the forecast",
                "is it going to be sunny",
            ],
            'BookRestaurant': [
                "book a table for two",
                "make a reservation at a restaurant",
                "find me a place to eat",
                "reserve a table for dinner",
                "book a restaurant for tonight",
            ],
            'SearchCreativeWork': [
                "find me a good movie",
                "search for books by Stephen King",
                "show me romantic comedies",
                "find songs by Taylor Swift",
                "search for action movies",
            ],
            'AddToPlaylist': [
                "add this to my playlist",
                "save this song to my favorites",
                "add to my workout playlist",
                "put this in my playlist",
                "save to my music collection",
            ],
            'RateBook': [
                "rate this book 5 stars",
                "give this book a good review",
                "I rate this book highly",
                "this book deserves 4 stars",
                "rate this book positively",
            ],
        }
        
        # Create training data
        train_texts = []
        train_labels = []
        
        for label, texts in intents_data.items():
            train_texts.extend(texts)
            train_labels.extend([label] * len(texts))
        
        with open(output_dir / 'train.json', 'w') as f:
            json.dump({
                'texts': train_texts,
                'labels': train_labels
            }, f)
        
        # Create test data (subset)
        test_texts = [t for i, t in enumerate(train_texts) if i % 5 == 0]
        test_labels = [l for i, l in enumerate(train_labels) if i % 5 == 0]
        
        with open(output_dir / 'test.json', 'w') as f:
            json.dump({
                'texts': test_texts,
                'labels': test_labels
            }, f)
        
        print(f"✓ Created {len(train_texts)} training examples")
        print(f"✓ Created {len(test_texts)} test examples")
        print(f"✓ Saved to: {output_dir}")
    
    def download_banking77(self):
        """Download Banking77 intent dataset"""
        print("\n" + "="*60)
        print("DOWNLOADING BANKING77 DATASET")
        print("="*60)
        
        try:
            dataset = load_dataset("banking77")
        except:
            print("ERROR: Could not download Banking77 dataset")
            return None
        
        output_dir = self.data_dir / "banking77"
        output_dir.mkdir(exist_ok=True)
        
        # Save train split
        train_texts = [ex['text'] for ex in dataset['train']]
        train_labels = [ex['label'] for ex in dataset['train']]
        
        with open(output_dir / 'train.json', 'w') as f:
            json.dump({
                'texts': train_texts,
                'labels': train_labels
            }, f)
        
        # Save test split
        test_texts = [ex['text'] for ex in dataset['test']]
        test_labels = [ex['label'] for ex in dataset['test']]
        
        with open(output_dir / 'test.json', 'w') as f:
            json.dump({
                'texts': test_texts,
                'labels': test_labels
            }, f)
        
        print(f"✓ Downloaded {len(train_texts)} training examples")
        print(f"✓ Downloaded {len(test_texts)} test examples")
        print(f"✓ Saved to: {output_dir}")
        
        return output_dir
    
    def download_wikitext(self, version='wikitext-2-v1'):
        """Download WikiText for text generation"""
        print("\n" + "="*60)
        print(f"DOWNLOADING WIKITEXT DATASET ({version})")
        print("="*60)
        
        dataset = load_dataset("wikitext", version)
        
        output_dir = self.data_dir / "wikitext"
        output_dir.mkdir(exist_ok=True)
        
        # Save train split
        with open(output_dir / 'train.txt', 'w') as f:
            for example in dataset['train']:
                if example['text'].strip():
                    f.write(example['text'] + '\n')
        
        # Save validation split
        with open(output_dir / 'validation.txt', 'w') as f:
            for example in dataset['validation']:
                if example['text'].strip():
                    f.write(example['text'] + '\n')
        
        # Save test split
        with open(output_dir / 'test.txt', 'w') as f:
            for example in dataset['test']:
                if example['text'].strip():
                    f.write(example['text'] + '\n')
        
        print(f"✓ Downloaded WikiText {version}")
        print(f"✓ Saved to: {output_dir}")
        
        return output_dir
    
    def create_sample_datasets(self):
        """Create small sample datasets for quick testing"""
        print("\n" + "="*60)
        print("CREATING SAMPLE DATASETS")
        print("="*60)
        
        # Intent classification samples
        intent_dir = self.data_dir / "intent_samples"
        intent_dir.mkdir(exist_ok=True)
        
        intent_data = {
            'texts': [
                "hello there",
                "hi how are you",
                "good morning",
                "what time is it",
                "when is the meeting",
                "how do I do this",
                "please turn on the lights",
                "start the music",
                "open the door",
            ],
            'labels': [
                'greeting', 'greeting', 'greeting',
                'question', 'question', 'question',
                'command', 'command', 'command'
            ]
        }
        
        with open(intent_dir / 'data.json', 'w') as f:
            json.dump(intent_data, f, indent=2)
        
        print(f"✓ Created intent samples: {intent_dir}")
        
        # Sentiment analysis samples
        sentiment_dir = self.data_dir / "sentiment_samples"
        sentiment_dir.mkdir(exist_ok=True)
        
        sentiment_data = {
            'texts': [
                "This movie was absolutely fantastic!",
                "I loved every minute of it",
                "Best film I've seen this year",
                "Terrible waste of time",
                "Very disappointed with this product",
                "Would not recommend",
                "It was okay, nothing special",
                "Average movie, not bad not great",
            ],
            'labels': [
                'positive', 'positive', 'positive',
                'negative', 'negative', 'negative',
                'neutral', 'neutral'
            ]
        }
        
        with open(sentiment_dir / 'data.json', 'w') as f:
            json.dump(sentiment_data, f, indent=2)
        
        print(f"✓ Created sentiment samples: {sentiment_dir}")
        
        # Text generation samples
        text_gen_dir = self.data_dir / "text_gen_samples"
        text_gen_dir.mkdir(exist_ok=True)
        
        with open(text_gen_dir / 'corpus.txt', 'w') as f:
            f.write("""The quick brown fox jumps over the lazy dog.
Machine learning is a subset of artificial intelligence.
Natural language processing helps computers understand human language.
Deep learning uses neural networks with multiple layers.
Apple Silicon provides unified memory architecture for ML acceleration.
MLX is Apple's framework for machine learning on Apple Silicon.
""")
        
        print(f"✓ Created text generation samples: {text_gen_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download NLP datasets")
    
    parser.add_argument('--all', action='store_true',
                       help='Download all datasets')
    parser.add_argument('--sentiment', action='store_true',
                       help='Download sentiment analysis datasets')
    parser.add_argument('--intent', action='store_true',
                       help='Download intent classification datasets')
    parser.add_argument('--generation', action='store_true',
                       help='Download text generation datasets')
    
    # Individual datasets
    parser.add_argument('--imdb', action='store_true',
                       help='Download IMDB reviews')
    parser.add_argument('--snips', action='store_true',
                       help='Download SNIPS intents')
    parser.add_argument('--banking77', action='store_true',
                       help='Download Banking77 intents')
    parser.add_argument('--wikitext', action='store_true',
                       help='Download WikiText')
    parser.add_argument('--samples', action='store_true',
                       help='Create small sample datasets')
    
    parser.add_argument('--data-dir', default='data',
                       help='Directory to save datasets (default: data)')
    parser.add_argument('--max-samples', type=int, default=25000,
                       help='Maximum samples for large datasets (default: 25000)')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    downloader = DatasetDownloader(args.data_dir)
    
    print("="*60)
    print("DATASET DOWNLOADER")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Max samples: {args.max_samples}")
    
    # Download based on flags
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
    
    print("\n" + "="*60)
    print("✓ DOWNLOAD COMPLETE!")
    print("="*60)
    print(f"\nDatasets saved to: {args.data_dir}/")
    print("\nAvailable datasets:")
    
    data_path = Path(args.data_dir)
    if data_path.exists():
        for subdir in sorted(data_path.iterdir()):
            if subdir.is_dir():
                files = list(subdir.glob('*.json')) + list(subdir.glob('*.txt'))
                print(f"  • {subdir.name}/ ({len(files)} files)")
    
    print("\nNext steps:")
    print("  1. Run examples: python examples_with_real_data.py")
    print("  2. Train models: python train_with_real_data.py")
    print("  3. See documentation: docs/DATASETS_AND_PREPROCESSING.md")


if __name__ == "__main__":
    main()
