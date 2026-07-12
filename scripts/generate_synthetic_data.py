"""
Synthetic data generators for the MLX NLP tutorial.

This is the single canonical source of sample data for the notebooks. The
``--samples`` flag in ``download_datasets.py`` delegates here, and the
notebooks themselves load these JSON / JSONL files directly.

All generators are deterministic given ``seed`` so re-running produces
identical files — handy for debugging notebook outputs.
"""

import argparse
import json
import random
from pathlib import Path

DEFAULT_SEED = 42

GREETINGS = [
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "hi there", "hello world", "greetings", "hey there", "what's up",
    "howdy", "yo", "hi friend", "hello everyone", "good day", "morning",
    "evening", "hi folks", "hello team",
]

QUESTIONS = [
    "what time is it", "how do I do this", "when is the meeting",
    "where is the office", "who is the ceo", "why is the sky blue",
    "what is the weather like", "how much does it cost", "can you help me",
    "what is your name", "how does this work", "where can I find help",
    "what is the capital of France", "when does the store open",
    "who are you", "why is this not working", "what are the hours",
    "how long will it take", "is this correct", "can I ask a question",
]

COMMANDS = [
    "turn on the lights", "play music", "stop", "go away", "open the door",
    "close the window", "set an alarm", "remind me to call mom",
    "send an email", "call john", "turn off the tv", "volume up",
    "volume down", "mute", "pause", "resume", "skip track",
    "show me the map", "navigate home", "lock the door",
]

POSITIVE_PHRASES = [
    "This is great", "I love this", "Amazing work", "Fantastic", "Excellent",
    "Very good", "Best ever", "So happy", "Wonderful experience", "Highly recommend",
    "Perfect", "Outstanding", "Brilliant", "Superb", "Awesome", "Delightful",
    "Enjoyed it a lot", "Very satisfied", "Top notch", "Five stars",
]

NEGATIVE_PHRASES = [
    "This is terrible", "I hate this", "Worst ever", "Awful", "Bad experience",
    "Very disappointed", "Waste of time", "Do not buy", "Horrible", "Poor quality",
    "Useless", "Broken", "Garbage", "Annoying", "Frustrating", "Not good",
    "Regret buying", "Terrible service", "Disaster", "Never again",
]

NEUTRAL_PHRASES = [
    "It is okay", "Average", "Not bad", "Could be better", "It is what it is",
    "Fine", "Mediocre", "Nothing special", "Just okay", "Standard",
    "As expected", "Normal", "Typical", "Fair", "So-so", "Alright",
    "Middle of the road", "Passable", "Decent", "Acceptable",
]

POSITIVE_ADJECTIVES = ["great", "good", "nice", "cool", "amazing"]
NEGATIVE_ADJECTIVES = ["bad", "terrible", "awful", "slow", "boring"]
NEUTRAL_ADJECTIVES = ["okay", "fine", "average", "alright"]
SUBJECTS = ["movie", "food", "service", "product", "app", "game", "book"]


def _seed_random(seed: int) -> None:
    random.seed(seed)


def generate_intent_data(output_dir: str = "data", seed: int = DEFAULT_SEED) -> Path:
    """Generate intent-classification samples (greeting / question / command)."""
    _seed_random(seed)
    print("Generating synthetic Intent Classification data...")

    data: dict[str, list] = {"texts": [], "labels": []}
    for text in GREETINGS:
        data["texts"].append(text)
        data["labels"].append("greeting")
    for text in QUESTIONS:
        data["texts"].append(text)
        data["labels"].append("question")
    for text in COMMANDS:
        data["texts"].append(text)
        data["labels"].append("command")

    modifiers = ["please", "could you", "can you", "hey", "ok"]
    for _ in range(50):
        cmd = random.choice(COMMANDS)
        mod = random.choice(modifiers)
        text = f"{mod} {cmd}" if random.random() > 0.5 else f"{cmd} {mod}"
        data["texts"].append(text)
        data["labels"].append("command")

        question = random.choice(QUESTIONS)
        data["texts"].append(f"{question} please")
        data["labels"].append("question")

    output_path = Path(output_dir) / "intent_samples" / "data.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data['texts'])} intent examples to {output_path}")
    return output_path


def generate_sentiment_data(output_dir: str = "data", seed: int = DEFAULT_SEED) -> Path:
    """Generate sentiment-analysis samples (positive / negative / neutral)."""
    _seed_random(seed)
    print("Generating synthetic Sentiment Analysis data...")

    data: dict[str, list] = {"texts": [], "labels": []}
    for text in POSITIVE_PHRASES:
        data["texts"].append(text)
        data["labels"].append("positive")
    for text in NEGATIVE_PHRASES:
        data["texts"].append(text)
        data["labels"].append("negative")
    for text in NEUTRAL_PHRASES:
        data["texts"].append(text)
        data["labels"].append("neutral")

    for _ in range(30):
        subj = random.choice(SUBJECTS)
        data["texts"].append(f"The {subj} was {random.choice(POSITIVE_ADJECTIVES)}")
        data["labels"].append("positive")
        data["texts"].append(f"The {subj} was {random.choice(NEGATIVE_ADJECTIVES)}")
        data["labels"].append("negative")
        data["texts"].append(f"The {subj} was {random.choice(NEUTRAL_ADJECTIVES)}")
        data["labels"].append("neutral")

    output_path = Path(output_dir) / "sentiment_samples" / "data.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data['texts'])} sentiment examples to {output_path}")
    return output_path


def generate_text_corpus(output_dir: str = "data") -> Path:
    """Generate the small text-generation corpus."""
    print("Generating synthetic Text Generation corpus...")

    base_text = """
    Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions. Recently, artificial neural networks have been able to surpass many previous approaches in performance.

    MLX is an array framework for machine learning on Apple silicon, brought to you by the Apple machine learning research team.
    MLX is designed by machine learning researchers for machine learning researchers. The framework is intended to be user-friendly, but still efficient to train and deploy models. The design of the framework itself is also conceptually simple. We intend to make it easy for researchers to extend and improve MLX with the goal of quickly exploring new ideas.

    The design of MLX is inspired by frameworks like NumPy, PyTorch, Jax, and ArrayFire. A notable difference from these frameworks and MLX is the unified memory model. Arrays in MLX live in shared memory. Operations on MLX arrays can be performed on any of the supported device types without moving data. Currently supported device types are the CPU and the GPU.

    Key features of MLX include:
    Familiar APIs: MLX has a Python API that closely follows NumPy. MLX also has fully featured C++, C, and Swift APIs, which closely mirror the Python API. MLX has higher-level packages like mlx.nn and mlx.optimizers with APIs that closely follow PyTorch to simplify building more complex models.
    Composable function transformations: MLX has composable function transformations for automatic differentiation, automatic vectorization, and computation graph optimization.
    Lazy computation: Computations in MLX are lazy. Arrays are only materialized when needed.
    Dynamic graph construction: Computation graphs in MLX are constructed dynamically. Changing the shapes of function arguments does not trigger slow compilations, and debugging is simple and intuitive.
    Multi-device: Operations can run on any of the supported devices (currently the CPU and the GPU).
    Unified memory: A notable difference from other frameworks and MLX is the unified memory model. Arrays in MLX live in shared memory. Operations on MLX arrays can be performed on any of the supported device types without moving data.
    """

    corpus = base_text * 5
    output_path = Path(output_dir) / "text_gen_samples" / "corpus.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"Saved {len(corpus)} characters to {output_path}")
    return output_path


def generate_rag_knowledge_base(output_dir: str = "data", seed: int = DEFAULT_SEED) -> Path:
    """Generate the small RAG knowledge base (real docs + a few filler rows)."""
    _seed_random(seed)
    print("Generating synthetic RAG Knowledge Base...")

    documents = [
        "MLX is an array framework for machine learning on Apple Silicon, brought to you by Apple machine learning research.",
        "The Unified Memory architecture of M1/M2/M3 chips allows the CPU and GPU to share the same memory pool.",
        "Unlike CUDA, MLX uses lazy evaluation, meaning computations are only executed when the result is needed.",
        "LSTMs are recurrent neural networks capable of learning long-term dependencies, but they are sequential and hard to parallelize.",
        "Transformers use the attention mechanism to process input sequences in parallel, making them faster to train than RNNs.",
        "LoRA (Low-Rank Adaptation) freezes pre-trained model weights and injects trainable rank decomposition matrices.",
        "Quantization reduces the precision of model weights (e.g., from 16-bit to 4-bit) to save memory and increase speed.",
        "RAG (Retrieval Augmented Generation) combines an LLM with a retrieval system to provide up-to-date information.",
        "Vector databases store embeddings of text, allowing for semantic search based on meaning rather than keywords.",
        "Apple Silicon's Neural Engine is a specialized NPU designed for accelerating machine learning inference.",
        "MLX supports automatic differentiation, vectorization, and computation graph optimization.",
        "Fine-tuning allows you to adapt a pre-trained model to a specific task or dataset.",
        "Prompt engineering is the art of crafting inputs to guide an LLM to generate desired outputs.",
        "Zero-shot learning is the ability of a model to perform a task without seeing any examples during training.",
        "Few-shot learning involves providing a small number of examples to the model at inference time.",
        "Chain-of-thought prompting encourages the model to explain its reasoning step-by-step.",
        "Hallucination is when an LLM generates incorrect or nonsensical information confidently.",
        "Temperature is a hyperparameter that controls the randomness of the model's output.",
        "Top-k sampling limits the model's choice to the k most likely next tokens.",
        "Top-p (nucleus) sampling limits the choice to the smallest set of tokens whose cumulative probability exceeds p.",
    ]

    topics = ["MLX", "Apple Silicon", "Deep Learning", "LLMs"]
    for i in range(30):
        topic = random.choice(topics)
        documents.append(f"Synthetic document #{i} about {topic} containing random facts to increase the database size.")

    output_path = Path(output_dir) / "rag_samples" / "knowledge_base.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)
    print(f"Saved {len(documents)} documents to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# LoRA chat-format data
# ---------------------------------------------------------------------------

# Templated intents with a few canonical rephrasings. We sample and shuffle to
# produce a chat-format dataset that mirrors what a real intent-classifier
# assistant might see.
INTENT_TEMPLATES: dict[str, list[str]] = {
    "greeting": [
        "Hello!", "Hi there.", "Hey, how's it going?", "Good morning!",
        "Hi!", "Hey!", "Hello, can you help me?", "Hi, what's up?",
        "Greetings.", "Hey there!", "Good afternoon.", "Good evening.",
        "Yo!", "Howdy.", "Hi friend.", "Hello world.",
    ],
    "question": [
        "What's the weather today?", "What time is the meeting?",
        "How do I reset my password?", "Where can I find help?",
        "Why is the network slow?", "Who is on call this week?",
        "When does the store open?", "How much does this cost?",
        "What is your name?", "Can you help me with my account?",
        "How long will this take?", "Is this correct?",
        "What are the office hours?", "Where is the office located?",
    ],
    "command": [
        "Turn on the lights.", "Play some music.", "Stop the music.",
        "Set a timer for 10 minutes.", "Send an email to John.",
        "Open the door.", "Close the window.", "Remind me to call mom.",
        "Navigate home.", "Volume up.", "Volume down.", "Mute the audio.",
        "Pause the video.", "Skip this track.", "Lock the door.",
        "Turn off the TV.", "Schedule a meeting for tomorrow at 3pm.",
    ],
}

ASSISTANT_RESPONSES: dict[str, list[str]] = {
    "greeting": [
        "Hi! How can I help you today?",
        "Hello! What can I do for you?",
        "Hey there! Ready when you are.",
    ],
    "question": [
        "Let me look that up for you.",
        "Great question — one moment.",
        "I'll find that information right away.",
    ],
    "command": [
        "Done! Let me know if you need anything else.",
        "Got it. Anything else?",
        "Completed. What's next?",
    ],
}


def generate_lora_chat_data(
    n_train: int = 800,
    n_val: int = 200,
    output_dir: str = "data",
    seed: int = DEFAULT_SEED,
) -> tuple[Path, Path]:
    """
    Generate chat-format data for the LoRA fine-tuning notebook.

    Writes ``data/lora_train.jsonl`` and ``data/lora_valid.jsonl`` in OpenAI-
    style ``{"messages": [...]}`` format. The dataset is large enough to
    demonstrate the LoRA training mechanics end-to-end on Apple Silicon while
    still being fast enough to finish in a few minutes on a 4-bit Llama-3.2
    base model.
    """
    _seed_random(seed)
    print(f"Generating synthetic LoRA chat data ({n_train} train / {n_val} val)...")

    rng = random.Random(seed)
    train_records: list[dict] = []
    val_records: list[dict] = []

    for _ in range(n_train):
        train_records.append(_sample_chat(rng))
    for _ in range(n_val):
        val_records.append(_sample_chat(rng))

    train_path = Path(output_dir) / "lora_train.jsonl"
    val_path = Path(output_dir) / "lora_valid.jsonl"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w", encoding="utf-8") as f:
        for record in train_records:
            f.write(json.dumps(record) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for record in val_records:
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(train_records)} train records to {train_path}")
    print(f"Saved {len(val_records)} val records to {val_path}")
    return train_path, val_path


def _sample_chat(rng: random.Random) -> dict:
    """Produce a single chat-format example with a system + user + assistant turn."""
    intent = rng.choice(list(INTENT_TEMPLATES.keys()))
    user_msg = rng.choice(INTENT_TEMPLATES[intent])
    assistant_msg = rng.choice(ASSISTANT_RESPONSES[intent])
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a concise command-line assistant. Identify the user's "
                    "intent (greeting, question, or command) and respond briefly."
                ),
            },
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic data for the MLX NLP tutorial.")
    parser.add_argument("--data-dir", default="data", help="Output directory (default: data)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})")
    parser.add_argument("--lora-train", type=int, default=800, help="LoRA train records (default: 800)")
    parser.add_argument("--lora-val", type=int, default=200, help="LoRA val records (default: 200)")
    parser.add_argument("--skip-lora", action="store_true", help="Skip LoRA chat data generation")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    generate_intent_data(args.data_dir, seed=args.seed)
    generate_sentiment_data(args.data_dir, seed=args.seed)
    generate_text_corpus(args.data_dir)
    generate_rag_knowledge_base(args.data_dir, seed=args.seed)
    if not args.skip_lora:
        generate_lora_chat_data(
            n_train=args.lora_train,
            n_val=args.lora_val,
            output_dir=args.data_dir,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()