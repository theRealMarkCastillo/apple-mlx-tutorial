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

    # Do not replicate the corpus across the chronological validation split.
    corpus = base_text.strip() + "\n"
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


# Labeled retrieval queries for notebooks 09 and 10. Each query names the index
# of the one knowledge-base document that answers it. "lexical" queries reuse
# the document's wording; "paraphrase" queries ask the same thing in other
# words, which is where bag-of-words retrieval breaks down.
RAG_EVAL_QUERIES: list[tuple[str, int, str]] = [
    ("What is MLX?", 0, "lexical"),
    ("Which library did Apple build for training models on Mac chips?", 0, "paraphrase"),
    ("How does unified memory let the CPU and GPU share memory?", 1, "lexical"),
    ("Do I have to copy tensors between the processor and graphics card on a Mac?", 1, "paraphrase"),
    ("What is lazy evaluation in MLX?", 2, "lexical"),
    ("Why isn't my array calculated until I print it?", 2, "paraphrase"),
    ("Why are LSTMs hard to parallelize?", 3, "lexical"),
    ("What is the drawback of recurrent nets that read one word at a time?", 3, "paraphrase"),
    ("How do transformers use attention to process sequences in parallel?", 4, "lexical"),
    ("Which architecture looks at every token at once instead of step by step?", 4, "paraphrase"),
    ("What does LoRA do to pre-trained weights?", 5, "lexical"),
    ("How can I adapt a big network by training only a few small extra matrices?", 5, "paraphrase"),
    ("How does quantization reduce the precision of model weights?", 6, "lexical"),
    ("How can I shrink a network so it needs less RAM?", 6, "paraphrase"),
    ("What is RAG retrieval augmented generation?", 7, "lexical"),
    ("How can a chatbot answer with facts newer than its training cutoff?", 7, "paraphrase"),
    ("What do vector databases store?", 8, "lexical"),
    ("How do I find documents that mean the same thing even with different words?", 8, "paraphrase"),
    ("What is the Neural Engine NPU?", 9, "lexical"),
    ("Which dedicated chip on a Mac speeds up inference?", 9, "paraphrase"),
    ("Does MLX support automatic differentiation?", 10, "lexical"),
    ("Can the framework compute gradients for me?", 10, "paraphrase"),
    ("What is fine-tuning a pre-trained model?", 11, "lexical"),
    ("How do I specialize an existing network for my own task?", 11, "paraphrase"),
    ("What is prompt engineering?", 12, "lexical"),
    ("How should I word my instructions to get better replies?", 12, "paraphrase"),
    ("What is zero-shot learning?", 13, "lexical"),
    ("Can a system handle a job it was never shown examples of?", 13, "paraphrase"),
    ("What is few-shot learning?", 14, "lexical"),
    ("Should I include a couple of worked examples in my request?", 14, "paraphrase"),
    ("What is chain-of-thought prompting?", 15, "lexical"),
    ("How do I get the assistant to show its working before answering?", 15, "paraphrase"),
    ("What is hallucination in an LLM?", 16, "lexical"),
    ("Why does the chatbot confidently make things up?", 16, "paraphrase"),
    ("What does temperature control?", 17, "lexical"),
    ("How do I make generated text less random?", 17, "paraphrase"),
    ("What is top-k sampling?", 18, "lexical"),
    ("How do I restrict generation to the few most probable candidates?", 18, "paraphrase"),
    ("What is top-p nucleus sampling?", 19, "lexical"),
    ("Which decoding method keeps the smallest set whose probabilities add up to a threshold?", 19, "paraphrase"),
]


def generate_rag_eval_queries(output_dir: str = "data") -> Path:
    """Write labeled retrieval queries to ``rag_samples/eval_queries.json``."""
    rows = [{"query": q, "relevant_doc": doc, "kind": kind} for q, doc, kind in RAG_EVAL_QUERIES]
    output_path = Path(output_dir) / "rag_samples" / "eval_queries.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved {len(rows)} labeled retrieval queries to {output_path}")
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

def generate_lora_chat_data(
    n_train: int = 800,
    n_val: int = 200,
    output_dir: str = "data",
    seed: int = DEFAULT_SEED,
) -> tuple[Path, Path]:
    """
    Generate chat-format data for the LoRA fine-tuning notebook.

    Writes ``data/train.jsonl`` and ``data/valid.jsonl`` (the filenames
    ``mlx_lm.lora`` expects out of the box) in OpenAI-style
    ``{"messages": [...]}`` format. The dataset is large enough to demonstrate
    the LoRA training mechanics end-to-end on Apple Silicon while still being
    fast enough to finish in a few minutes on a 4-bit Qwen2.5 base model.
    """
    _seed_random(seed)
    print(f"Generating synthetic LoRA chat data ({n_train} train / {n_val} val)...")

    rng = random.Random(seed)
    train_records: list[dict] = []
    val_records: list[dict] = []

    if n_train < 1 or n_val < 1:
        raise ValueError("LoRA split sizes must be positive")
    # Split source prompts BEFORE sampling repetitions. No validation prompt
    # can occur in training, even with hundreds of examples from this tiny pool.
    train_templates, val_templates = {}, {}
    for intent, templates in INTENT_TEMPLATES.items():
        templates = list(templates)
        rng.shuffle(templates)
        cut = max(1, len(templates) // 5)
        val_templates[intent] = templates[:cut]
        train_templates[intent] = templates[cut:]
    for _ in range(n_train):
        train_records.append(_sample_chat(rng, train_templates))
    for _ in range(n_val):
        val_records.append(_sample_chat(rng, val_templates))

    train_path = Path(output_dir) / "train.jsonl"
    val_path = Path(output_dir) / "valid.jsonl"
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


def _sample_chat(rng: random.Random, templates: dict[str, list[str]]) -> dict:
    """Produce a single chat-format example with a system + user + assistant turn."""
    intent = rng.choice(list(templates))
    user_msg = rng.choice(templates[intent])
    assistant_msg = intent
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Classify the user message. Reply with exactly one label: "
                    "greeting, question, or command."
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
    generate_rag_eval_queries(args.data_dir)
    if not args.skip_lora:
        generate_lora_chat_data(
            n_train=args.lora_train,
            n_val=args.lora_val,
            output_dir=args.data_dir,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()