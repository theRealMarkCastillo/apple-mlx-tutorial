import json
import random
from pathlib import Path

def generate_intent_data():
    print("Generating synthetic Intent Classification data...")
    
    greetings = [
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening", 
        "hi there", "hello world", "greetings", "hey there", "what's up", 
        "howdy", "yo", "hi friend", "hello everyone", "good day", "morning",
        "evening", "hi folks", "hello team"
    ]
    
    questions = [
        "what time is it", "how do I do this", "when is the meeting", 
        "where is the office", "who is the ceo", "why is the sky blue",
        "what is the weather like", "how much does it cost", "can you help me",
        "what is your name", "how does this work", "where can I find help",
        "what is the capital of France", "when does the store open",
        "who are you", "why is this not working", "what are the hours",
        "how long will it take", "is this correct", "can I ask a question"
    ]
    
    commands = [
        "turn on the lights", "play music", "stop", "go away", "open the door",
        "close the window", "set an alarm", "remind me to call mom",
        "send an email", "call john", "turn off the tv", "volume up",
        "volume down", "mute", "pause", "resume", "skip track",
        "show me the map", "navigate home", "lock the door"
    ]
    
    # Generate variations
    data = {"texts": [], "labels": []}
    
    # Add base data
    for text in greetings:
        data["texts"].append(text)
        data["labels"].append("greeting")
        
    for text in questions:
        data["texts"].append(text)
        data["labels"].append("question")
        
    for text in commands:
        data["texts"].append(text)
        data["labels"].append("command")
        
    # Generate synthetic variations
    modifiers = ["please", "could you", "can you", "hey", "ok"]
    
    for _ in range(50):
        # Synthetic Commands
        cmd = random.choice(commands)
        mod = random.choice(modifiers)
        if random.random() > 0.5:
            text = f"{mod} {cmd}"
        else:
            text = f"{cmd} {mod}"
        data["texts"].append(text)
        data["labels"].append("command")
        
        # Synthetic Questions
        q = random.choice(questions)
        data["texts"].append(f"{q} please")
        data["labels"].append("question")
        
    # Save
    output_path = Path("data/intent_samples/data.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data['texts'])} intent examples to {output_path}")

def generate_sentiment_data():
    print("Generating synthetic Sentiment Analysis data...")
    
    positives = [
        "This is great", "I love this", "Amazing work", "Fantastic", "Excellent",
        "Very good", "Best ever", "So happy", "Wonderful experience", "Highly recommend",
        "Perfect", "Outstanding", "Brilliant", "Superb", "Awesome", "Delightful",
        "Enjoyed it a lot", "Very satisfied", "Top notch", "Five stars"
    ]
    
    negatives = [
        "This is terrible", "I hate this", "Worst ever", "Awful", "Bad experience",
        "Very disappointed", "Waste of time", "Do not buy", "Horrible", "Poor quality",
        "Useless", "Broken", "Garbage", "Annoying", "Frustrating", "Not good",
        "Regret buying", "Terrible service", "Disaster", "Never again"
    ]
    
    neutrals = [
        "It is okay", "Average", "Not bad", "Could be better", "It is what it is",
        "Fine", "Mediocre", "Nothing special", "Just okay", "Standard",
        "As expected", "Normal", "Typical", "Fair", "So-so", "Alright",
        "Middle of the road", "Passable", "Decent", "Acceptable"
    ]
    
    data = {"texts": [], "labels": []}
    
    # Add base data
    for text in positives:
        data["texts"].append(text)
        data["labels"].append("positive")
    for text in negatives:
        data["texts"].append(text)
        data["labels"].append("negative")
    for text in neutrals:
        data["texts"].append(text)
        data["labels"].append("neutral")
        
    # Generate variations
    subjects = ["movie", "food", "service", "product", "app", "game", "book"]
    
    for _ in range(30):
        subj = random.choice(subjects)
        
        # Positive
        adj = random.choice(["great", "good", "nice", "cool", "amazing"])
        data["texts"].append(f"The {subj} was {adj}")
        data["labels"].append("positive")
        
        # Negative
        adj = random.choice(["bad", "terrible", "awful", "slow", "boring"])
        data["texts"].append(f"The {subj} was {adj}")
        data["labels"].append("negative")
        
        # Neutral
        adj = random.choice(["okay", "fine", "average", "alright"])
        data["texts"].append(f"The {subj} was {adj}")
        data["labels"].append("neutral")

    # Save
    output_path = Path("data/sentiment_samples/data.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data['texts'])} sentiment examples to {output_path}")

def generate_text_corpus():
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
    
    # Repeat and vary slightly to increase size
    corpus = base_text * 5
    
    output_path = Path("data/text_gen_samples/corpus.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(corpus)
    print(f"Saved {len(corpus)} characters to {output_path}")

def generate_rag_knowledge_base():
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
        "Top-p (nucleus) sampling limits the choice to the smallest set of tokens whose cumulative probability exceeds p."
    ]
    
    # Generate more variations
    topics = ["MLX", "Apple Silicon", "Deep Learning", "LLMs"]
    for i in range(30):
        topic = random.choice(topics)
        documents.append(f"Synthetic document #{i} about {topic} containing random facts to increase the database size.")
        
    output_path = Path("data/rag_samples/knowledge_base.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(documents, f, indent=2)
    print(f"Saved {len(documents)} documents to {output_path}")

if __name__ == "__main__":
    generate_intent_data()
    generate_sentiment_data()
    generate_text_corpus()
    generate_rag_knowledge_base()
