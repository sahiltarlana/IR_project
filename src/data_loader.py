"""
Data loading module for Adaptive-RAG experiments.
Loads samples from single-hop (SQuAD, NQ, TriviaQA) and multi-hop (MuSiQue, HotpotQA, 2Wiki) datasets.
"""
import json
import random
import os
from datasets import load_dataset

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_cache')  # stays in IR/data_cache

def load_squad(n=50):
    ds = load_dataset("rajpurkar/squad", split="validation", cache_dir=CACHE_DIR)
    samples = random.sample(range(len(ds)), min(n, len(ds)))
    return [{"question": ds[i]["question"], "answer": ds[i]["answers"]["text"][0],
             "context": ds[i]["context"], "dataset": "SQuAD", "complexity": "B"} for i in samples]

def load_nq(n=50):
    ds = load_dataset("nq_open", split="validation", cache_dir=CACHE_DIR)
    samples = random.sample(range(len(ds)), min(n, len(ds)))
    return [{"question": ds[i]["question"], "answer": ds[i]["answer"][0],
             "context": "", "dataset": "NQ", "complexity": "B"} for i in samples]

def load_triviaqa(n=50):
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation", cache_dir=CACHE_DIR)
    samples = random.sample(range(len(ds)), min(n, len(ds)))
    return [{"question": ds[i]["question"], "answer": ds[i]["answer"]["value"],
             "context": "", "dataset": "TriviaQA", "complexity": "B"} for i in samples]

def load_hotpotqa(n=50):
    ds = load_dataset("hotpot_qa", "distractor", split="validation", cache_dir=CACHE_DIR)
    samples = random.sample(range(len(ds)), min(n, len(ds)))
    return [{"question": ds[i]["question"], "answer": ds[i]["answer"],
             "context": " ".join(
                 sent for sents in ds[i]["context"]["sentences"] for sent in sents
             ), "dataset": "HotpotQA", "complexity": "C"} for i in samples]

def load_all_datasets(n_per_dataset=50):
    """Load samples from all available datasets."""
    print("Loading datasets...")
    all_data = []
    loaders = [
        ("SQuAD", load_squad),
        ("NQ", load_nq),
        ("TriviaQA", load_triviaqa),
        ("HotpotQA", load_hotpotqa),
    ]
    for name, loader in loaders:
        try:
            data = loader(n_per_dataset)
            all_data.extend(data)
            print(f"  {name}: {len(data)} samples loaded")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
    print(f"Total: {len(all_data)} samples")
    return all_data

# Simple factual questions for "no retrieval" (Level A) testing
SIMPLE_QUESTIONS = [
    {"question": "What is the capital of France?", "answer": "Paris", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "What is the chemical symbol for water?", "answer": "H2O", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "What is the speed of light in vacuum?", "answer": "299792458 meters per second", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "What year did World War II end?", "answer": "1945", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "What is the boiling point of water?", "answer": "100 degrees Celsius", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "Who discovered penicillin?", "answer": "Alexander Fleming", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "What is the square root of 144?", "answer": "12", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "What continent is Brazil in?", "answer": "South America", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "How many days are in a week?", "answer": "7", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "What is the freezing point of water?", "answer": "0 degrees Celsius", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "Who was the first president of the United States?", "answer": "George Washington", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "What gas do plants absorb from the atmosphere?", "answer": "carbon dioxide", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "What is the largest ocean on Earth?", "answer": "Pacific Ocean", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "How many legs does a spider have?", "answer": "8", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "What is the currency of Japan?", "answer": "Yen", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "Who developed the theory of relativity?", "answer": "Albert Einstein", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "What is the tallest mountain in the world?", "answer": "Mount Everest", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "What planet is known as the Red Planet?", "answer": "Mars", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "What is the smallest prime number?", "answer": "2", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "What language is spoken in Brazil?", "answer": "Portuguese", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "How many continents are there?", "answer": "7", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "Who wrote the Odyssey?", "answer": "Homer", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "What is the largest mammal?", "answer": "Blue whale", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo", "context": "", "dataset": "Simple", "complexity": "A"},
    {"question": "How many bones are in the adult human body?", "answer": "206", "context": "", "dataset": "Simple", "complexity": "A"},
]
