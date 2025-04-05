import re
import json
from typing import List, Dict, Tuple
from collections import defaultdict
import unicodedata

# List of negation trigger words
NEG_TRIGGERS = [
    "no", "sin", "niega", "niegan", "negado", "negativa", "no hay evidencia de",
    "ausencia de", "descarta", "descartan", "no presenta",
    "negativo", "negativos", "negativas", "neg", "afebril"
]

# List of uncertainty trigger words
UNC_TRIGGERS = [
    "posible", "probable", "sugiere", "sospecha", "podría", "aparentemente", "dudoso",
    "no se puede descartar", "es posible que",
    "probablemente", "sugestiva de", "sugestivo de", "dudosa", "se orienta", "valorar", "podria",
    "se considera", "no se descarta", "puede ser", "sugiere que", "podría tratarse de",
    "no se puede excluir", "no se descartan", "compatible con", "no concluyente",
    "sugiere diagnóstico de", "pudiera corresponder a", "puede representar", "debería considerarse",
    "presuntivamente", "hallazgos no concluyentes", "aspecto compatible con",
    "sospecha de", "compatibles con", "sugestivos de", "parece", "aparente", "sugestivas de",
    "posiblemente", "probables", "sospechosa de", "dudosamente", "impresiona", "desconoce",
    "posibilidad de", "no se puede asegurar", "difícil valorar", "hallazgos ambiguos",
    "aspecto que podría corresponder", "probabilidad baja de", "no se puede confirmar ni descartar",
    "sin signos concluyentes", "sospechoso de"
]

# Window size for determining the scope of triggers
WINDOW_SIZE = 5

def normalize_token(token: str) -> str:
    """
    Normalizes a token by:
    - Converting to lowercase
    - Removing accents
    - Stripping leading and trailing punctuation

    Parameters:
    - token (str): The input token to normalize.

    Returns:
    - str: The normalized token.
    """
    token = token.lower()
    token = ''.join(c for c in unicodedata.normalize('NFD', token) if unicodedata.category(c) != 'Mn')
    token = re.sub(r"\W+$", "", token)
    token = re.sub(r"^\W+", "", token)
    return token

def preprocess_text(text: str) -> List[str]:
    """
    Preprocesses text by replacing newlines with spaces and splitting into sentences.

    Parameters:
    - text (str): The input text to preprocess.

    Returns:
    - List[str]: A list of sentences.
    """
    text = re.sub(r"[\n\r]+", " ", text)
    sentences = re.split(r"(?<=[.!?]) +", text)
    return sentences

def tokenize(text: str) -> List[str]:
    """
    Tokenizes a sentence into words by splitting on spaces.

    Parameters:
    - text (str): The input sentence to tokenize.

    Returns:
    - List[str]: A list of tokens (words).
    """
    return text.split()

def find_scopes(tokens: List[str], trigger_words: List[str], label: str) -> List[Tuple[int, int, str]]:
    """
    Identifies triggers and their scopes in tokenized text.

    Parameters:
    - tokens (List[str]): The tokenized text.
    - trigger_words (List[str]): A list of trigger words to search for.
    - label (str): The label for the trigger (e.g., "NEG" or "UNC").

    Returns:
    - List[Tuple[int, int, str]]: A list of tuples representing trigger positions and their scopes.
    """
    scopes = []
    norm_tokens = [normalize_token(tok) for tok in tokens]
    for i in range(len(norm_tokens)):
        for trig in trigger_words:
            trig_parts = [normalize_token(p) for p in trig.split()]
            if norm_tokens[i:i+len(trig_parts)] == trig_parts:
                start = max(0, i - WINDOW_SIZE)
                end = min(len(tokens), i + len(trig_parts) + WINDOW_SIZE)
                scopes.append((i, label))
                scope_label = "NSCO" if label == "NEG" else "USCO"
                scopes.append(((start, end), scope_label))
    return scopes

def apply_rules(text: str) -> List[Dict]:
    """
    Applies the rule-based system to detect negation and uncertainty in text.

    Parameters:
    - text (str): The input text to process.

    Returns:
    - List[Dict]: A list of annotations with detected triggers and scopes.
    """
    annotations = []
    sentences = preprocess_text(text)
    current_offset = 0

    for sent in sentences:
        tokens = tokenize(sent)
        for idx, label in find_scopes(tokens, NEG_TRIGGERS, "NEG") + find_scopes(tokens, UNC_TRIGGERS, "UNC"):
            if isinstance(idx, tuple):
                start_token, end_token = idx
                start_char = len(" ".join(tokens[:start_token]))
                end_char = len(" ".join(tokens[:end_token]))
            else:
                start_char = len(" ".join(tokens[:idx]))
                end_char = start_char + len(tokens[idx])
            annotations.append({
                "value": {
                    "start": current_offset + start_char,
                    "end": current_offset + end_char,
                    "labels": [label]
                }
            })
        current_offset += len(sent) + 1

    return annotations

def run_on_dataset(input_file: str, output_file: str):
    """
    Processes a dataset and applies the rule-based system to detect negation and uncertainty.

    Parameters:
    - input_file (str): Path to the input JSON file containing the dataset.
    - output_file (str): Path to the output JSON file to save the results.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for sample in data:
        text = sample["data"]["text"]
        predictions = apply_rules(text)
        sample["predictions"] = [{"result": predictions}]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Run the rule-based system on the specified dataset
run_on_dataset("../resources/negacio_test_v2024.json", "../resources/test_pred_rules.json")