import json
import re
from typing import List, Dict, Tuple
import spacy

# Load multilingual spaCy model (Catalan & Spanish support)
nlp = spacy.blank("es")  # Use 'xx_ent_wiki_sm' if needed

# --- Cue Lexicons ---
NEGATION_CUES = {"no", "niega", "sin", "ausencia", "descarta", "niegan"}
UNCERTAINTY_CUES = {"posible", "duda", "sugiere", "probable", "podría", "se sospecha"}

# --- Utility Functions ---

def load_json(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_cue_and_scope(text: str, cues: set, label_cue: str, label_scope: str, window: int = 5) -> Tuple[List[Dict], List[Tuple[str, str]]]:
    annotations = []
    debug_info = []
    doc = nlp(text)

    for i, token in enumerate(doc):
        lower_token = token.text.lower()
        if lower_token in cues:
            # --- Cue Annotation ---
            cue_start = token.idx
            cue_end = cue_start + len(token.text)
            annotations.append({
                "start": cue_start,
                "end": cue_end,
                "labels": [label_cue]
            })

            # --- Scope Annotation ---
            scope_tokens = []
            j = i + 1
            while j < len(doc) and len(scope_tokens) < window:
                if doc[j].is_punct:
                    break
                scope_tokens.append(doc[j])
                j += 1

            if scope_tokens:
                scope_start = scope_tokens[0].idx
                scope_end = scope_tokens[-1].idx + len(scope_tokens[-1].text)
                annotations.append({
                    "start": scope_start,
                    "end": scope_end,
                    "labels": [label_scope]
                })

                # Debug output of cue → scope pair
                cue_text = token.text
                scope_text = text[scope_start:scope_end]
                debug_info.append((cue_text, scope_text))

    return annotations, debug_info

def detect_negation_uncertainty(report: Dict) -> Dict:
    text = report["data"]["text"]
    annotations = []

    neg_annots, neg_debug = find_cue_and_scope(text, NEGATION_CUES, "NEG", "NSCO")
    unc_annots, unc_debug = find_cue_and_scope(text, UNCERTAINTY_CUES, "UNC", "USCO")

    annotations.extend(neg_annots)
    annotations.extend(unc_annots)

    # Debug: Show extracted pairs
    if neg_debug or unc_debug:
        print("\n--- Report ---")
        for cue, scope in neg_debug:
            print(f"NEG: '{cue}' → '{scope}'")
        for cue, scope in unc_debug:
            print(f"UNC: '{cue}' → '{scope}'")

    return {
        "data": {"text": text},
        "annotations": annotations,
        "meta": {
            "negation_cues": len(neg_debug),
            "uncertainty_cues": len(unc_debug)
        }
    }

def process_dataset(input_path: str, output_path: str):
    data = load_json(input_path)
    results = [detect_negation_uncertainty(report) for report in data]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# --- Run Example ---
if __name__ == "__main__":
    input_path = "negacio_train_v2024.json"
    output_path = "predictions_train.json"

    process_dataset(input_path, output_path)
