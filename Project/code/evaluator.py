import json
from typing import List, Dict, Tuple
from collections import defaultdict

# Function to check if two spans overlap
def spans_overlap(span1, span2):
    return span1[0] < span2[1] and span2[0] < span1[1]

# Main evaluation function
def evaluate(pred_file: str, gold_file: str, gold_key: str = "annotations"):
    """
    Evaluates the performance of predicted spans against gold-standard spans.

    Parameters:
    - pred_file (str): Path to the JSON file containing predicted annotations.
    - gold_file (str): Path to the JSON file containing gold-standard annotations.
    - gold_key (str): Key in the gold-standard JSON file to extract annotations (default: "annotations").
    """

    # Helper function to extract spans from the input data
    def extract_spans(data: List[Dict], key: str) -> Dict[str, List[Tuple[int, int, str]]]:
        """
        Extracts spans from the input data based on the specified key.

        Parameters:
        - data (List[Dict]): List of dictionaries containing annotation data.
        - key (str): Key to extract spans from (e.g., "predictions" or "annotations").

        Returns:
        - Dict[str, List[Tuple[int, int, str]]]: A dictionary mapping document IDs to lists of spans.
          Each span is represented as a tuple (start, end, label).
        """
        span_dict = {}
        for item in data:
            spans = []
            key_list = item.get(key, [])
            if isinstance(key_list, list) and len(key_list) > 0:
                for ann in key_list[0].get("result", []):
                    start = ann["value"]["start"]
                    end = ann["value"]["end"]
                    label = ann["value"]["labels"][0]
                    spans.append((start, end, label))
            span_dict[item["data"]["id"]] = spans
        return span_dict

    # Load predictions and gold-standard annotations from JSON files
    with open(pred_file, "r", encoding="utf-8") as f:
        preds = json.load(f)
    with open(gold_file, "r", encoding="utf-8") as f:
        golds = json.load(f)

    # Extract spans from predictions and gold-standard data
    pred_spans = extract_spans(preds, "predictions")
    gold_spans = extract_spans(golds, gold_key)

    # Initialize metrics for each label
    metrics = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    # Iterate through each document in the gold-standard data
    for sample in golds:
        doc_id = sample["data"]["id"]
        gold_set = gold_spans.get(doc_id, [])
        pred_set = pred_spans.get(doc_id, [])

        matched = set()  # Track matched gold spans
        # Compare predicted spans with gold-standard spans
        for pred in pred_set:
            found_match = False
            for i, gold in enumerate(gold_set):
                # Check if spans overlap and labels match
                if gold[2] == pred[2] and spans_overlap(gold, pred) and i not in matched:
                    metrics[pred[2]]["TP"] += 1  # True Positive
                    matched.add(i)
                    found_match = True
                    break
            if not found_match:
                metrics[pred[2]]["FP"] += 1  # False Positive

        # Count unmatched gold spans as False Negatives
        for i, gold in enumerate(gold_set):
            if i not in matched:
                metrics[gold[2]]["FN"] += 1

    # Calculate and print precision, recall, and F1-score for each label
    for label, counts in metrics.items():
        TP = counts["TP"]
        FP = counts["FP"]
        FN = counts["FN"]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"\nLabel: {label}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall:    {recall:.2f}")
        print(f"  F1-Score:  {f1:.2f}")

# Example usage of the evaluate function
evaluate("../resources/test_pred_rules.json", "../resources/negacio_test_v2024.json", gold_key="predictions")