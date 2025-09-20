import logging
import math
import re
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize


eval_logger = logging.getLogger("lmms-eval")

_NUM_CHOICES = 16
_OPTION_STRINGS = [str(i) for i in range(_NUM_CHOICES)]
_DEFAULT_POST_PROMPT = "\nAnswer with the number corresponding to the correct count (0-15)."

_TALLYQA_METRIC_KEYS = [
    "TallyQA-simple-Accuracy",
    "TallyQA-simple-AUCROC",
    "TallyQA-simple-AUCPR",
    "TallyQA-complex-Accuracy",
    "TallyQA-complex-AUCROC",
    "TallyQA-complex-AUCPR",
    "TallyQA-final-Accuracy",
    "TallyQA-final-AUCROC",
    "TallyQA-final-AUCPR",
]


def _load_image(image_like: Any) -> Image.Image:
    if isinstance(image_like, Image.Image):
        return image_like.convert("RGB")
    if isinstance(image_like, bytes):
        return Image.open(BytesIO(image_like)).convert("RGB")
    if isinstance(image_like, str):
        return Image.open(image_like).convert("RGB")
    if isinstance(image_like, dict):
        path = image_like.get("path")
        if path:
            return Image.open(path).convert("RGB")
        data = image_like.get("bytes")
        if data:
            return Image.open(BytesIO(data)).convert("RGB")
    raise TypeError(f"Unsupported image payload type: {type(image_like)}")


def tallyqa_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    # HuggingFace JSON loader stores the relative path under `image` or `image_path`
    candidate = doc.get("image") or doc.get("image_path")
    if candidate is None:
        raise KeyError("TallyQA sample missing image reference")
    try:
        return [_load_image(candidate)]
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to load TallyQA image for question {doc.get('question_id')}" ) from exc


def tallyqa_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, str]] = None) -> str:
    question = doc.get("question") or doc.get("prompt")
    if question is None:
        raise KeyError("TallyQA sample missing `question`")

    post_prompt = _DEFAULT_POST_PROMPT
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "Count the objects referenced in the question.") if lmms_eval_specific_kwargs else "Count the objects referenced in the question."
    if lmms_eval_specific_kwargs and "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    return f"{pre_prompt}\nQuestion: {question}{post_prompt}"


def tallyqa_doc_to_target(doc: Dict[str, Any]) -> str:
    answer = doc.get("answer")
    if isinstance(answer, (int, np.integer)):
        return str(int(answer))
    if isinstance(answer, str) and answer.isdigit():
        return answer
    raise ValueError(f"Unsupported TallyQA answer format: {answer}")


def _parse_prediction(text: str) -> Optional[int]:
    # Prefer standalone numbers between 0 and 15
    match = re.search(r"(?<!\d)(1[0-5]|[0-9])(?!\d)", text)
    if match:
        return int(match.group(1))
    return None


def _extract_probability_vector(results: List[Any], predicted: Optional[int]) -> np.ndarray:
    if len(results) > 1:
        candidate = results[1]
        if isinstance(candidate, dict):
            values = candidate.get("choice_scores") or candidate.get("logits") or candidate.get("probabilities")
        else:
            values = candidate
        if values is not None:
            try:
                scores = np.asarray(list(values.values()) if isinstance(values, dict) else values, dtype=float)
                if scores.ndim == 1 and scores.size == _NUM_CHOICES:
                    # Treat as logits/log-probs and normalise to probabilities
                    scores = scores - np.max(scores)
                    probs = np.exp(scores)
                    denom = probs.sum()
                    if denom > 0:
                        return probs / denom
            except Exception:  # pragma: no cover - defensive
                pass
    # Fallback: one-hot distribution centred on prediction
    probs = np.full(_NUM_CHOICES, 1.0 / _NUM_CHOICES, dtype=float)
    if predicted is not None:
        probs = np.zeros(_NUM_CHOICES, dtype=float)
        probs[predicted] = 1.0
    return probs


def tallyqa_process_results(doc: Dict[str, Any], results: List[Any]):
    response_text = results[0] if results else ""
    predicted = _parse_prediction(response_text)
    if predicted is None:
        # Attempt to fall back to verbal cues (e.g., "five") by mapping words to digits
        word_map = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
        }
        for token, value in word_map.items():
            if token in response_text.lower():
                predicted = value
                break
    if predicted is None:
        predicted = 0

    probabilities = _extract_probability_vector(results, predicted)
    target = int(tallyqa_doc_to_target(doc))
    split = "simple" if doc.get("issimple") or doc.get("split") == "simple" else "complex"

    payload = {
        "question_id": doc.get("question_id") or doc.get("id"),
        "split": split,
        "target": target,
        "prediction": int(predicted),
        "probabilities": probabilities,
    }

    return {metric: payload for metric in _TALLYQA_METRIC_KEYS}


def _filter_by_split(results: List[Dict[str, Any]], split: Optional[str]) -> List[Dict[str, Any]]:
    if split is None:
        return results
    return [item for item in results if item["split"] == split]


def _accuracy(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    correct = sum(int(item["prediction"] == item["target"]) for item in results)
    return correct / len(results)


def _prepare_probabilities(results: List[Dict[str, Any]]):
    if not results:
        return None, None, None
    labels = np.array([item["target"] for item in results], dtype=int)
    probs = np.stack([item["probabilities"] for item in results], axis=0)
    classes = np.unique(labels)
    if len(classes) < 2:
        return labels, probs, classes
    # Restrict to observed classes to avoid numerical issues
    return labels, probs[:, classes], classes


def _roc_auc(results: List[Dict[str, Any]]) -> float:
    labels, probs, classes = _prepare_probabilities(results)
    if labels is None or len(classes) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs, labels=classes, multi_class="ovo"))


def _pr_auc(results: List[Dict[str, Any]]) -> float:
    labels, probs, classes = _prepare_probabilities(results)
    if labels is None or len(classes) < 2:
        return float("nan")
    label_matrix = label_binarize(labels, classes=classes)
    return float(average_precision_score(label_matrix, probs))


def tallyqa_accuracy_simple(results: List[Dict[str, Any]]) -> float:
    return _accuracy(_filter_by_split(results, "simple"))


def tallyqa_accuracy_complex(results: List[Dict[str, Any]]) -> float:
    return _accuracy(_filter_by_split(results, "complex"))


def tallyqa_accuracy_final(results: List[Dict[str, Any]]) -> float:
    return _accuracy(results)


def tallyqa_aucroc_simple(results: List[Dict[str, Any]]) -> float:
    return _roc_auc(_filter_by_split(results, "simple"))


def tallyqa_aucroc_complex(results: List[Dict[str, Any]]) -> float:
    return _roc_auc(_filter_by_split(results, "complex"))


def tallyqa_aucroc_final(results: List[Dict[str, Any]]) -> float:
    return _roc_auc(results)


def tallyqa_aucpr_simple(results: List[Dict[str, Any]]) -> float:
    return _pr_auc(_filter_by_split(results, "simple"))


def tallyqa_aucpr_complex(results: List[Dict[str, Any]]) -> float:
    return _pr_auc(_filter_by_split(results, "complex"))


def tallyqa_aucpr_final(results: List[Dict[str, Any]]) -> float:
    return _pr_auc(results)
