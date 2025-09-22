import json
import logging
import math
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score


eval_logger = logging.getLogger("lmms-eval")

_DEFAULT_PRE_PROMPT = "You will be shown an image. Decide whether the statement is true or false."
_DEFAULT_POST_PROMPT = "\nAnswer with either `True` or `False`."


def load_vsr_dataset(dataset_path: str, dataset_kwargs: Dict[str, Any]) -> DatasetDict:
    """Load the locally mirrored VSR metadata into a DatasetDict."""

    params = dict(dataset_kwargs)

    base_path = Path(params.pop("data_root", dataset_path))
    metadata_file = params.pop("metadata_file", "metadata-full.json")
    split_name = params.pop("split_name", "test")

    metadata_path = Path(metadata_file)
    if not metadata_path.is_absolute():
        metadata_path = base_path / metadata_file

    if not metadata_path.exists():
        raise FileNotFoundError(f"VSR metadata file not found at {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as fp:
        raw_payload = json.load(fp)

    if isinstance(raw_payload, dict):
        records = list(raw_payload.values())
    elif isinstance(raw_payload, list):
        records = raw_payload
    else:
        raise TypeError("VSR metadata must be a JSON object or array.")

    examples: List[Dict[str, Any]] = []
    for example in sorted(records, key=lambda item: str(item.get("example_id", ""))):
        example_id = example.get("example_id")
        example_id = str(example_id)

        image_ref = example.get("img_path") or example.get("image_path") or example.get("image")
        if image_ref:
            image_path = Path(str(image_ref))
            if not image_path.is_absolute():
                image_path = base_path / image_path
            image_path = image_path.resolve()
        else:
            raise ValueError(f"VSR sample {example_id} missing image path")

        label = example.get("true_false")
        if isinstance(label, str):
            lowered = label.lower()
            if lowered in {"true", "t", "1", "yes"}:
                label = True
            elif lowered in {"false", "f", "0", "no"}:
                label = False
        else:
            label = bool(label)

        path_str = str(image_path)

        rebuilt = {
            "example_id": example_id,
            "id": example_id,
            "caption": str(example.get("caption", "")),
            "statement": str(example.get("caption", "")),
            "subj": str(example.get("subj") or ""),
            "obj": str(example.get("obj") or ""),
            "relation": str(example.get("relation") or ""),
            "label": bool(label),
            "image": path_str,
            "img": path_str,
            "image_path": path_str,
        }

        examples.append(rebuilt)

    features = Features(
        {
            "example_id": Value("string"),
            "id": Value("string"),
            "caption": Value("string"),
            "statement": Value("string"),
            "subj": Value("string"),
            "obj": Value("string"),
            "relation": Value("string"),
            "label": Value("bool"),
            "image": Value("string"),
            "img": Value("string"),
            "image_path": Value("string"),
        }
    )

    dataset = Dataset.from_list(examples, features=features)
    return DatasetDict({split_name: dataset})


def _load_image(image_like: Any) -> Image.Image:
    """Best-effort conversion of an arbitrary image reference to a RGB PIL.Image."""
    if isinstance(image_like, Image.Image):
        return image_like.convert("RGB")
    if isinstance(image_like, bytes):
        return Image.open(BytesIO(image_like)).convert("RGB")
    if isinstance(image_like, str):
        return Image.open(image_like).convert("RGB")
    if isinstance(image_like, dict):
        if "path" in image_like:
            return Image.open(image_like["path"]).convert("RGB")
        if "bytes" in image_like:
            return Image.open(BytesIO(image_like["bytes"])).convert("RGB")
    raise TypeError(f"Unsupported image payload type: {type(image_like)}")


def vsr_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    image = doc.get("image") or doc.get("img") or doc.get("image_path")
    if image is None:
        raise KeyError("VSR doc is missing the `image` field")
    try:
        return [_load_image(image)]
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to load VSR image for sample {doc.get('id', '<unknown>')}") from exc


def vsr_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, str]] = None) -> str:
    caption = doc.get("caption") or doc.get("statement")
    if caption is None:
        raise KeyError("VSR doc missing `caption`")

    pre_prompt = _DEFAULT_PRE_PROMPT
    post_prompt = _DEFAULT_POST_PROMPT
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", pre_prompt)
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", post_prompt)

    return f"{pre_prompt}\nStatement: {caption}{post_prompt}"


def vsr_doc_to_target(doc: Dict[str, Any]) -> str:
    label = doc.get("label")
    if isinstance(label, bool):
        return "True" if label else "False"
    if isinstance(label, (int, np.integer)):
        return "True" if int(label) == 1 else "False"
    if isinstance(label, str):
        lowered = label.lower()
        if lowered in {"true", "t", "1", "yes"}:
            return "True"
        if lowered in {"false", "f", "0", "no"}:
            return "False"
    raise ValueError(f"Unsupported VSR label type: {type(label)}")


def _parse_boolean_prediction(text: str) -> Optional[bool]:
    match = re.search(r"(true|false)", text.lower())
    if not match:
        return None
    return match.group(1) == "true"


def _extract_confidence(text: str) -> Optional[float]:
    numbers = [float(token) for token in re.findall(r"(?<!\w)(?:1(?:\.0+)?|0?\.\d+)(?!\w)", text)]
    if not numbers:
        return None
    # Heuristic: use last probability-like number
    prob = numbers[-1]
    if 0.0 <= prob <= 1.0:
        return prob
    return None


def _ensure_probability(prob: Optional[float], fallback_bool: bool) -> float:
    if prob is not None and not math.isnan(prob):
        return float(max(0.0, min(1.0, prob)))
    return 1.0 if fallback_bool else 0.0


def _label_to_bool(label: str) -> bool:
    return label.lower() == "true"


def vsr_process_results(doc: Dict[str, Any], results: List[str]):
    prediction_text = results[0] if results else ""
    predicted_bool = _parse_boolean_prediction(prediction_text)

    label_bool = _label_to_bool(vsr_doc_to_target(doc))

    prob_true = _extract_confidence(prediction_text)
    if predicted_bool is None:
        predicted_bool = prob_true is not None and prob_true >= 0.5
    prob_true = _ensure_probability(prob_true, predicted_bool)

    payload = {
        "question_id": doc.get("id") or doc.get("question_id") or doc.get("index"),
        "prediction": bool(predicted_bool),
        "label": int(label_bool),
        "prob_true": float(prob_true),
    }

    return {
        "VSR-ExactMatch": payload,
        "VSR-AUCROC": payload,
        "VSR-AUCPR": payload,
        "VSR-AlwaysTrue": payload,
    }


def _gather_scores(results: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if not results:
        return {"labels": np.array([], dtype=float), "scores": np.array([], dtype=float)}
    labels = np.array([item["label"] for item in results], dtype=float)
    scores = np.array([item["prob_true"] for item in results], dtype=float)
    return {"labels": labels, "scores": scores}


def vsr_aggregate_accuracy(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    correct = sum(int(item["prediction"] == item["label"]) for item in results)
    return correct / len(results)


def vsr_aggregate_auc_roc(results: List[Dict[str, Any]]) -> float:
    stats = _gather_scores(results)
    labels, scores = stats["labels"], stats["scores"]
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def vsr_aggregate_auc_pr(results: List[Dict[str, Any]]) -> float:
    stats = _gather_scores(results)
    labels, scores = stats["labels"], stats["scores"]
    if len(labels) == 0:
        return 0.0
    if len(np.unique(labels)) < 2:
        # average_precision_score expects both classes; degrade gracefully
        return float(sum(labels) / len(labels))
    return float(average_precision_score(labels, scores))


def vsr_aggregate_always_true(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    positive = sum(item["label"] for item in results)
    return positive / len(results)
