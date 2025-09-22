import io
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from PIL import Image


eval_logger = logging.getLogger("lmms-eval")


def load_mindcube_dataset(dataset_path: str, dataset_kwargs: Dict[str, Any]) -> DatasetDict:
    """Load the locally mirrored MindCube metadata into a DatasetDict."""

    params = dict(dataset_kwargs)

    base_path = Path(params.pop("data_root", dataset_path))
    metadata_file = params.pop("metadata_file", "metadata-full.json")
    metadata_path = Path(metadata_file)
    if not metadata_path.is_absolute():
        metadata_path = base_path / metadata_file

    if not metadata_path.exists():
        raise FileNotFoundError(f"MindCube metadata file not found at {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as fp:
        raw_payload = json.load(fp)

    if isinstance(raw_payload, dict):
        records = list(raw_payload.values())
    elif isinstance(raw_payload, list):
        records = raw_payload
    else:
        raise TypeError("MindCube metadata must be a JSON object or array.")

    split_key = params.pop("split_key", "split")
    image_key = params.pop("image_key", "img_path")

    def _flatten_meta(meta_obj):
        if meta_obj is None:
            return []
        if isinstance(meta_obj, list):
            flattened = []
            for item in meta_obj:
                flattened.extend(_flatten_meta(item))
            return flattened
        return [str(meta_obj)]

    split_to_examples: Dict[str, List[Dict[str, Any]]] = {}
    for example in sorted(records, key=lambda item: str(item.get("example_id", ""))):
        example_id = str(example.get("example_id") or example.get("id") or "")
        split = (example.get(split_key) or "test").strip() or "test"

        image_entries = example.get(image_key) or []
        resolved_images: List[str] = []
        for image_path in image_entries:
            candidate = Path(str(image_path))
            if not candidate.is_absolute():
                candidate = base_path / candidate
            resolved_images.append(str(candidate))

        category = example.get("category")
        if isinstance(category, list):
            category_values = [str(item) for item in category if item is not None]
        else:
            category_values = [str(category)] if category is not None else []

        meta_info = example.get("meta_info")
        flattened_meta = _flatten_meta(meta_info)
        meta_str = "; ".join(item for item in flattened_meta if item)

        type_value = str(example.get("type") or "")

        rebuilt = {
            "example_id": example_id,
            "id": example_id,
            "question": str(example.get("question", "")),
            "answer": str(example.get("answer") or example.get("gt_answer") or ""),
            "images": resolved_images,
            "image_paths": resolved_images,
            "frames": resolved_images,
            "type": type_value,
            "category": category_values,
            "meta_info": meta_str,
            "split": split,
        }

        split_to_examples.setdefault(split, []).append(rebuilt)

    features = Features(
        {
            "example_id": Value("string"),
            "id": Value("string"),
            "question": Value("string"),
            "answer": Value("string"),
            "images": Sequence(Value("string")),
            "image_paths": Sequence(Value("string")),
            "frames": Sequence(Value("string")),
            "type": Value("string"),
            "category": Sequence(Value("string")),
            "meta_info": Value("string"),
            "split": Value("string"),
        }
    )

    dataset_dict = {}
    for split_name, examples in split_to_examples.items():
        dataset_dict[split_name] = Dataset.from_list(examples, features=features)

    return DatasetDict(dataset_dict)

def _load_image_sequence(images: List[Any]) -> List[Image.Image]:
    frames: List[Image.Image] = []
    for image in images:
        if isinstance(image, Image.Image):
            frames.append(image.convert("RGB"))
        elif isinstance(image, bytes):
            frames.append(Image.open(io.BytesIO(image)).convert("RGB"))
        elif isinstance(image, str):
            frames.append(Image.open(image).convert("RGB"))
        elif isinstance(image, dict):
            if "path" in image:
                frames.append(Image.open(image["path"]).convert("RGB"))
            elif "bytes" in image:
                frames.append(Image.open(io.BytesIO(image["bytes"])).convert("RGB"))
        else:
            raise TypeError(f"Unsupported image payload type for MindCube: {type(image)}")
    return frames


def mindcube_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    images = doc.get("images") or doc.get("image_paths") or doc.get("frames")
    if not images:
        raise KeyError("MindCube sample missing `images`")
    return _load_image_sequence(images)


def mindcube_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    question = doc.get("question")
    if question is None:
        raise KeyError("MindCube sample missing `question`")
    meta = doc.get("meta_info")
    prefix = "Reason about the 3D configuration before answering." if not lmms_eval_specific_kwargs else lmms_eval_specific_kwargs.get("pre_prompt", "Reason about the 3D configuration before answering.")
    suffix = "\nRespond with a concise answer." if not lmms_eval_specific_kwargs else lmms_eval_specific_kwargs.get("post_prompt", "\nRespond with a concise answer.")
    if meta:
        return f"{prefix}\nContext: {meta}\nQuestion: {question}{suffix}"
    return f"{prefix}\nQuestion: {question}{suffix}"


def mindcube_doc_to_target(doc: Dict[str, Any]) -> str:
    answer = doc.get("answer") or doc.get("gt_answer")
    if answer is None:
        raise KeyError("MindCube sample missing `answer`")
    return str(answer)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _normalise_type(question_type: Any) -> str:
    if isinstance(question_type, list):
        return "|".join(str(part) for part in question_type)
    return str(question_type)


def mindcube_process_results(doc: Dict[str, Any], results: List[str]):
    prediction = results[0].strip() if results else ""
    payload = {
        "example_id": doc.get("id") or doc.get("example_id"),
        "question_type": _normalise_type(doc.get("type", "unknown")),
        "prediction": prediction,
        "answer": mindcube_doc_to_target(doc),
    }
    return {
        "accuracy__MindCube-Overall": payload,
        "accuracy__MindCube-MacroPerType": payload,
    }


def mindcube_aggregate_overall(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    correct = sum(int(_normalise(item["prediction"]) == _normalise(item["answer"])) for item in results)
    return correct / len(results)


def mindcube_aggregate_macro_per_type(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    buckets: Dict[str, List[int]] = {}
    for item in results:
        key = item.get("question_type", "unknown")
        buckets.setdefault(key, []).append(int(_normalise(item["prediction"]) == _normalise(item["answer"])))
    per_type_scores = {key: (sum(values) / len(values)) if values else 0.0 for key, values in buckets.items()}
    for key, score in sorted(per_type_scores.items()):
        eval_logger.info(f"MindCube accuracy for `{key}`: {score:.3f}")
    return float(np.mean(list(per_type_scores.values()))) if per_type_scores else 0.0
