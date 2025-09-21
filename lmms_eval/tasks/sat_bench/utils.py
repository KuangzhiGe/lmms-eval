import io
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset, DatasetDict
from PIL import Image


eval_logger = logging.getLogger("lmms-eval")


def load_sat_bench_dataset(dataset_path: str, dataset_kwargs: Dict[str, Any]) -> DatasetDict:
    """Load the locally mirrored SAT-Bench metadata into a DatasetDict."""

    base_path = Path(dataset_kwargs.pop("data_root", dataset_path))
    metadata_file = dataset_kwargs.pop("metadata_file", "metadata-full.json")
    metadata_path = Path(metadata_file)
    if not metadata_path.is_absolute():
        metadata_path = base_path / metadata_file

    if not metadata_path.exists():
        raise FileNotFoundError(f"SAT-Bench metadata file not found at {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as fp:
        raw_payload = json.load(fp)

    if isinstance(raw_payload, dict):
        records = list(raw_payload.values())
    elif isinstance(raw_payload, list):
        records = raw_payload
    else:
        raise TypeError("SAT-Bench metadata must be a JSON object or array.")

    split_key = dataset_kwargs.pop("split_key", "split")
    image_key = dataset_kwargs.pop("image_key", "img_path")

    split_to_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for example in sorted(records, key=lambda item: item.get("example_id", 0)):
        example_split = example.get(split_key, "val") or "val"

        image_entries = example.get(image_key) or []
        resolved_images: List[str] = []
        for image_path in image_entries:
            image_path = str(image_path)
            candidate = Path(image_path)
            if not candidate.is_absolute():
                candidate = base_path / image_path
            resolved_images.append(str(candidate))

        rebuilt = {
            "example_id": int(example.get("example_id", 0)),
            "id": str(example.get("example_id", "")),
            "question": example.get("question", ""),
            "answer_choices": list(example.get("choices", [])),
            "choices": list(example.get("choices", [])),
            "correct_answer": str(example.get("correct_answer", "")),
            "answer": str(example.get("correct_answer", "")),
            "question_type": example.get("question_type", "unknown"),
            "images": resolved_images,
            "img_path": resolved_images,
            "split": example_split,
            "dataset": example.get("dataset", "SAT"),
        }

        split_to_examples[example_split].append(rebuilt)

    dataset_dict = {
        split_name: Dataset.from_list(examples)
        for split_name, examples in split_to_examples.items()
    }

    return DatasetDict(dataset_dict)


def _load_image_sequence(images: List[Any]) -> List[Image.Image]:
    image_list: List[Image.Image] = []
    for image in images:
        if isinstance(image, Image.Image):
            image_list.append(image.convert("RGB"))
        elif isinstance(image, bytes):
            image_list.append(Image.open(io.BytesIO(image)).convert("RGB"))
        elif isinstance(image, str):
            image_list.append(Image.open(image).convert("RGB"))
        elif isinstance(image, dict):
            if "path" in image:
                image_list.append(Image.open(image["path"]).convert("RGB"))
            elif "bytes" in image:
                image_list.append(Image.open(io.BytesIO(image["bytes"])).convert("RGB"))
        else:
            raise TypeError(f"Unsupported image payload type for SAT-Bench: {type(image)}")
    return image_list


def sat_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    images = doc.get("images") or doc.get("image_sequence") or doc.get("frames")
    if not images:
        raise KeyError("SAT sample missing `images`")
    return _load_image_sequence(images)


def sat_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    question = doc.get("question")
    if question is None:
        raise KeyError("SAT sample missing `question`")
    answer_choices = doc.get("answer_choices") or doc.get("choices") or []
    choice_lines = []
    for idx, choice in enumerate(answer_choices):
        option = chr(ord("A") + idx)
        choice_lines.append(f"{option}. {choice}")
    pre_prompt = "Reason step by step about the sequence of frames before answering." if not lmms_eval_specific_kwargs else lmms_eval_specific_kwargs.get("pre_prompt", "Reason step by step about the sequence of frames before answering.")
    post_prompt = "\nRespond with the full answer text." if not lmms_eval_specific_kwargs else lmms_eval_specific_kwargs.get("post_prompt", "\nRespond with the full answer text.")
    body = "\n".join(choice_lines)
    return f"{pre_prompt}\nQuestion: {question}\n{body}{post_prompt}"


def sat_doc_to_target(doc: Dict[str, Any]) -> str:
    answer = doc.get("answer") or doc.get("correct_answer")
    if answer is None:
        raise KeyError("SAT sample missing `answer`")
    return str(answer)


def _normalise_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def sat_process_results(doc: Dict[str, Any], results: List[str]):
    prediction = results[0].strip() if results else ""
    payload = {
        "example_id": doc.get("example_id") or doc.get("id"),
        "question_type": doc.get("question_type", "unknown"),
        "prediction": prediction,
        "answer": sat_doc_to_target(doc),
    }
    return {
        "accuracy__SAT-Overall": payload,
        "accuracy__SAT-MacroPerType": payload,
    }


def sat_aggregate_overall(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    correct = sum(int(_normalise_answer(item["prediction"]) == _normalise_answer(item["answer"])) for item in results)
    return correct / len(results)


def sat_aggregate_macro_per_type(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    buckets: Dict[str, List[int]] = {}
    for item in results:
        key = item.get("question_type", "unknown")
        buckets.setdefault(key, []).append(int(_normalise_answer(item["prediction"]) == _normalise_answer(item["answer"])))
    per_type_scores = {key: (sum(values) / len(values)) if values else 0.0 for key, values in buckets.items()}
    for key, score in sorted(per_type_scores.items()):
        eval_logger.info(f"SAT-Bench accuracy for `{key}`: {score:.3f}")
    return float(np.mean(list(per_type_scores.values()))) if per_type_scores else 0.0
