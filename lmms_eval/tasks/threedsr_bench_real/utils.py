import json
import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset, DatasetDict
from PIL import Image


eval_logger = logging.getLogger("lmms-eval")

THREEDSR_TYPE_TO_SUBTYPES: Dict[str, List[str]] = {
    "location": [
        "location_above",
        "location_closer_to_camera",
        "location_next_to",
    ],
    "height": ["height_higher"],
    "orientation": [
        "orientation_in_front_of",
        "orientation_on_the_left",
        "orientation_viewpoint",
    ],
    "multi_object": [
        "multi_object_closer_to",
        "multi_object_facing",
        "multi_object_viewpoint_towards_object",
        "multi_object_parallel",
        "multi_object_same_direction",
    ],
}

THREEDSR_SUBTYPES = [sub for subs in THREEDSR_TYPE_TO_SUBTYPES.values() for sub in subs]
THREEDSR_METRIC_KEYS = ["accuracy__OVERALL"]
THREEDSR_METRIC_KEYS.extend([f"accuracy__{type_name}" for type_name in THREEDSR_TYPE_TO_SUBTYPES])
THREEDSR_METRIC_KEYS.extend([f"accuracy__{subtype}" for subtype in THREEDSR_SUBTYPES])

_DEFAULT_POST_PROMPT = "\nAnswer with the option letter (A, B, C, or D)."


def load_threedsr_bench_dataset(dataset_path: str, dataset_kwargs: Dict[str, Any]) -> DatasetDict:
    """Load the locally mirrored 3DSR-Bench metadata into a DatasetDict."""

    base_path = Path(dataset_kwargs.pop("data_root", dataset_path))
    metadata_file = dataset_kwargs.pop("metadata_file", "metadata-full.json")
    split_name = dataset_kwargs.pop("split_name", "test")

    metadata_path = Path(metadata_file)
    if not metadata_path.is_absolute():
        metadata_path = base_path / metadata_file

    if not metadata_path.exists():
        raise FileNotFoundError(f"3DSR-Bench metadata file not found at {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as fp:
        raw_payload = json.load(fp)

    if isinstance(raw_payload, dict):
        records = list(raw_payload.values())
    elif isinstance(raw_payload, list):
        records = raw_payload
    else:
        raise TypeError("3DSR-Bench metadata must be a JSON object or array.")

    examples: List[Dict[str, Any]] = []
    for example in sorted(records, key=lambda item: str(item.get("example_id", ""))):
        example_id = str(example.get("example_id") or example.get("id") or "")
        question = str(example.get("question", ""))
        choices = example.get("choices") or []
        labelled_choices = {}
        filtered_choices: List[str] = []
        for idx in range(4):
            choice_value = choices[idx] if idx < len(choices) else None
            if choice_value is not None:
                choice_str = str(choice_value)
                filtered_choices.append(choice_str)
            else:
                choice_str = None
            labelled_choices[chr(ord("A") + idx)] = choice_str

        answer_letter = str(example.get("answer", "")).strip().upper()

        image_ref = example.get("img_path") or example.get("image_path") or example.get("image")
        if image_ref:
            image_path = Path(str(image_ref))
            if not image_path.is_absolute():
                image_path = base_path / image_path
            image_path = image_path.resolve()
        else:
            image_path = None

        rebuilt = {
            "index": example_id,
            "example_id": example_id,
            "question": question,
            "A": labelled_choices.get("A"),
            "B": labelled_choices.get("B"),
            "C": labelled_choices.get("C"),
            "D": labelled_choices.get("D"),
            "answer": answer_letter,
            "answer_choices": filtered_choices,
            "choices": filtered_choices,
            "category": example.get("category") or example.get("type"),
            "image": str(image_path) if image_path else None,
            "image_path": str(image_path) if image_path else None,
            "img": str(image_path) if image_path else None,
            "image_source": example.get("image_source"),
            "image_url": example.get("image_url"),
        }

        examples.append(rebuilt)

    dataset = Dataset.from_list(examples)
    return DatasetDict({split_name: dataset})


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


def threedsr_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    candidate = doc.get("image") or doc.get("image_path") or doc.get("img")
    if candidate is None:
        raise KeyError("3DSR doc missing image reference")
    try:
        return [_load_image(candidate)]
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to load 3DSR image for example {doc.get('index')}" ) from exc


def threedsr_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    question = doc.get("question")
    if question is None:
        raise KeyError("3DSR doc missing `question`")
    options = [
        doc.get("A"),
        doc.get("B"),
        doc.get("C"),
        doc.get("D"),
    ]
    choice_lines = [f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(options) if choice is not None]
    pre_prompt = "Identify the best option for the 3D spatial reasoning question." if not lmms_eval_specific_kwargs else lmms_eval_specific_kwargs.get("pre_prompt", "Identify the best option for the 3D spatial reasoning question.")
    post_prompt = _DEFAULT_POST_PROMPT if not lmms_eval_specific_kwargs else lmms_eval_specific_kwargs.get("post_prompt", _DEFAULT_POST_PROMPT)
    return f"{pre_prompt}\nQuestion: {question}\n" + "\n".join(choice_lines) + post_prompt


def threedsr_doc_to_target(doc: Dict[str, Any]) -> str:
    answer = doc.get("answer")
    if not isinstance(answer, str):
        raise ValueError("3DSR answer must be a string letter")
    return answer.strip().upper()


def _parse_choice_letter(text: str) -> str:
    match = re.search(r"([A-D])", text.upper())
    if match:
        return match.group(1)
    return ""


def threedsr_process_results(doc: Dict[str, Any], results: List[str]):
    prediction_text = results[0] if results else ""
    predicted = _parse_choice_letter(prediction_text)
    if not predicted:
        predicted = "A"

    target = threedsr_doc_to_target(doc)
    is_correct = int(predicted == target)
    example_id = str(doc.get("index") or doc.get("example_id") or doc.get("id"))
    base_question = example_id.split("-")[0]
    payload = {
        "example_id": example_id,
        "base_question": base_question,
        "subtype": str(doc.get("category") or doc.get("type")),
        "is_correct": is_correct,
    }
    return {metric: payload for metric in THREEDSR_METRIC_KEYS}


def _aggregate_by_question(results: List[Dict[str, Any]]):
    aggregated: Dict[str, Dict[str, Any]] = {}
    for item in results:
        base_question = item["base_question"]
        if base_question not in aggregated:
            aggregated[base_question] = {
                "score": 1,
                "subtype": item["subtype"],
            }
        aggregated[base_question]["score"] *= int(item["is_correct"])
    return aggregated


def threedsr_accuracy_overall(results: List[Dict[str, Any]]) -> float:
    aggregated = _aggregate_by_question(results)
    if not aggregated:
        return 0.0
    return float(np.mean([entry["score"] for entry in aggregated.values()]))


def _scores_for_subtypes(results: List[Dict[str, Any]], subtypes: List[str]) -> float:
    aggregated = _aggregate_by_question(results)
    scores = [entry["score"] for entry in aggregated.values() if entry["subtype"] in subtypes]
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _scores_for_subtype(results: List[Dict[str, Any]], subtype: str) -> float:
    return _scores_for_subtypes(results, [subtype])


def threedsr_accuracy_location(results: List[Dict[str, Any]]) -> float:
    return _scores_for_subtypes(results, THREEDSR_TYPE_TO_SUBTYPES["location"])


def threedsr_accuracy_height(results: List[Dict[str, Any]]) -> float:
    return _scores_for_subtypes(results, THREEDSR_TYPE_TO_SUBTYPES["height"])


def threedsr_accuracy_orientation(results: List[Dict[str, Any]]) -> float:
    return _scores_for_subtypes(results, THREEDSR_TYPE_TO_SUBTYPES["orientation"])


def threedsr_accuracy_multi_object(results: List[Dict[str, Any]]) -> float:
    return _scores_for_subtypes(results, THREEDSR_TYPE_TO_SUBTYPES["multi_object"])


def threedsr_accuracy_location_above(results: List[Dict[str, Any]]) -> float:
    return _scores_for_subtype(results, "location_above")


def threedsr_accuracy_location_closer_to_camera(results: List[Dict[str, Any]]) -> float:
    return _scores_for_subtype(results, "location_closer_to_camera")


def threedsr_accuracy_location_next_to(results: List[Dict[str, Any]]) -> float:
    return _scores_for_subtype(results, "location_next_to")


def threedsr_accuracy_height_higher(results: List[Dict[str, Any]]) -> float:
    return _scores_for_subtype(results, "height_higher")


def threedsr_accuracy_orientation_in_front_of(results: List[Dict[str, Any]]) -> float:
    return _scores_for_subtype(results, "orientation_in_front_of")


def threedsr_accuracy_orientation_on_the_left(results: List[Dict[str, Any]]) -> float:
    return _scores_for_subtype(results, "orientation_on_the_left")


def threedsr_accuracy_orientation_viewpoint(results: List[Dict[str, Any]]) -> float:
    return _scores_for_subtype(results, "orientation_viewpoint")


def threedsr_accuracy_multi_object_closer_to(results: List[Dict[str, Any]]) -> float:
    return _scores_for_subtype(results, "multi_object_closer_to")


def threedsr_accuracy_multi_object_facing(results: List[Dict[str, Any]]) -> float:
    return _scores_for_subtype(results, "multi_object_facing")


def threedsr_accuracy_multi_object_viewpoint_towards_object(results: List[Dict[str, Any]]) -> float:
    return _scores_for_subtype(results, "multi_object_viewpoint_towards_object")


def threedsr_accuracy_multi_object_parallel(results: List[Dict[str, Any]]) -> float:
    return _scores_for_subtype(results, "multi_object_parallel")


def threedsr_accuracy_multi_object_same_direction(results: List[Dict[str, Any]]) -> float:
    return _scores_for_subtype(results, "multi_object_same_direction")
