import logging
import re
from io import BytesIO
from typing import Any, Dict, List

import numpy as np
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
