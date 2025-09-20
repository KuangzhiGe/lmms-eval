import io
import logging
import re
from typing import Any, Dict, List

import numpy as np
from PIL import Image


eval_logger = logging.getLogger("lmms-eval")


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
