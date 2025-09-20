import io
import logging
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter

eval_logger = logging.getLogger("lmms-eval")


def msr_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def msr_doc_to_visual(doc):
    # image_list = [image.convert("RGB") for image in doc["images"]]
    image_list = []
    for img_data in doc["images"]:
        image = Image.open(io.BytesIO(img_data))
        image = image.convert("RGB")
        image_list.append(image)
    return image_list


def extract_single_choice_with_word_boundary(pred, gt):
    pattern_1 = r"``([^`]*)``"
    match = re.search(pattern_1, pred)
    if match:
        pred = match.group(1)

    pattern_2 = r"`([^`]*)`"
    match = re.search(pattern_2, pred)
    if match:
        pred = match.group(1)

    pattern_add = r"\{([^}]*)\}"
    match = re.search(pattern_add, pred)
    if match:
        pred = match.group(1)

    pattern_3 = r"\b[A-D]\b(?!\s[a-zA-Z])"
    match = re.search(pattern_3, pred)
    if match:
        pred = match.group()
    else:
        return None

    answer = gt.lower().replace("\n", " ").strip()
    predict = pred.lower().replace("\n", " ").strip()
    try:
        if answer == predict[0]:
            return 1.0
        elif predict[0] == "(" and answer == predict[1]:
            return 1.0
        elif predict[0:7] == "option " and answer == predict[7]:
            return 1.0
        elif predict[0:14] == "the answer is " and answer == predict[14]:
            return 1.0
    except Exception as e:
        return 0.0
    return 0.0


def msr_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = results[0]
    gt = doc["answer"]

    score = extract_single_choice_with_word_boundary(pred, gt)
    category = doc["question_type"]
    l2_category = doc["question_type"]
    if score is None:
        return {category: {"question_id": doc["id"], "l2_category": l2_category, "score": 0, "note": "can not find anwser"}, "average": {"question_id": doc["id"], "l2_category": l2_category, "score": 0, "note": "can not find anwser"}}
    return {category: {"question_id": doc["id"], "l2_category": l2_category, "score": score}, "average": {"question_id": doc["id"], "l2_category": l2_category, "score": score}}


def msr_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    l2_category_scores = defaultdict(list)
    for result in results:
        score = result["score"]
        l2_category = result["l2_category"]
        l2_category_scores[l2_category].append(score)

    l2_category_avg_score = {}
    for l2_category, scores in l2_category_scores.items():
        avg_score = sum(scores) / len(scores)
        l2_category_avg_score[l2_category] = avg_score
        eval_logger.info(f"{l2_category}: {avg_score:.2f}")

    all_scores = [score for scores in l2_category_scores.values() for score in scores]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return avg_score

# === MMSI-Bench (multi-image social reasoning) ===

def _load_image_sequence(images):
    image_list = []
    for img in images:
        if isinstance(img, Image.Image):
            image_list.append(img.convert("RGB"))
        elif isinstance(img, bytes):
            image_list.append(Image.open(io.BytesIO(img)).convert("RGB"))
        elif isinstance(img, str):
            image_list.append(Image.open(img).convert("RGB"))
        elif isinstance(img, dict):
            if "path" in img:
                image_list.append(Image.open(img["path"]).convert("RGB"))
            elif "bytes" in img:
                image_list.append(Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
        else:
            raise TypeError(f"Unsupported image payload type for MMSI-Bench: {type(img)}")
    return image_list


def mmsi_doc_to_visual(doc):
    images = doc.get("images") or doc.get("image_sequence")
    if not images:
        raise KeyError("MMSI-Bench sample missing `images`")
    return _load_image_sequence(images)


def mmsi_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc.get("question")
    if question is None:
        raise KeyError("MMSI-Bench sample missing `question`")
    thought = doc.get("thought")
    prefix = "Reason about the social situation depicted in the images." if not lmms_eval_specific_kwargs else lmms_eval_specific_kwargs.get("pre_prompt", "Reason about the social situation depicted in the images.")
    suffix = "\nProvide a concise answer." if not lmms_eval_specific_kwargs else lmms_eval_specific_kwargs.get("post_prompt", "\nProvide a concise answer.")
    if thought:
        return f"{prefix}\nContext: {thought}\nQuestion: {question}{suffix}"
    return f"{prefix}\nQuestion: {question}{suffix}"


def mmsi_doc_to_target(doc):
    answer = doc.get("answer") or doc.get("label")
    if answer is None:
        raise KeyError("MMSI-Bench sample missing `answer`")
    return str(answer)


def mmsi_process_results(doc, results):
    prediction = results[0].strip() if results else ""
    payload = {
        "question_id": doc.get("id") or doc.get("question_id"),
        "question_type": doc.get("question_type", "unknown"),
        "prediction": prediction,
        "answer": mmsi_doc_to_target(doc),
    }
    return {
        "accuracy__MMSI-Overall": payload,
        "accuracy__MMSI-MacroPerType": payload,
    }


def _normalise_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def mmsi_aggregate_overall(results):
    if not results:
        return 0.0
    correct = 0
    for item in results:
        if _normalise_answer(item["prediction"]) == _normalise_answer(item["answer"]):
            correct += 1
    return correct / len(results)


def mmsi_aggregate_macro_per_type(results):
    if not results:
        return 0.0
    buckets = defaultdict(list)
    for item in results:
        key = item.get("question_type", "unknown")
        is_correct = int(_normalise_answer(item["prediction"]) == _normalise_answer(item["answer"]))
        buckets[key].append(is_correct)
    per_type = {key: (sum(values) / len(values)) if values else 0.0 for key, values in buckets.items()}
    for key, score in sorted(per_type.items()):
        eval_logger.info(f"MMSI-Bench accuracy for `{key}`: {score:.3f}")
    return float(np.mean(list(per_type.values()))) if per_type else 0.0
