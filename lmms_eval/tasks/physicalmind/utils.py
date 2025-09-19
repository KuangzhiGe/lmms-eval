import pandas as pd
import os
import sys

def doc_to_text(doc):
    """
    定义了输入给模型的文本部分。
    """
    question_type = doc.get('question_type', 'unknown')
    question = doc.get('question', '')
    if question_type == 'single_choice':
        suffix = " Please choose the most appropriate answer from the options. Answer with the option letter only."
    elif question_type == 'multiple_choice':
        suffix = " Please choose all the appropriate answers from the options. Answer with the option letters only."
    elif question_type == "fill_in_the_blank":
        suffix = " Please fill in the blank with the most appropriate word or phrase. Answer with a single word or phrase only."
    elif question_type == "open_end_questions":
        suffix = " Please provide a detailed answer."
    return doc['question'] + suffix

def physicalmind_doc_to_visual(doc):
    cache_dir = "/mnt/world_foundational_model/gkz/data/physicalmind-bench"
    video_path = doc["video_path"]
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]

def doc_to_target(doc):
    """
    定义了参考答案。
    """
    return doc['answer']

def process_results(doc, results):
    """
    处理模型输出并定义最终保存的格式。

    Args:
        doc: .csv 文件中的一行数据。
        results: 模型生成的文本输出。

    Returns:
        一个包含各类信息的字典。
    """
    return {
        "experiment_name": doc.get("experiment_name"),
        "question_type": doc.get("question_type"),
        "video_path": doc.get("video_path"),
        "question": doc.get("question"),
        "answer": doc.get("answer"),
        "model_output": results[0],
    }