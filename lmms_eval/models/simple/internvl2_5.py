import io
import logging
import math
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs
from decord import VideoReader, cpu
from huggingface_hub import snapshot_download
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoTokenizer

from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INTERNVL_CHAT_ROOT = PROJECT_ROOT / "InternVL" / "internvl_chat"
if INTERNVL_CHAT_ROOT.exists() and str(INTERNVL_CHAT_ROOT) not in sys.path:
    sys.path.insert(0, str(INTERNVL_CHAT_ROOT))

try:
    from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel
except ImportError as exc:  # pragma: no cover - defensive
    raise ImportError(
        "InternVL2.5 requires the official InternVL repository. Please clone it "
        "next to lmms-eval (../InternVL) following the upstream instructions."
    ) from exc


eval_logger = logging.getLogger("eval_logger")


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE = 448
DEFAULT_MAX_TILES = 12
DEFAULT_GEN_KWARGS = dict(max_new_tokens=1024, do_sample=False)


def build_transform(input_size: int) -> T.Compose:
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: Sequence[Tuple[int, int]], width: int, height: int, image_size: int) -> Tuple[int, int]:
    best_ratio = (1, 1)
    best_diff = float("inf")
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_diff:
            best_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = DEFAULT_MAX_TILES, image_size: int = DEFAULT_IMAGE_SIZE, use_thumbnail: bool = False) -> List[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_ratio[0]
    target_height = image_size * target_ratio[1]
    blocks = target_ratio[0] * target_ratio[1]

    resized = image.resize((target_width, target_height))
    tiles: List[Image.Image] = []
    step = target_width // image_size
    for idx in range(blocks):
        box = (
            (idx % step) * image_size,
            (idx // step) * image_size,
            ((idx % step) + 1) * image_size,
            ((idx // step) + 1) * image_size,
        )
        tiles.append(resized.crop(box))

    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def split_model(model_name: str) -> dict:
    world_size = torch.cuda.device_count()
    num_layers = {
        "InternVL2_5-1B": 24,
        "InternVL2_5-2B": 24,
        "InternVL2_5-4B": 36,
        "InternVL2_5-8B": 32,
        "InternVL2_5-26B": 48,
        "InternVL2_5-38B": 64,
        "InternVL2_5-78B": 80,
    }.get(model_name, 32)

    device_map = {}
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    schedule = [num_layers_per_gpu] * world_size
    schedule[0] = math.ceil(schedule[0] * 0.5)

    layer_idx = 0
    for device_id, quota in enumerate(schedule):
        for _ in range(quota):
            device_map[f"language_model.model.layers.{layer_idx}"] = device_id
            layer_idx += 1

    device_map.update(
        {
            "vision_model": 0,
            "mlp1": 0,
            "language_model.model.tok_embeddings": 0,
            "language_model.model.embed_tokens": 0,
            "language_model.output": 0,
            "language_model.model.norm": 0,
            "language_model.lm_head": 0,
            f"language_model.model.layers.{num_layers - 1}": 0,
        }
    )
    return device_map


def _flatten(nested: Iterable) -> List:
    stack = [nested]
    output = []
    while stack:
        current = stack.pop()
        if isinstance(current, (list, tuple)):
            stack.extend(reversed(current))
        else:
            output.append(current)
    return output


def _ensure_pil(image_like) -> Image.Image:
    if isinstance(image_like, Image.Image):
        return image_like.convert("RGB")
    if isinstance(image_like, str):
        return Image.open(image_like).convert("RGB")
    if isinstance(image_like, bytes):
        return Image.open(io.BytesIO(image_like)).convert("RGB")
    if isinstance(image_like, np.ndarray):
        return Image.fromarray(image_like).convert("RGB")
    if isinstance(image_like, dict):
        if "path" in image_like:
            return Image.open(image_like["path"]).convert("RGB")
        if "bytes" in image_like:
            return Image.open(io.BytesIO(image_like["bytes"])).convert("RGB")
    raise TypeError(f"Unsupported image payload type: {type(image_like)}")


def _first_cuda_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@register_model("internvl2_5")
class InternVL2_5(lmms):
    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL2_5-8B",
        modality: str = "image",
        device: str = "cuda:0",
        device_map: str = "auto",
        batch_size: str = "1",
        max_tiles: int = DEFAULT_MAX_TILES,
        num_frames: int = 8,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_flash_attn: bool = True,
        **_: object,
    ):
        super().__init__()

        self.path = pretrained
        self.modality = modality
        self.max_tiles = max_tiles
        self.num_frames = num_frames
        self.image_size = DEFAULT_IMAGE_SIZE
        self.image_transform = build_transform(self.image_size)
        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "InternVL2.5 currently supports batch_size=1"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator

        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif device_map == "auto":
            self._device = torch.device(device)
            self.device_map = split_model(pretrained.split("/")[-1])
        else:
            self._device = torch.device(device)
            self.device_map = device_map

        if isinstance(self.device_map, dict) and torch.cuda.is_available():
            min_device = min(v for v in self.device_map.values() if isinstance(v, int))
            self._vision_device = torch.device(f"cuda:{min_device}")
        else:
            self._vision_device = _first_cuda_device()

        config = InternVLChatConfig.from_pretrained(self.path)
        self.llm_layers = config.llm_config.num_hidden_layers

        model_kwargs = dict(
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attn=use_flash_attn,
        )
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        if isinstance(self.device_map, dict):
            model_kwargs["device_map"] = self.device_map

        self.model = self._load_model(model_kwargs)
        self.tokenizer = self._load_tokenizer()

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in {
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            }

            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected DeepSpeed; ensure zero stage is set to 0 for evaluation")

            if accelerator.distributed_type in {DistributedType.FSDP, DistributedType.DEEPSPEED}:
                self.model = accelerator.prepare(self.model)
            else:
                self.model = accelerator.prepare_model(self.model, evaluation_mode=True)

            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            if not isinstance(self.device_map, dict):
                self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

        eval_logger.info("InternVL2.5 loaded from %s", self.path)

    # ---------------------------------------------------------------------
    # Loading utilities
    # ------------------------------------------------------------------
    def _ensure_snapshot(self) -> bool:
        if not isinstance(self.path, str) or os.path.isdir(self.path):
            return False
        eval_logger.warning("Forcing snapshot re-download for %s due to missing files", self.path)
        snapshot_download(
            repo_id=self.path,
            repo_type="model",
            allow_patterns=[
                "*.py",
                "*.json",
                "*.txt",
                "*.model",
                "*.safetensors",
                "*.bin",
                "*tokenizer*",
                "*config*",
            ],
            local_dir=None,
            local_dir_use_symlinks=False,
            force_download=True,
        )
        return True

    def _load_model(self, model_kwargs: dict) -> InternVLChatModel:
        try:
            return InternVLChatModel.from_pretrained(self.path, **model_kwargs).eval()
        except FileNotFoundError:
            if not self._ensure_snapshot():
                raise
            return InternVLChatModel.from_pretrained(self.path, **model_kwargs).eval()

    def _load_tokenizer(self) -> AutoTokenizer:
        try:
            return AutoTokenizer.from_pretrained(self.path, trust_remote_code=True, use_fast=False)
        except FileNotFoundError:
            if not self._ensure_snapshot():
                raise
            return AutoTokenizer.from_pretrained(self.path, trust_remote_code=True, use_fast=False)

    # ------------------------------------------------------------------
    # Core LMMS interface methods
    # ------------------------------------------------------------------
    def generate_until(self, requests) -> List[str]:
        outputs: List[str] = []
        progress = tqdm(total=len(requests), disable=(self._rank != 0), desc="InternVL2.5 responding")

        for request in requests:
            prompt, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            generation_kwargs = self._prepare_generation_kwargs(gen_kwargs)

            doc = self.task_dict[task][split][doc_id]
            visuals = doc_to_visual(doc)

            if self.modality == "image":
                pixel_values, num_patches = self._prepare_images(visuals)
                question = self._inject_image_tokens(prompt, num_patches)
                response = self.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    generation_config=generation_kwargs,
                    num_patches_list=num_patches,
                )
            elif self.modality == "video":
                pixel_values, num_patches, question = self._prepare_video(visuals, prompt)
                response = self.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    generation_config=generation_kwargs,
                    num_patches_list=num_patches,
                )
            else:
                raise ValueError(f"Unsupported modality: {self.modality}")

            outputs.append(response)
            progress.update(1)

        progress.close()
        return outputs

    def generate_until_multi_round(self, requests) -> List[str]:  # pragma: no cover - optional
        raise NotImplementedError("InternVL2.5 multi-round generation is not implemented")

    def loglikelihood(self, requests):  # pragma: no cover - not needed for this model
        raise NotImplementedError("InternVL2.5 does not support loglikelihood evaluation")

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------
    def _prepare_generation_kwargs(self, user_kwargs: dict) -> dict:
        kwargs = dict(DEFAULT_GEN_KWARGS)
        for key, value in (user_kwargs or {}).items():
            if key == "until":
                continue
            kwargs[key] = value
        return kwargs

    def _prepare_images(self, visuals) -> Tuple[Optional[torch.Tensor], Optional[List[int]]]:
        flat_visuals = _flatten(visuals)
        if not flat_visuals:
            return None, None

        tensors: List[torch.Tensor] = []
        patch_counts: List[int] = []
        for visual in flat_visuals:
            pil_image = _ensure_pil(visual)
            tiles = dynamic_preprocess(pil_image, max_num=self.max_tiles, image_size=self.image_size, use_thumbnail=True)
            stacked = torch.stack([self.image_transform(tile) for tile in tiles])
            tensors.append(stacked)
            patch_counts.append(stacked.shape[0])

        pixel_values = torch.cat(tensors, dim=0).to(torch.bfloat16).to(self._vision_device)
        return pixel_values, patch_counts

    def _prepare_video(self, visuals, prompt: str) -> Tuple[torch.Tensor, List[int], str]:
        flat_visuals = _flatten(visuals)
        if len(flat_visuals) != 1:
            raise ValueError(f"InternVL2.5 expects exactly one video, got {len(flat_visuals)}")

        video_path = flat_visuals[0]
        if isinstance(video_path, dict) and "path" in video_path:
            video_path = video_path["path"]
        if not isinstance(video_path, str):
            raise TypeError("Video input must be a file path")

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        if self.num_frames > max_frame:
            frame_indices = np.linspace(0, max_frame, num=self.num_frames, dtype=int)
        else:
            seg_size = (max_frame + 1) / self.num_frames
            frame_indices = [int(seg_size * idx + seg_size / 2) for idx in range(self.num_frames)]

        tiles: List[torch.Tensor] = []
        patch_counts: List[int] = []
        for frame_idx in frame_indices:
            image = Image.fromarray(vr[min(frame_idx, max_frame)].asnumpy()).convert("RGB")
            processed = dynamic_preprocess(image, max_num=1, image_size=self.image_size, use_thumbnail=True)
            stacked = torch.stack([self.image_transform(tile) for tile in processed])
            tiles.append(stacked)
            patch_counts.append(stacked.shape[0])

        pixel_values = torch.cat(tiles, dim=0).to(torch.bfloat16).to(self._vision_device)
        frame_tokens = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(patch_counts))])
        question = frame_tokens + prompt
        return pixel_values, patch_counts, question

    def _inject_image_tokens(self, prompt: str, num_patches: Optional[List[int]]) -> str:
        if not num_patches:
            return prompt
        image_tokens = " ".join(["<image>"] * len(num_patches))
        return f"{image_tokens}\n{prompt}"

    # ------------------------------------------------------------------
    # Misc helpers required by lmms-eval
    # ------------------------------------------------------------------
    @property
    def model(self):  # pragma: no cover - property defined for accelerator compatibility
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def tokenizer(self):  # pragma: no cover - property defined for completeness
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer = value

    def loglikelihood_rolling(self, requests):  # pragma: no cover
        raise NotImplementedError

    def fewshot_limit(self) -> int:
        return 0

    def max_length(self) -> int:
        return self.tokenizer.model_max_length

    def get_tokenizer(self):
        return self.tokenizer
