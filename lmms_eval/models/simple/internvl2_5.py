"""Thin wrapper around :class:`InternVL2` to provide an InternVL2.5 entry point.

The InternVL2.5 checkpoints share the same code-path as InternVL2, so we simply
reuse the implementation while exposing a dedicated registry alias with a more
appropriate default `pretrained` identifier.
"""

from __future__ import annotations

from lmms_eval.api.registry import register_model

from .internvl2 import InternVL2


@register_model("internvl2_5")
class InternVL2_5(InternVL2):
    """InternVL2.5 model wrapper.

    The base :class:`InternVL2` implementation already supports the InternVL2.5
    architecture; this subclass simply registers a new model name with a default
    Hugging Face checkpoint that points to the InternVL2.5 weights.
    """

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL2_5-2B",
        **kwargs,
    ):
        super().__init__(pretrained=pretrained, **kwargs)
