"""Capability inventory for optional MLX modality packages."""

from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    package: str
    module: str
    summary: str
    features: tuple[str, ...]
    commands: tuple[str, ...]


CAPABILITY_SPECS: tuple[CapabilitySpec, ...] = (
    CapabilitySpec(
        name="mlx-lm",
        package="mlx-lm",
        module="mlx_lm",
        summary="Text-only MLX loader used for legacy Qwen/Llama pack training and eval.",
        features=(
            "text model loading",
            "tokenized perplexity eval",
            "LoRA pack train/apply compatibility",
        ),
        commands=("mlx_lm.generate",),
    ),
    CapabilitySpec(
        name="mlx-vlm",
        package="mlx-vlm",
        module="mlx_vlm",
        summary="Vision-language and omni-model runtime for Gemma 4 unified any-to-any bases on macOS.",
        features=(
            "Gemma 4 unified model loading",
            "image/audio/video prompt generation",
            "OpenAI-compatible local server and chat UI",
        ),
        commands=("mlx_vlm.generate", "mlx_vlm.server", "mlx_vlm.chat_ui"),
    ),
    CapabilitySpec(
        name="mlx-audio",
        package="mlx-audio",
        module="mlx_audio",
        summary="Dedicated Apple Silicon audio stack for speech IO around any-to-any packs.",
        features=(
            "text-to-speech",
            "speech-to-text",
            "speech-to-speech, VAD, diarization, and enhancement workflows",
        ),
        commands=("mlx_audio.tts.generate", "python -m mlx_audio.stt.generate", "mlx_audio.server"),
    ),
)


def capability_report() -> list[dict[str, Any]]:
    """Return import/version status for optional MLX modality packages."""

    rows: list[dict[str, Any]] = []
    for spec in CAPABILITY_SPECS:
        module_spec = importlib.util.find_spec(spec.module)
        try:
            version = metadata.version(spec.package)
        except metadata.PackageNotFoundError:
            version = None
        rows.append(
            {
                "name": spec.name,
                "package": spec.package,
                "module": spec.module,
                "installed": module_spec is not None and version is not None,
                "version": version,
                "summary": spec.summary,
                "features": list(spec.features),
                "commands": list(spec.commands),
            }
        )
    return rows


def missing_capabilities(rows: list[dict[str, Any]] | None = None) -> list[str]:
    """Return names of packages missing from the current environment."""

    report = rows if rows is not None else capability_report()
    return [str(row["name"]) for row in report if not row["installed"]]

