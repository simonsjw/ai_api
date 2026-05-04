"""
Local resource monitoring for any provider using GPU / system resources.

This module provides a provider-agnostic way to check available GPU VRAM
and system RAM before loading or running models. It is intended to be used
by any local inference backend (Ollama, vLLM, llama.cpp, etc.).

Detection prefers nvidia-smi (most accurate) and falls back to torch.cuda.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

import torch

from .errors import APIError


class ResourceError(APIError):
    """Insufficient local resources (GPU VRAM or system RAM) detected
    before attempting to run a model.

    This is a pre-flight error raised when free resources fall below the
    configured minimum thresholds.
    """

    pass


__all__ = ["ResourceError", "check_local_resources"]


def check_local_resources(
    *,
    min_vram_gb: float = 6.0,
    min_system_ram_gb: float = 8.0,
    logger: logging.Logger | None = None,
    allow_cpu_fallback: bool = True,
) -> dict[str, Any]:
    """Inspect local host resources (GPU + system RAM).

    Preference order for GPU detection:
    1. `nvidia-smi` (most accurate, works even without PyTorch)
    2. `torch.cuda` (fallback when nvidia-smi is unavailable or fails)

    Records structured availability via the supplied (or module) logger.
    Raises `ResourceError` if any monitored resource is below threshold.

    Parameters
    ----------
    min_vram_gb : float, default 6.0
        Minimum free VRAM (in GiB) required.
    min_system_ram_gb : float, default 8.0
        Minimum available system RAM (GiB).
    logger : logging.Logger, optional
        Logger to use for structured `extra=` records. Defaults to module logger.
    allow_cpu_fallback : bool, default True
        If False, absence of a usable GPU is treated as a hard failure.

    Returns
    -------
    dict[str, Any]
        Structured snapshot of detected resources (useful for logging/metrics).

    Raises
    ------
    ResourceError
        If free VRAM or system RAM is below the respective minimum.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    resources: dict[str, Any] = {
        "gpu_available": False,
        "detection_method": "none",
        "gpu_vram_total_gb": None,
        "gpu_vram_free_gb": None,
        "system_ram_total_gb": None,
        "system_ram_available_gb": None,
    }

    # ------------------------------------------------------------------
    # 1. NVIDIA path (preferred – works in containers, WSL, bare metal)
    # ------------------------------------------------------------------
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        total_vram = 0.0
        free_vram = 0.0
        for line in lines:
            total_str, free_str = line.split(",")
            total_vram += float(total_str) / 1024.0
            free_vram += float(free_str) / 1024.0

        resources.update(
            {
                "gpu_available": True,
                "detection_method": "nvidia-smi",
                "gpu_vram_total_gb": round(total_vram, 2),
                "gpu_vram_free_gb": round(free_vram, 2),
            }
        )
        logger.info(
            "NVIDIA GPU resources detected via nvidia-smi",
            extra={k: v for k, v in resources.items() if v is not None},
        )

    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception) as exc:
        logger.debug(
            "nvidia-smi unavailable or failed – falling back to torch.cuda",
            extra={"error": str(exc), "error_type": type(exc).__name__},
        )

        # ------------------------------------------------------------------
        # 2. torch.cuda fallback (only if nvidia path failed)
        # ------------------------------------------------------------------
        if torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                total_gb = total_bytes / (1024 ** 3)
                free_gb = free_bytes / (1024 ** 3)

                resources.update(
                    {
                        "gpu_available": True,
                        "detection_method": "torch.cuda",
                        "gpu_vram_total_gb": round(total_gb, 2),
                        "gpu_vram_free_gb": round(free_gb, 2),
                    }
                )
                logger.info(
                    "GPU resources detected via torch.cuda (nvidia-smi fallback)",
                    extra={k: v for k, v in resources.items() if v is not None},
                )
            except Exception as torch_exc:
                logger.warning(
                    "torch.cuda available but mem_get_info failed",
                    extra={"error": str(torch_exc)},
                )
        else:
            resources["gpu_available"] = False
            logger.info(
                "No CUDA-capable GPU detected (neither nvidia-smi nor torch.cuda)",
                extra=resources,
            )

    # ------------------------------------------------------------------
    # 3. System RAM (always useful, independent of GPU path)
    # ------------------------------------------------------------------
    try:
        import psutil  # optional dependency – graceful degradation

        mem = psutil.virtual_memory()
        resources["system_ram_total_gb"] = round(mem.total / (1024 ** 3), 2)
        resources["system_ram_available_gb"] = round(mem.available / (1024 ** 3), 2)
        logger.info(
            "System RAM snapshot",
            extra={
                "system_ram_total_gb": resources["system_ram_total_gb"],
                "system_ram_available_gb": resources["system_ram_available_gb"],
            },
        )
    except ImportError:
        logger.debug("psutil not installed – skipping system RAM check")
    except Exception as ram_exc:
        logger.warning("Failed to read system RAM", extra={"error": str(ram_exc)})

    # ------------------------------------------------------------------
    # 4. Threshold enforcement
    # ------------------------------------------------------------------
    vram_free = resources.get("gpu_vram_free_gb") or 0.0
    if resources["gpu_available"] and vram_free < min_vram_gb:
        raise ResourceError(
            f"Insufficient free GPU VRAM ({vram_free:.1f} GiB free < {min_vram_gb:.1f} GiB required). "
            "Consider a smaller/quantized model, reducing context length, or CPU fallback.",
            details={
                "detected": resources,
                "min_vram_gb": min_vram_gb,
                "min_system_ram_gb": min_system_ram_gb,
            },
        )

    ram_avail = resources.get("system_ram_available_gb") or 0.0
    if ram_avail < min_system_ram_gb:
        if not allow_cpu_fallback:
            raise ResourceError(
                f"Insufficient system RAM ({ram_avail:.1f} GiB < {min_system_ram_gb:.1f} GiB). "
                "CPU fallback disabled.",
                details=resources,
            )
        else:
            logger.warning(
                "Low system RAM – CPU fallback may be slow or unstable",
                extra={"available_ram_gb": ram_avail, "threshold": min_system_ram_gb},
            )

    if not resources["gpu_available"] and not allow_cpu_fallback:
        raise ResourceError(
            "No GPU detected and CPU fallback disabled. "
            "Install NVIDIA drivers + nvidia-smi or enable torch CUDA.",
            details=resources,
        )

    logger.info("Local resource check passed", extra=resources)
    return resources
