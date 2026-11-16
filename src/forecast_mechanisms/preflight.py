"""Check case files and available resources before inference."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .core import read_jsonl, validate_cases
from .model import EXPECTED_BLOCKS, EXPECTED_HIDDEN


def preflight(cases_path: str | Path, work_dir: str | Path, min_disk_gib: float = 20.0, min_gpu_gib: float = 24.0) -> dict[str, Any]:
    if min_disk_gib <= 0 or min_gpu_gib <= 0:
        raise ValueError("resource thresholds must be positive")
    cases = read_jsonl(cases_path)
    validate_cases(cases)
    work = Path(work_dir)
    disk_anchor = work if work.exists() else work.parent
    if not disk_anchor.exists():
        raise ValueError("work directory or parent does not exist")
    disk_free_gib = shutil.disk_usage(disk_anchor).free / 2**30
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda else None
        gpu_gib = torch.cuda.get_device_properties(0).total_memory / 2**30 if cuda else None
        gpu_free_gib = torch.cuda.mem_get_info(0)[0] / 2**30 if cuda else None
    except ImportError:
        cuda, gpu_name, gpu_gib, gpu_free_gib = False, None, None, None
    # Five first-round arms plus at most six second-round conditions. This
    # overestimates storage when a first verdict cannot be parsed.
    max_prompts = len(cases) * 11
    activation_gib = max_prompts * (EXPECTED_BLOCKS + 1) * EXPECTED_HIDDEN * 4 / 2**30
    checks = {
        "cuda_available": cuda,
        "gpu_memory_sufficient": gpu_free_gib is not None and gpu_free_gib >= min_gpu_gib,
        "disk_sufficient": disk_free_gib >= min_disk_gib,
    }
    return {
        "ready": all(checks.values()), "checks": checks,
        "gpu_name": gpu_name, "gpu_memory_gib": gpu_gib,
        "gpu_free_memory_gib": gpu_free_gib,
        "minimum_free_gpu_gib": min_gpu_gib, "minimum_free_disk_gib": min_disk_gib,
        "disk_free_gib": disk_free_gib, "case_count": len(cases),
        "maximum_prompt_count": max_prompts,
        "estimated_activation_gib": activation_gib,
    }
