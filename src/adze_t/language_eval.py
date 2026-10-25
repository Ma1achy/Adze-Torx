"""Task-separated evaluation contracts for raw-byte model outputs."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Protocol, Sequence


GENERATION_PROTOCOL_V0 = "PHASE_I_BYTE_GENERATION_EVAL_V0"
STOP_POLICY_V0 = "committed_extent_or_answer_independent_budget_v0"


class EvaluationTask(str, Enum):
    TEACHER_LATENT_RECONSTRUCTION = "teacher_latent_reconstruction"
    DENOISED_LATENT_RECONSTRUCTION = "denoised_latent_reconstruction"
    CONDITIONAL_GENERATION = "conditional_generation"
    UNCONDITIONAL_GENERATION = "unconditional_generation"
    INFILLING = "infilling"


_REFERENCE_TASKS = {
    EvaluationTask.TEACHER_LATENT_RECONSTRUCTION,
    EvaluationTask.DENOISED_LATENT_RECONSTRUCTION,
    EvaluationTask.CONDITIONAL_GENERATION,
    EvaluationTask.INFILLING,
}


@dataclass(frozen=True)
class GenerationCase:
    case_id: str
    task: EvaluationTask
    prompt: bytes
    suffix: bytes
    byte_budget: int
    expected: bytes | None = None

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if self.byte_budget <= 0:
            raise ValueError("byte_budget must be positive")
        if self.task in _REFERENCE_TASKS and self.expected is None:
            raise ValueError(f"{self.task.value} requires a reference output")
        if self.task == EvaluationTask.UNCONDITIONAL_GENERATION and self.expected is not None:
            raise ValueError("unconditional generation cannot carry an oracle reference")


@dataclass(frozen=True)
class GenerationRequest:
    protocol: str
    case_id: str
    task: EvaluationTask
    prompt: bytes
    suffix: bytes
    byte_budget: int
    stop_policy: str
    root: int


@dataclass(frozen=True)
class ComputeObservation:
    wall_seconds: float
    peak_rss_bytes: int | None = None
    model_applications: int | None = None
    estimated_flops: float | None = None

    def __post_init__(self) -> None:
        if self.wall_seconds < 0:
            raise ValueError("wall time must be nonnegative")
        for name, value in (
            ("peak_rss_bytes", self.peak_rss_bytes),
            ("model_applications", self.model_applications),
            ("estimated_flops", self.estimated_flops),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be nonnegative")


class ByteGenerator(Protocol):
    def generate(self, request: GenerationRequest) -> bytes: ...


def conditional_case(
    case_id: str, sequence: bytes, prompt_bytes: int, *, byte_budget: int
) -> GenerationCase:
    if not 0 < prompt_bytes < len(sequence):
        raise ValueError("prompt_bytes must split a non-empty sequence")
    return GenerationCase(
        case_id,
        EvaluationTask.CONDITIONAL_GENERATION,
        sequence[:prompt_bytes],
        b"",
        byte_budget,
        sequence[prompt_bytes:],
    )


def infilling_case(
    case_id: str, sequence: bytes, start: int, end: int, *, byte_budget: int
) -> GenerationCase:
    if not 0 <= start < end <= len(sequence):
        raise ValueError("invalid infilling interval")
    return GenerationCase(
        case_id,
        EvaluationTask.INFILLING,
        sequence[:start],
        sequence[end:],
        byte_budget,
        sequence[start:end],
    )


def unconditional_case(case_id: str, *, byte_budget: int) -> GenerationCase:
    return GenerationCase(
        case_id,
        EvaluationTask.UNCONDITIONAL_GENERATION,
        b"",
        b"",
        byte_budget,
    )


def reconstruction_case(
    case_id: str,
    expected: bytes,
    *,
    denoised: bool,
    byte_budget: int,
) -> GenerationCase:
    task = (
        EvaluationTask.DENOISED_LATENT_RECONSTRUCTION
        if denoised
        else EvaluationTask.TEACHER_LATENT_RECONSTRUCTION
    )
    return GenerationCase(case_id, task, b"", b"", byte_budget, expected)


def _case_root(root: int, case_id: str) -> int:
    digest = hashlib.sha256(f"{root}:{case_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _edit_distance(left: bytes, right: bytes) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _repetition_rate(value: bytes, n: int = 4) -> float:
    if len(value) < n:
        return 0.0
    grams = [value[index : index + n] for index in range(len(value) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def _task_metrics(cases: Sequence[GenerationCase], outputs: Sequence[bytes]) -> dict[str, object]:
    output_bytes = sum(len(output) for output in outputs)
    byte_counts = [0] * 256
    valid_utf8 = 0
    for output in outputs:
        for value in output:
            byte_counts[value] += 1
        try:
            output.decode("utf-8", errors="strict")
            valid_utf8 += 1
        except UnicodeDecodeError:
            pass
    metrics: dict[str, object] = {
        "count": len(cases),
        "generated_bytes": output_bytes,
        "generated_byte_counts": byte_counts,
        "generated_byte_coverage": sum(count > 0 for count in byte_counts),
        "utf8_valid_fraction": valid_utf8 / len(cases),
        "mean_4gram_repetition_rate": sum(map(_repetition_rate, outputs)) / len(cases),
        "distinct_output_fraction": len(set(outputs)) / len(outputs),
    }
    references = [case.expected for case in cases]
    if all(reference is not None for reference in references):
        expected = [reference for reference in references if reference is not None]
        widths = [max(len(output), len(reference)) for output, reference in zip(outputs, expected)]
        correct = sum(
            sum(left == right for left, right in zip(output, reference))
            for output, reference in zip(outputs, expected)
        )
        metrics.update(
            {
                "reference_bytes": sum(map(len, expected)),
                "exact_accuracy": sum(
                    output == reference for output, reference in zip(outputs, expected)
                )
                / len(cases),
                "aligned_byte_accuracy": correct / max(sum(widths), 1),
                "mean_normalized_edit_distance": sum(
                    _edit_distance(output, reference) / max(width, 1)
                    for output, reference, width in zip(outputs, expected, widths)
                )
                / len(cases),
            }
        )
    return metrics


def evaluate_outputs(
    cases: Sequence[GenerationCase],
    outputs: Sequence[bytes],
    compute: ComputeObservation,
) -> dict[str, object]:
    if len(cases) != len(outputs) or not cases:
        raise ValueError("cases and outputs must have the same non-zero length")
    if any(not isinstance(output, bytes) for output in outputs):
        raise TypeError("generator outputs must be bytes")
    grouped: dict[str, dict[str, object]] = {}
    for task in EvaluationTask:
        indices = [index for index, case in enumerate(cases) if case.task == task]
        if indices:
            grouped[task.value] = _task_metrics(
                [cases[index] for index in indices], [outputs[index] for index in indices]
            )
    generated_bytes = sum(len(output) for output in outputs)
    return {
        "protocol": GENERATION_PROTOCOL_V0,
        "count": len(cases),
        "tasks": grouped,
        "generated_bytes": generated_bytes,
        "bytes_per_second": generated_bytes / max(compute.wall_seconds, 1.0e-12),
        "cases_per_second": len(cases) / max(compute.wall_seconds, 1.0e-12),
        "compute": asdict(compute),
    }


def run_generation_harness(
    generator: ByteGenerator,
    cases: Sequence[GenerationCase],
    *,
    root: int,
    peak_rss_bytes: int | None = None,
    model_applications: int | None = None,
    estimated_flops: float | None = None,
) -> tuple[tuple[bytes, ...], dict[str, object]]:
    """Generate with case-declared, answer-independent budgets."""
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("generation case IDs must be unique")
    started = time.monotonic()
    outputs = tuple(
        generator.generate(
            GenerationRequest(
                protocol=GENERATION_PROTOCOL_V0,
                case_id=case.case_id,
                task=case.task,
                prompt=case.prompt,
                suffix=case.suffix,
                byte_budget=case.byte_budget,
                stop_policy=STOP_POLICY_V0,
                root=_case_root(root, case.case_id),
            )
        )
        for case in cases
    )
    if any(not isinstance(output, bytes) for output in outputs):
        raise TypeError("generator outputs must be bytes")
    if any(len(output) > case.byte_budget for case, output in zip(cases, outputs)):
        raise ValueError("generator exceeded a predeclared byte budget")
    observation = ComputeObservation(
        wall_seconds=time.monotonic() - started,
        peak_rss_bytes=peak_rss_bytes,
        model_applications=model_applications,
        estimated_flops=estimated_flops,
    )
    return outputs, evaluate_outputs(cases, outputs, observation)
