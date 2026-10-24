"""Conditional-generation and infilling evaluation for raw byte outputs."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class GenerationCase:
    case_id: str
    prompt: bytes
    expected: bytes
    suffix: bytes = b""
    task: str = "conditional"


@dataclass(frozen=True)
class ComputeObservation:
    wall_seconds: float
    peak_rss_bytes: int | None = None
    model_applications: int | None = None
    estimated_flops: float | None = None


class ByteGenerator(Protocol):
    def generate(self, prompt: bytes, *, suffix: bytes, max_bytes: int, seed: int) -> bytes: ...


def conditional_case(case_id: str, sequence: bytes, prompt_bytes: int) -> GenerationCase:
    if not 0 < prompt_bytes < len(sequence):
        raise ValueError("prompt_bytes must split a non-empty sequence")
    return GenerationCase(case_id, sequence[:prompt_bytes], sequence[prompt_bytes:])


def infilling_case(case_id: str, sequence: bytes, start: int, end: int) -> GenerationCase:
    if not 0 <= start < end <= len(sequence):
        raise ValueError("invalid infilling interval")
    return GenerationCase(
        case_id, sequence[:start], sequence[start:end], sequence[end:], "infilling"
    )


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


def evaluate_outputs(
    cases: Sequence[GenerationCase],
    outputs: Sequence[bytes],
    compute: ComputeObservation,
) -> dict[str, object]:
    if len(cases) != len(outputs) or not cases:
        raise ValueError("cases and outputs must have the same non-zero length")
    exact = 0
    correct = 0
    compared = 0
    edit = 0.0
    valid_utf8 = 0
    repetition = 0.0
    output_bytes = 0
    for case, output in zip(cases, outputs, strict=True):
        exact += output == case.expected
        width = max(len(output), len(case.expected))
        correct += sum(a == b for a, b in zip(output, case.expected, strict=False))
        compared += width
        edit += _edit_distance(output, case.expected) / max(width, 1)
        try:
            output.decode("utf-8", errors="strict")
            valid_utf8 += 1
        except UnicodeDecodeError:
            pass
        repetition += _repetition_rate(output)
        output_bytes += len(output)
    payload: dict[str, object] = {
        "count": len(cases),
        "exact_accuracy": exact / len(cases),
        "aligned_byte_accuracy": correct / max(compared, 1),
        "mean_normalized_edit_distance": edit / len(cases),
        "utf8_valid_fraction": valid_utf8 / len(cases),
        "mean_4gram_repetition_rate": repetition / len(cases),
        "distinct_output_fraction": len(set(outputs)) / len(outputs),
        "generated_bytes": output_bytes,
        "bytes_per_second": output_bytes / max(compute.wall_seconds, 1.0e-12),
        "cases_per_second": len(cases) / max(compute.wall_seconds, 1.0e-12),
        "compute": asdict(compute),
    }
    return payload


def run_generation_harness(
    generator: ByteGenerator,
    cases: Sequence[GenerationCase],
    *,
    seed: int,
    max_bytes: int,
    peak_rss_bytes: int | None = None,
    model_applications: int | None = None,
    estimated_flops: float | None = None,
) -> tuple[tuple[bytes, ...], dict[str, object]]:
    started = time.monotonic()
    outputs = tuple(
        generator.generate(
            case.prompt,
            suffix=case.suffix,
            max_bytes=min(max_bytes, max(1, len(case.expected))),
            seed=seed + index,
        )
        for index, case in enumerate(cases)
    )
    observation = ComputeObservation(
        wall_seconds=time.monotonic() - started,
        peak_rss_bytes=peak_rss_bytes,
        model_applications=model_applications,
        estimated_flops=estimated_flops,
    )
    return outputs, evaluate_outputs(cases, outputs, observation)
