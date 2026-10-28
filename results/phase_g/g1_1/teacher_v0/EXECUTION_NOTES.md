# Execution notes, failures and scope

## Before official training

- Frozen BOUNDED_TEACHER_PROTOCOL.md and G11_RESULTS.md hashes matched. All727
  historical hashes,13 G1 manifest entries and22 G11 entries matched, including
  their independent14/23-file archives. No prior evidence was rewritten.
- Authorized AND interpretation was recorded in PROSPECTIVE_CLARIFICATION.md
  before new-task gradients/trials. Other frozen scientific criteria unchanged.
- Initial development test collection used an interpreter editable-install path
  pointing at main and could not import the new module. Correct isolated commands
  explicitly set PYTHONPATH=src:scripts. No model/training gate ran in that attempt.
- First focused run:2 passed/5 failed, exposing scalar-versus-vector PRNG context
  misuse and floating subtraction at the inclusive guard edge. Both fixed before
  any full-model gradient/trial. The zero-noise context uses the original fixed
  application ID0; evidence retains original example IDs. No stochastic operator
  arm is authorized by this implementation.
- Intermediate formatting/typing issues were fixed before source checkpoints.
  Seven focused regressions then passed; the combined teacher/task set passed13.
- ed957f7 passed new-task full-model preflight at0 counted updates. Its complete
  evidence and discarded trial states are retained under implementation_preflight_v0
  and implementation_preflight_v0_discarded_trial, with original bytes intact.
- Before official updates,4689293 added explicit checkpoint/evaluation binding,
  per-update frozen-moment verification and undefined relative-accuracy handling
  for zero-hit controls. Absolute accuracy inference, all NLL calculations and
  criteria unchanged. Seven focused regressions passed in53.84s. Required
  new-task preflight repeated and passed under that source. Both prospective
  step1 states are NOT COUNTED and were not training parents.

## Official experiment

Exactly100 updates, no extension. Checkpoint0 has exact fresh F1-overlaid means
and all-zero Adam moments, not either discarded trial or an old optimizer.
Every completed update persisted in a contiguous immutable receipt chain.
Checkpoints0/10/25/50/100 each have paired912-example A/B validation bound to
their actual parameter tree and original validation IDs. None was selected by
performance; endpoint100 was fixed. No invalid/nonfinite official attempt,
checkpoint failure, optimizer reset or hidden extra update occurred.

## Evaluation recovery

The first endpoint driver failed at eager NumPy embedding indexing after loading
an otherwise valid checkpoint. ENDPOINT_ATTEMPT_1_FAILURE.json records the exact
cause/path/source. No optimizer ran and no scientific gate had returned false.
A new real-model regression reproduces the error and confirms lossless JAX
conversion plus eager/JIT agreement (1 passed/7 deselected in7.24s).

The separately committed cd2a416 wrapper verifies frozen execution source hashes,
protocol/parents, runtime/backend, pinned Torx revision and full parameter bits,
then invokes unchanged endpoint logic. The retry used identical endpoint100
weights, examples, roots, criteria and bootstrap seeds, without training or
selecting favorable outcomes. All endpoint controls and uncertainty completed.
The original source remains unchanged; later reporting scripts are not presented
as preregistered training code.

## Interpretation and reporting limits

All frozen bounded teacher-necessity criteria pass. This is NOT learned-commit,
sampled-policy, fixed-neutral-superiority, convergence, language or full-G success.
Nonquery performance falls versus checkpoint50 although both mandated update0
guards pass; the full report retains this tradeoff. Class3 extent recall and low
whole-packet exactness are disclosed separately. Teacher emission correctness is
not a claim about generated extents. Diagnostic-only relative accuracy is null
when its control has zero hits, never an invented finite/infinite effect.

Structural action/score-ownership code was not changed. Historical exact/MC
oracles remain immutable; regular/slow suites provide the current regression
record in VALIDATION.json, not a new sampled scientific training arm.

No protected-test neural metrics were exposed. No original rejected-task arm,
argmax/sampled training, full G campaign, H/I science, codec/LM training or S4.
No main merge/push; isolated G commits only. Accepted teachers/checkpoints and
F/G evidence retain their original hashes.
