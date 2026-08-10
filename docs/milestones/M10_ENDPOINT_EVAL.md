# M10 — Endpoint and ablation suite

Assemble the complete fixed-capacity Adze-T endpoint.

## Required ablations

At minimum:

- causal draft vs global refinement;
- Q recurrence;
- S denoising;
- R refinement;
- boundary channel;
- length channel;
- direct-carrier vs committed hierarchical routing;
- weight tying;
- cycle conditioning;
- selector type;
- adaptive vs fixed compute;
- byte representation.

## Evaluation rule

Do not claim a mechanism helps based only on final benchmark score if multiple mechanisms changed simultaneously.

## Gate

Every central architectural claim has a controlled measurement and regression test where practical.
