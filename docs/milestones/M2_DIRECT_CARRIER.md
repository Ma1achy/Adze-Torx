# M2 — Direct-carrier reconstruction

## Goal

Train the smallest Adze-like persistent carrier without learned segmentation or outer refinement.

Start with:

```text
fixed C
h channel active
b fixed
length fixed
direct site interactions
R = 1
fixed S
small Q
```

Use a tiny synthetic reconstruction task before bytes.

## Requirements

- context conditioning path;
- training-only clean carrier target encoder;
- stochastic Torx core;
- deterministic/debug baseline;
- observable state trajectory;
- checkpointable model state;
- exact shape invariants.

## Gate

- training loss decreases across several seeds;
- stochastic update beats no-update/random baseline;
- debug/deterministic controls behave as predicted;
- gradient diagnostics remain finite;
- all carrier states retain fixed shapes.
