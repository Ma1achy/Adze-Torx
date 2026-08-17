# FIXED_STATE_TRANSITION_V0

`FIXED_STATE_TRANSITION_V0` is a controlled-depth benchmark for repeated
application of a learned fixed transition. It does not supply a fresh random
program or lookup table in each prompt.

## Fixed rule

The state is a periodic ring of 64 bits, serialized as eight bytes. Within
each byte, lower-indexed state positions use least-significant-bit-first order.

For every bit position `i`, one update is elementary cellular automaton Rule
30:

```text
next[i] = left[i] XOR (center[i] OR right[i])
left[i] = state[(i - 1) mod 64]
right[i] = state[(i + 1) mod 64]
```

Rule 30 was predeclared because the Boolean update is nonlinear, local, and
mixing, with no expected rapid attraction to an identity or constant state for
random 64-bit inputs. It was not selected using model performance.

## Protocol

Depths are exactly:

```text
1, 2, 3, 4, 6, 8, 12, 16
```

Every prompt has ten bytes:

```text
byte 0: task tag 0xE2
byte 1: requested transition depth
bytes 2..9: random initial 64-bit state
```

Every target has eight bytes and is the complete state after exactly the
requested number of Rule-30 updates. Prompt and target lengths are constant
across depth.

## Baselines

- bit accuracy: 50%
- byte accuracy: 1/256
- exact 64-bit state accuracy: 2^-64

The calibration and authoritative gates additionally require learned Q0/Q1
depth curves; chance alone is not a sufficient difficulty criterion.

## Seeds

- train: 930
- validation: 931
- test: 932
- model-independent transition-quality audit: 933
