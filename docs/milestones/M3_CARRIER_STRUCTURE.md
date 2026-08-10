# M3 — Boundary/length channels and corruption

Add:

```text
h
b
length
```

with distinct observed/predicted structural state.

## Requirements

- UNKNOWN corruption state for discrete structure;
- absorbing categorical corruption;
- `length=0` non-emitting semantics;
- separate losses/metrics per channel;
- direct-carrier routing remains fixed;
- corruption can be disabled channel-by-channel.

## Ablations

```text
h only
h + b
h + length
h + b + length
```

## Gate

Each state channel is recoverable independently and joint training does not silently destroy content performance.
