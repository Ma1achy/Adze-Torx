# G.1 exploratory geometry follow-up

Declared after primary G.1 diagnosis, before this follow-up model execution.
No training, criterion changes, task selection or production edits.

Source inspection of `dit.py:330–355` shows committed partition geometry changes
block/within-block positional embeddings, although observed b/ell embeddings
stay neutral. Correct-versus-shifted effects therefore conflate mask and geometry.

On all original256 paired inputs, run both structures with:

1. block/within embeddings zeroed, attention retained;
2. block/within embeddings zeroed, all12 attention outputs zeroed.

Keep carrier/observed-b/ell embeddings, RoPE coordinates, all other weights,
inputs, noise roots and lengths unchanged. Both-zero control must reconstruct
the same identity-aligned h/byte outputs under repacking (max-logit error<=1e-5).
This is a numerical isolation check, not a scientific performance threshold.
Report all results, including any failures. No outcome from this follow-up
may replace or relax the old necessity gate.
