# G.1.1 — Authorized bounded teacher100 results

## Outcome

All frozen bounded teacher-necessity criteria pass.
Exactly100 new updates; endpoint100 fixed before execution. No extension or checkpoint selection. Protected-test examples/metrics remain unexposed.

This establishes bounded trained use of future-peer attention under teacher-committed structure only. It does NOT establish superiority to fixed-neutral/overconnected routing, learned semantic commits, sampled score-gradient trainability, language modelling, convergence, or Phase-G completion.

## Fixed endpoint and paired uncertainty

| Metric | A correct teacher | B valid singleton cut |
|---|---:|---:|
| Query payload byte accuracy | 59.831870% | 3.691521% |
| Query payload NLL (full256 vocabulary) | 1.242239118 | 11.162091255 |
| All3 query payload bytes exact | 20.175439% | 0.000000% |
| Whole packet byte accuracy | 88.116533% | 81.350881% |
| Whole packet exact | 3.618421% | 0.000000% |

A−B accuracy: 56.140352pp; paired95% CI [54.42251566 57.89565167]pp.
NLL A−B: -9.919852183; paired95% CI [-10.22324804359763, -9.592879803772922].
Relative NLL change: -88.870911%; CI [-89.39563331 -88.31219992]%.
2000 whole-example paired percentile replicates, resampled16 per each57 length strata; seeds1161/1162, controlcode0. Query3 bytes are not independently resampled. NLL relative change is ratio of aggregate means.

## Both nonquery guards (prospective AND)

Outside-query-carrier accuracy: 16.775563% → 91.371542% (+74.595979pp); required no loss >2pp.
Nonquery NLL: 11.713524818 → 0.332729131 (-97.159445%); required increase <=5%.
Equal-example means are primary as declared before any new-task gradient. Byte-weighted and complementary nonquery definitions remain separately recorded, never substituted into the guard.

## Checkpoint curves

| Update | A query accuracy | A query NLL | B query accuracy | B query NLL | A nonquery accuracy | A whole packet accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.6082% | 34.105274 | 1.5716% | 36.625740 | 16.7756% | 14.6855% |
| 10 | 4.0205% | 4.120398 | 4.0205% | 4.149336 | 48.9490% | 45.3980% |
| 25 | 2.4123% | 3.572403 | 2.5950% | 3.601905 | 81.1210% | 72.7712% |
| 50 | 3.5088% | 3.423562 | 2.9971% | 3.515099 | 95.5139% | 85.2348% |
| 100 | 59.8319% | 1.242239 | 3.6915% | 11.162091 | 91.3715% | 88.1165% |

Nonquery accuracy declines from95.51% at50 to91.37% at100: a visible retrieval/reconstruction tradeoff, not hidden by the easy update0 guard. Whole-packet exactness is only3.62%; passing this query-focused bounded gate is not general reconstruction success. All100 batch loss components and gradient norms are in training.jsonl; full checkpoint validation is per-example, not just an averaged loss.

## Negative controls

| Control | Query accuracy | Query NLL |
|---|---:|---:|
| C | 3.691521% | 9.698366165 |
| D_substitute | 3.216375% | 9.885070801 |
| D_outside | 60.124272% | 1.251760840 |
| E_model | 3.326024% | 11.452812195 |
| E_codec | 0.000000% | 57.251060486 |

C erases peer before encoding; D_substitute replaces it with independent content; D_outside changes nuisance signatures; E_model/E_codec are causal direct decoding with oracle extents, not generation. C_post and no-attention controls are also tested in the fixed64-example flow audit.
Unavailable-information counterlabel responses: {'A': 40.071044921875, 'B': 0.0, 'C': 0.0, 'C_post': 0.0, 'D_outside': 40.071044921875, 'D_substitute': 0.0, 'no_attention': 0.0, 'E_codec': 0.0, 'E_model': 0.0}. All B/C/C_post/D_substitute/E/no-attention query-logit responses are exactly0. Full28-symbol negative slices remain invariant and score1/28 under the separately labelled valid-domain argmax. Model accuracy above always uses all256 logits.
All correct query→peer paths are reachable and all cut paths excluded on the frozen64 examples. Pack/unpool identity, separate dynamic capacity cuts, overflow rejection, oracle injection, eager/JIT and deterministic/public-Torx parity pass. See endpoint_flow.json for full numerical traces; CIs for other controls are in endpoint_controls.json and cannot override A−B.

## Prescribed stratification

All57 lengths, query→peer gaps, outside/full outside competitors, fixed-neutral reachability and final-partial packets are in endpoint_strata.json with counts. Full/partial nonquery byte supports and summed losses/correct counts are retained per example. Queries are always full extent: partial-query support0, no invented partial-query quality score.

| Stratum | Value | Count | A accuracy | B accuracy | A NLL | B NLL |
|---|---:|---:|---:|---:|---:|---:|
| gap | 1 | 425 | 57.1765% | 3.3725% | 1.304519 | 11.677500 |
| gap | 2 | 201 | 62.8524% | 4.9751% | 1.152750 | 10.676020 |
| gap | 3 | 126 | 60.8466% | 3.7037% | 1.223226 | 10.748780 |
| gap | 4 | 87 | 64.7510% | 1.9157% | 1.086061 | 10.955880 |
| gap | 5 | 73 | 59.3607% | 4.1096% | 1.344995 | 10.458930 |
| length | 8 | 16 | 66.6667% | 2.0833% | 0.856394 | 17.139834 |
| length | 9 | 16 | 66.6667% | 2.0833% | 0.721052 | 18.346253 |
| length | 10 | 16 | 68.7500% | 2.0833% | 1.014176 | 19.437009 |
| length | 11 | 16 | 60.4167% | 2.0833% | 1.205495 | 21.889680 |
| length | 12 | 16 | 68.7500% | 8.3333% | 1.041934 | 14.628017 |
| length | 13 | 16 | 64.5833% | 6.2500% | 1.063604 | 16.324986 |
| length | 14 | 16 | 66.6667% | 8.3333% | 0.986984 | 17.978705 |
| length | 15 | 16 | 60.4167% | 2.0833% | 1.017632 | 14.637861 |
| length | 16 | 16 | 68.7500% | 2.0833% | 0.857866 | 14.278544 |
| length | 17 | 16 | 58.3333% | 2.0833% | 1.090324 | 13.223410 |
| length | 18 | 16 | 52.0833% | 2.0833% | 1.507158 | 13.941587 |
| length | 19 | 16 | 66.6667% | 8.3333% | 0.819744 | 11.639180 |
| length | 20 | 16 | 62.5000% | 6.2500% | 1.359930 | 12.060724 |
| length | 21 | 16 | 70.8333% | 2.0833% | 0.918543 | 13.410289 |
| length | 22 | 16 | 70.8333% | 4.1667% | 0.759437 | 10.522154 |
| length | 23 | 16 | 70.8333% | 6.2500% | 1.031714 | 10.901659 |
| length | 24 | 16 | 47.9167% | 4.1667% | 1.552931 | 12.236291 |
| length | 25 | 16 | 54.1667% | 2.0833% | 1.108715 | 12.402065 |
| length | 26 | 16 | 60.4167% | 4.1667% | 1.263955 | 11.262363 |
| length | 27 | 16 | 75.0000% | 2.0833% | 1.076466 | 11.169709 |
| length | 28 | 16 | 62.5000% | 0.0000% | 1.033163 | 11.319527 |
| length | 29 | 16 | 58.3333% | 4.1667% | 1.234751 | 11.761101 |
| length | 30 | 16 | 52.0833% | 6.2500% | 1.265569 | 10.253484 |
| length | 31 | 16 | 62.5000% | 2.0833% | 1.113575 | 10.475059 |
| length | 32 | 16 | 45.8333% | 0.0000% | 1.510594 | 11.795632 |
| length | 33 | 16 | 64.5833% | 8.3333% | 1.282640 | 10.460517 |
| length | 34 | 16 | 50.0000% | 0.0000% | 1.447005 | 9.739482 |
| length | 35 | 16 | 54.1667% | 2.0833% | 1.500399 | 9.510329 |
| length | 36 | 16 | 56.2500% | 0.0000% | 1.417076 | 12.055814 |
| length | 37 | 16 | 68.7500% | 4.1667% | 1.146083 | 8.292558 |
| length | 38 | 16 | 60.4167% | 0.0000% | 1.396595 | 10.967914 |
| length | 39 | 16 | 43.7500% | 4.1667% | 1.532881 | 10.179144 |
| length | 40 | 16 | 66.6667% | 4.1667% | 1.299568 | 12.843423 |
| length | 41 | 16 | 58.3333% | 0.0000% | 1.292332 | 10.997194 |
| length | 42 | 16 | 52.0833% | 2.0833% | 1.461377 | 9.174744 |
| length | 43 | 16 | 56.2500% | 12.5000% | 1.318763 | 7.158226 |
| length | 44 | 16 | 60.4167% | 2.0833% | 1.087850 | 9.473860 |
| length | 45 | 16 | 60.4167% | 8.3333% | 1.329617 | 8.042271 |
| length | 46 | 16 | 54.1667% | 2.0833% | 1.220744 | 8.864315 |
| length | 47 | 16 | 54.1667% | 0.0000% | 1.451615 | 10.587940 |
| length | 48 | 16 | 66.6667% | 6.2500% | 0.976157 | 7.329125 |
| length | 49 | 16 | 66.6667% | 2.0833% | 1.160365 | 8.113389 |
| length | 50 | 16 | 56.2500% | 8.3333% | 1.277952 | 6.454093 |
| length | 51 | 16 | 54.1667% | 10.4167% | 1.488887 | 8.413723 |
| length | 52 | 16 | 47.9167% | 0.0000% | 1.659728 | 9.108908 |
| length | 53 | 16 | 75.0000% | 2.0833% | 1.034239 | 10.763531 |
| length | 54 | 16 | 64.5833% | 4.1667% | 1.093687 | 9.867210 |
| length | 55 | 16 | 64.5833% | 6.2500% | 1.175307 | 8.837144 |
| length | 56 | 16 | 58.3333% | 2.0833% | 1.376772 | 8.308511 |
| length | 57 | 16 | 50.0000% | 0.0000% | 1.460476 | 9.198021 |
| length | 58 | 16 | 54.1667% | 4.1667% | 1.477865 | 8.016572 |
| length | 59 | 16 | 54.1667% | 10.4167% | 1.427077 | 6.982683 |
| length | 60 | 16 | 43.7500% | 2.0833% | 1.849854 | 8.087479 |
| length | 61 | 16 | 58.3333% | 4.1667% | 1.358523 | 7.236353 |
| length | 62 | 16 | 60.4167% | 2.0833% | 1.438424 | 7.997028 |
| length | 63 | 16 | 54.1667% | 4.1667% | 1.578727 | 8.776111 |
| length | 64 | 16 | 58.3333% | 0.0000% | 1.377334 | 11.366468 |
| outside_competitor | 0 | 23 | 63.7681% | 2.8986% | 0.972109 | 17.192688 |
| outside_competitor | 1 | 889 | 59.7300% | 3.7120% | 1.249228 | 11.006069 |
| outside_full_competitor | 0 | 111 | 65.7658% | 3.3033% | 0.947588 | 18.629039 |
| outside_full_competitor | 1 | 801 | 59.0096% | 3.7453% | 1.283071 | 10.127346 |
| fixed_neutral_reachable | 0 | 132 | 53.0303% | 2.7778% | 1.459166 | 7.919473 |
| fixed_neutral_reachable | 1 | 780 | 60.9829% | 3.8462% | 1.205528 | 11.710842 |
| final_partial | 0 | 240 | 58.7500% | 2.6389% | 1.283810 | 11.468810 |
| final_partial | 1 | 672 | 60.2183% | 4.0675% | 1.227392 | 11.052549 |

## Initialization, health and provenance

Training source commit `4689293a73f22a37dbb156cef658820ea03985fd`; protocol binding `bd733883a39e307d920b4a81665a39eb0d370841424b0e0f3a05335386021153`. Frozen original protocol SHA35cbd1fed3b0490b73b83693c7246b01d5d41057295046b95d4022e514d8e29c. Prospective conjunctive clarification and both implementation preflights are preserved.
F1 parent SHA520d3723d3a3187aa08ff3f467f434d43b579cdc4b0ce667727dfad96e83cf57; G01 external codec SHA213d71be58b4e3a0eb685330ff8f20ad0d2574184d59060890baa4a2eed586e7. Initial full tree84ab204d1cea2a1833cbbdeadc7993879ad29a3c4c354d4960c221c951daaeeb. Exact overlay means, fresh Adam0; no prior optimizer inheritance.
Both shared-parameter S1 applications use condition0 and identical observation h: neutral predictor → teacher commit → replay. NOT S2 denoising. L4/Q3/S1/R0; lambda0; AdamW3e-4/wd.01/clip1/batch32; frozen declared loss weights. External codec and target encoder/rho remain unchanged; the output decoder is explicitly trainable.
New-task first raw full-model gradient norm 2201.717285156; clipped1. Prospective update finite and discarded. Official masked raw norm range [10.17209243774414, 2201.673583984375]. All101 parameter/Adam states finite; frozen parameters AND moments bitwise unchanged at every durable state. 727 protected historical hashes and G1/G11 manifests still match.
Per-example b/ell predictions, boundary counts/F1, per-class extent supports/correct counts, exact predicted structure, emission length, h-MSE, byte NLL/exactness and missing/extra counts are retained in each validation file. Teacher commitment correctness is oracle control, not a learned-commit claim. structural_churn denotes neutral→teacher changed-site fraction per replay, NOT temporal learned-policy churn.

## Checkpoint payload identities (full means/Adam m/v)

| Update / absolute Adam step | Payload SHA256 |
|---:|---|
| 0 | `a8a4cdded14e0f18fe12c8710f83f4f4b426aa2bfd0b1c440586aa32a96f5b2f` |
| 10 | `e296607f6377f080327eeb9404e2e9a740c05594da30fbfb1ea79465639e2609` |
| 25 | `570eb4b2a670e3e3f398777b21d924f464121bbd7d9c3f5839b299471cf56102` |
| 50 | `225fdae1622f572660ab745748ce1047ab2d4f6c63479507da81c840242a6cca` |
| 100 | `fa798be7d9d6244054b1053bd86c340c4d2f3582290dc72a31e6877c24f81a93` |

completion_integrity.json retains exact paths and parameter-tree identities. Immutable journal receipts form a verified contiguous parent-hash chain,0..100; training.jsonl is a derived view. No failed attempt is a counted official update.

## Execution correction and limitations

The first endpoint driver failed on an eager NumPy/JAX indexing boundary after recovery; it did not change any state or fail a scientific criterion. ENDPOINT_ATTEMPT_1_FAILURE.json preserves it. A separately committed wrapper losslessly converts arrays and calls unchanged endpoint logic after validating source/parents/runtime and parameter bits. endpoint_array_conversion.json binds its exact source hash. No data, roots, thresholds, weights or update count changed; no extra training or selective rerun occurred.
The first engineering preflight was also preserved before a pre-training reporting/identity hardening revision. Both trial updates are NOT COUNTED, never parents of official training. No hidden extension or rejected-task restart.

## Validation and reproduction

PYTHONPATH=src:scripts python scripts/run_phase_g11_teacher.py --stage preflight and --stage train executed under4689293 (hard cap100). This historical source snapshot is bound in protocol_binding.json; later reporting files must not be confused with a changed training protocol. Current evaluation reproduction: PYTHONPATH=src:scripts python scripts/evaluate_phase_g11_teacher.py. It verifies the frozen execution sources/checkpoint, uses no optimizer and does not expose protected test.
Focused tests, whole-project regular/slow suites, format/lint/types/boundaries and explicit private-Torx scan are recorded in VALIDATION.json and validation_logs/. Check exact counts/status there; historical counts are not substituted.
No main merge/push. Accepted F/G histories preserved. No argmax/sampled training, full G campaign, H/I science, codec/LM training or S4. Further experiments require separate authorization. STOP.
