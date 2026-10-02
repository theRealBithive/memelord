# Mutation testing record

Tool: mutmut 3.8 · scope: `core/brain.py`, `core/dedup.py`, `ratings/embeddings.py`
(the `[tool.mutmut]` section in `pyproject.toml` has the run command and why the
timeout constant is raised). Tests selected: `tests/core`, `tests/ratings`.

Denominators are the generated mutants per file (`grep -cE '^\s*def x_\w+__mutmut_[0-9]+' mutants/<file>`):
brain 341 (incl. the `_PooledEncoder` methods), dedup 22, embeddings 118 → **555**.

## Runs on 2026-10-02

| run | killed | timeout | survived | note |
|---|---|---|---|---|
| 3 | 411 | 22 | 122 | before the tests listed below |
| 4 (stopped at 520/555) | 403 | 21 | 96 | fresh run after the added tests; stopped by the operator with 35 mutants unevaluated (`_load_from_hub` 1–21 and the tail of `brain.py`), see below |

Timeouts: the covering tests import `transformers` lazily, so a mutant of
`get_transform`/`get_encoder`/`_load_from_hub`/`_PooledEncoder` pays the import
inside the 45 s wall limit of a forked child. Each of the 22 was re-run alone
with `MUTANT_UNDER_TEST=<id>`; every one fails its test (killed).

### Run 4 against run 3

Run 4 was stopped at 520 of 555 mutants because a full run takes about 35 minutes here (one child at a time on a shared SQLite file, each child paying the torch/transformers import); the project rule is now to run mutmut only when one of the three scoped modules changes.

- Every one of the 96 survivors of run 4 also survived run 3; no new survivor appeared.
- None of the 27 mutants verified killed by hand (`kill_verification`) survived run 4.
- 17 of the 21 timeouts are the same import-heavy mutants as in run 3, each verified killed by running its covering test by hand with `MUTANT_UNDER_TEST`. The other four (`encode` 18, `get_encoder` 1–3) were killed outright in run 3 and timed out in run 4 while the full test suite ran on the same machine; they count as killed on the strength of run 3.
- The 35 unevaluated mutants include 5 run-3 survivors (all in the accepted/equivalent lists below) and 5 run-3 timeouts.

## Survivors of run 3 and their disposition

### Killed by tests added the same day (26 + the timed-out `get_encoder_4`)
Verified individually with `MUTANT_UNDER_TEST` before run 4.

| mutant | test |
|---|---|
| brain `embedding_to_bytes` 2, 4, 6 | `test_embedding_to_bytes_normalises_dtype_and_batch_dim_to_768_float32` |
| brain `embedding_to_bytes` 11, 12; `bytes_to_embedding` 8 | `test_dimension_errors_name_expected_and_actual_length` |
| brain `save_classifier` 1, 3, 5 | `test_save_classifier_creates_missing_parent_directories` |
| brain `encode` 11, 12, 82 | `test_encode_moves_batches_to_where_the_encoder_lives`, `test_encode_honours_an_explicit_device_over_the_encoder_home` |
| brain `get_encoder` 4 (timeout in run 3) | `test_autodetection_asks_for_cpu_when_cuda_is_absent` |
| embeddings `reencode_stale_embeddings` 43, 45, 46, 50, 51 | `test_reencode_hands_its_encoder_transform_and_batch_size_to_the_encoder_pass` |
| embeddings `reencode_stale_embeddings` 72, 77 | `test_every_unreadable_file_in_a_chunk_is_counted_and_the_rest_is_still_encoded` |
| embeddings `reencode_stale_embeddings` 81 | `test_a_rating_given_while_the_pass_runs_survives` |
| embeddings `stale_images` 3; dedup `from_db` 33 | `test_current_stamp_without_a_vector_is_stale_and_stays_out_of_the_index` |
| dedup `from_db` 24, 27 | `test_stored_fingerprint_is_recognised_again_and_a_distant_one_is_not` |
| dedup `from_db` 41, 43 | `test_embedding_bank_stays_float32_after_adding_to_an_empty_index` |

### Equivalent (14): identical behaviour on every reachable input
- `encode` 14 — `torch.device("cpu")` → `None` for an encoder without parameters: `Tensor.to(None)` is a no-op and the transform yields CPU tensors, so the batch is on the CPU either way.
- `encode` 86 — `torch.stack(tensors)` defaults to `dim=0`.
- `predict_proba` 10, `embedding_to_bytes` 8, `DedupIndex.add` 20 — numpy treats any negative reshape size as "infer" (`reshape(-2)` ≡ `reshape(-1)`, checked on numpy 2.5.3).
- `DedupIndex.add` 12 — `reshape(-1)` gives a 1-d row and `np.vstack` promotes it to `(1, 768)`; the bank is identical.
- `cosine_similarity_matrix` 33, 43 — the replacement for a zero norm only ever divides a zero vector: 0/1 = 0/2.
- `reencode_stale_embeddings` 57, 60, 61, 64, 67, 68 — `strict=` on zips whose operands come from the same list (`chunk`/`paths`) or are equal in length by `brain.encode`'s contract (`valid_paths`/`embeddings`).

### Accepted as outside the contract (82) — operator decision requested
- **Log text and cadence (79).** `encode` 2, 3, 22–25, 31–33, 35–40, 46, 47, 72, 73, 77, 94–101, 103, 105–110, 112–118, 120–122, 125, 131–133, 135, 137, 139, 142, 143 (54); `load_classifier` 8–12, 19–21, 26–29 (12); `reencode_stale_embeddings` 3, 4, 47, 52, 91, 96, 98–102, 107, 108 (13). They change only loguru/logging lines: progress wording, how often a line is written, and the rate/ETA arithmetic shown in it. No contract item (V1–V9) covers log output; pinning the wording would test the code, not a promise.
- **Default argument values (3).** `encode` 1 (`batch_size` 32→33), `reencode_stale_embeddings` 1 (`chunk_size` 256→257) and 2 (`batch_size` 32→33): performance and memory knobs with identical results.
