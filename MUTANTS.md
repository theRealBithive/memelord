# Mutation testing record

Tool: mutmut 3.8 · scope: the modules listed in `only_mutate` of the `[tool.mutmut]`
section in `pyproject.toml` (which also has the run command and why the timeout
constant is raised). Tests selected: `tests/core`, `tests/ratings`. Each run below
names the modules it covered; a run is scoped to the modules that changed.

Denominators are the generated mutants per file
(`grep -cE '^\s*def x.*__mutmut_[0-9]+\(' mutants/<file>`; the pattern must allow the
`ǁ` separators mutmut puts into class-method names, or those are not counted).

## Runs on 2026-10-03 (text search: `core/siglip.py`, `ratings/vector_bank.py`, `ratings/search.py`, `ratings/similar.py`)

Denominators of run 1: siglip 73, vector_bank 115, search 248, similar 26 → **462**.
All runs against a copy of the dev DB, with `only_mutate` narrowed to the modules of
the run and restored afterwards.

| run | scope | mutants | killed | timeout | survived | note |
|---|---|---|---|---|---|---|
| 1 | all four | 462 | 360 | 17 | 85 | before the tests listed below |
| 2 | vector_bank | 125 | 91 | 0 | 34 | after `rank` switched to the plain matmul (`_unit` adds 13 mutants, `rank` shrinks from 20 to 17) |
| 3 | vector_bank + search | 371 | 326 | 0 | 45 | after the operator asked for the 23 survivors first filed as "outside the contract" to be killed too; vector_bank has 123 mutants once `_load` lost its explicit iterator chunk size. The 45 are the equivalent list below plus `encode_stale_search_embeddings` 9, killed after run 3 by the one-query test (verified alone) |

Timeouts of run 1: all 17 are in `core/siglip.py` and pay the lazy transformers
import inside the wall limit. Each was re-run alone with `MUTANT_UNDER_TEST=<id>`
against `tests/core/test_siglip.py`: 16 fail their test (killed); the one real
survivor, `_SiglipImageEncoder.forward` 1 (`pixel_values=None`), is in the list
of kills below.

### Killed by tests added the same day (43 + the timed-out `forward` 1), two removed by simplification
Each verified alone with `MUTANT_UNDER_TEST` against the updated test module or by
the next run.

| mutant | test |
|---|---|
| siglip `_SiglipImageEncoder.forward` 1 | `test_adapter_returns_pooler_output_not_the_first_patch` (the batch reaches the tower) |
| siglip `get_image_encoder` 15 | `test_image_encoder_loads_vision_tower_in_eval_mode_on_cpu` (the adapter wraps the loaded tower) |
| siglip `encode_text` 19 | `test_text_model_receives_only_input_ids` (the padded ids reach the model) |
| search `stale_search_images` 3 | `test_current_stamp_without_a_vector_is_stale` |
| search `encode_stale_search_embeddings` 13 | `test_unrated_images_follow_download_order_not_insertion_order` |
| search `encode_stale_search_embeddings` 59, 61, 62, 63, 66, 67, 68 | `test_the_encoder_pass_receives_the_arguments_it_was_given` |
| search `encode_stale_search_embeddings` 88, 102 | `test_missing_and_unreadable_files_are_counted_and_stay_stale` (two files of each kind) |
| search `encode_stale_search_embeddings` 3, 4, 107, 112, 114–118, 123, 124 | `test_progress_line_reports_done_of_total_under_the_label` (run 3) |
| search `encode_stale_search_embeddings` 1, 2 | `test_defaults_are_chunks_of_256_rows_in_batches_of_32` (run 3) |
| search `encode_stale_search_embeddings` 9 | `test_rows_are_loaded_without_the_taste_blob_and_in_one_query` (after run 3, verified alone) |
| search `last_index_report` 24, 25, 26 | `test_a_worker_that_left_no_result_is_reported_in_plain_words` (run 3) |
| search `last_index_report` 45, 46 | `test_the_stop_rule_is_reported_with_the_count`, exact sentence (run 3) |
| vector_bank `_refresh_if_changed_locked` 3 | `test_unchanged_version_costs_two_counts_and_reads_no_blob` (returns False) |
| vector_bank `_load` 29; `rank` 3, 5 | float32 assertions in `test_ranking_equals_a_fresh_computation_from_the_database`, `test_empty_bank_ranks_to_nothing_without_error`, `test_blobs_from_the_database_round_trip_into_unit_rows` |
| vector_bank `_load` 33, 36 | `test_ranking_equals_a_fresh_computation_from_the_database` compares the similarity values; see the production changes below |
| vector_bank `_unit` 12 (run 2) | `test_a_zero_query_ranks_everything_at_zero_without_nan` |
| vector_bank `_unit` 4, 6 (run 2) | `test_a_float64_query_is_ranked_in_float32` |
| vector_bank `rank` 14, 16 (11, 13 after the matmul change) | `test_ties_keep_content_hash_order_even_in_large_groups` (run 3) |
| vector_bank `_load` 2, 13 | removed: `.iterator()` now runs without an explicit chunk size; 2000 was Django's default, so the knob was dead |

Production changes after run 1: `VectorBank.rank` ranked with
`brain.cosine_similarity_matrix`, which re-normalises and copies the whole bank on
every call, so the row normalisation in `_load` was dead weight (its Frobenius-norm
mutants 33 and 36 survived) and every search copied 2 × 77 MB at 25k images. `rank`
now takes the plain matmul of the unit rows with the unit query (`_unit`). After
run 2, `_load` dropped `chunk_size=2000` from `iterator()` for the reason in the
table. Run 3 measures both modules after these changes.

No accepted bucket: the 23 survivors first filed as outside the contract (log line,
message wording, default values, sort kind) were killed or removed at the
operator's request the same day.

### Equivalent (45): identical behaviour on every reachable input
Ids are run 1's; after the two production changes `rank` 19 is 16, and the `_load`
ids from 21 on shift by two (19, 20, 22, 24, 25, 39, 46, 47 in run 3).

- vector_bank `__init__` 5, 6, 7, 9, 11, 12; `reset` 1, 2, 3, 5, 7, 8; `_load` 21, 22, 24, 26, 27 — placeholder state. `rank()` always calls `_refresh_if_changed_locked()` first and a version of `None` or `""` never equals a pair of counts, so `_load` overwrites `_hashes` and `_matrix` before anything reads them; in the empty branch `_matrix` is never read because `rank` returns on an empty `_hashes` (`None` is as falsy as `[]`). Only `size()` before the first `rank()` could tell, and no production code calls `size()`.
- vector_bank `current_version` 4 — counting non-purged instead of purged rows. The matrix is the image of the stamp set, which the purge flag does not change; a purge moves either count and a fresh start zeroes both, so both versions reload at every point where the matrix content could differ. Only the number of SQL reads in the compound case "one purge plus one insert between two calls" differs.
- vector_bank `_load` 41, 48, 49 — the zero-norm replacement only ever divides a zero row (0/1 = 0/2), and with the condition removed a zero row would become NaN; no stored vector is all-zero (the pooled output of either encoder never is), same argument as `cosine_similarity_matrix` 33, 43 below.
- vector_bank `rank` 19 — `kind="STABLE"`: numpy parses the sort kind case-insensitively (`np.argsort(kind="STABLE")` sorts on numpy 2.5.3), so it is the same call.
- vector_bank `_unit` 2, 8 — `reshape(None)` and `reshape(-2)`: every caller hands in a 1-d vector (`encode_text`, `bytes_to_embedding`), and numpy infers the size for any negative value; `reshape(-1)` stays as the statement of the expected shape.
- vector_bank `first_allowed` 4, 7, 8; search `encode_stale_search_embeddings` 73, 76, 77, 80, 83, 84 — `strict=` on zips whose operands have equal length by construction (`rank()` returns hashes and similarities of one length; `chunk`/`paths` come from the same list; `valid_paths`/`embeddings` by `brain.encode`'s contract).
- search `encode_stale_search_embeddings` 8 — `only()` always includes the primary key, so dropping `content_hash` from its list changes nothing.
- search `encode_stale_search_embeddings` 14 — `nulls_last=None` drops the NULLS LAST clause; SQLite sorts NULL below every value, so in DESC order it comes last anyway. SQLite is the only backend.
- search `encode_stale_search_embeddings` 97 — `save()` without `update_fields` on a row loaded with `.only("content_hash", "file_path")`: Django limits such a save to the loaded fields plus the ones assigned since (`file_path`, `search_embedding`, `search_embedding_model`), so a score given while the pass runs survives either way. The explicit list stays because it states what the job may touch.
- search `last_index_report` 29, 31, 34, 37, 39, 42 — defaults of `.get("remaining")` and `.get("encoded")`: every successful result comes from `run_search_index`, which always writes both keys, so the defaults are never consulted.
- search `candidate_images` 16 — `score >= min_score` is already false for NULL in SQL, so `score__isnull=False` is redundant for the database; it stays so the reader does not need SQL's three-valued logic.
- search `rank_by_text` 5; similar `rank_similar` 7 — `values_list(flat=True)` without a field name yields the first concrete field, which is the primary key `content_hash`; the name stays because the equivalence hangs on field order.

## Runs on 2026-10-02 (`core/brain.py`, `core/dedup.py`, `ratings/embeddings.py`)

Denominators: brain 341 (incl. the `_PooledEncoder` methods), dedup 22, embeddings 118 → **555**.

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
