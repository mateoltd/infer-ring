# Qwen3.6 distributed MTP checkpoint

## Objective

Add opt-in native MTP speculative decoding for the pipeline-sharded
Qwen3.6-35B-A3B target without changing the default autoregressive path.

The target remains the exact verifier. Draft-head quantization therefore
cannot change the target distribution; a failed or unprofitable speculative
round must fall back to ordinary autoregressive decoding.

## Minimum viable path

1. Load the `qwen3_5_mtp` sidecar on rank 0.
2. Return the terminal target hidden state to rank 0 with verifier logits.
3. Draft a small block on rank 0 and verify it in one distributed target pass.
4. Restore every rank's local attention and recurrent state after partial
   acceptance.
5. Record proposed/accepted tokens and net generation throughput.
6. Disable speculation automatically when memory or measured throughput is
   worse than autoregressive decoding.

## Initial bounds

- Target weights remain at 4-bit.
- Begin at draft depth 1, then test depths 2 and 3.
- Only one target model may be loaded during a benchmark.
- Keep the existing autoregressive path as the default and fallback.
- Do not retain the feature as the flight default unless it is stable and
  reaches at least 1.3x net generation throughput on the actual Mac+iPhone
  pair.

## Stop conditions

Stop the experiment without weakening the existing runtime if:

- the target cannot load without iOS jetsam or disruptive macOS pressure;
- exact greedy output differs from autoregressive output;
- distributed cache rollback cannot be made deterministic;
- net throughput remains below 1.3x after draft-depth tuning; or
- the sidecar's additional memory makes the target unstable.
