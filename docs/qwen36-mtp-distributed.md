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

## Device result (M3 Pro 18 GB + iPhone 15 Pro)

Status: **no-go for Qwen3.6-35B-A3B on this pair**. Native MTP remains
available behind `INFER_RING_MTP=1`, but is disabled by default.

Tested artifacts:

- Target: `mlx-community/Qwen3.6-35B-A3B-4bit`
  at revision `38740b847e4cb78f352aba30aa41c76e08e6eb46`
  (four affine 4-bit shards, 19 GB on disk).
- Draft sidecar: `mlx-community/Qwen3.6-35B-A3B-MTP-4bit`
  (affine 4-bit, 465 MB on disk).
- Architecture limit: 262,144 positions; Infer Ring's distributed flight
  limit remains 25,000 tokens because the iPhone was previously measured
  near its Q8-KV jetsam boundary there.

Observed baseline behavior:

1. The model loaded successfully with both a 31/9 and a 35/5 Mac/iPhone
   layer split. Cached reloads took approximately 18 seconds.
2. A 96-token autoregressive request on the 31/9 split remained incomplete
   after more than six minutes, establishing generation below 0.27 token/s.
3. A later run exposed an unbounded single-token Metal command buffer and
   terminated with `kIOGPUCommandBufferCallbackErrorTimeout`. The engine now
   honors `INFER_RING_PIPELINE_EVAL_INTERVAL` during decode as well as
   prefill, and watchdog-safe runs used a four-layer interval.
4. Even with that fix, the iPhone worker disappeared before the first token
   on the 31/9 split. The 35/5 safety split also closed both ranks after
   roughly 41 seconds without producing a first token.

The autoregressive verifier therefore fails the stability and usability
gates before speculation is enabled. An MTP speed ratio would be misleading:
MTP still invokes the same verifier and cannot repair verifier GPU timeouts,
iOS process loss, or sub-0.27 token/s target execution. The feature is kept
as experimental source work, with exact rollback and automatic low-acceptance
fallback, but it is not the flight default.
