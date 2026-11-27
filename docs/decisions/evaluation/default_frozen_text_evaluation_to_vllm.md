# Default frozen text evaluation to vLLM

Date: 2026-08-17.
Status: current.

- Use a persistent vLLM engine with continuous batching for large frozen
  text-only base/LoRA evaluations. Request only the literal decision-token
  logprobs and constrain generation to one token; do not inherit the training
  microbatch for inference.
- Retain Transformers eager inference as a bounded parity/debug path and for
  adapters vLLM cannot represent. Before a new serving campaign, compare the
  causal master with the checksum-rebased serving adapter on identical rows,
  tolerate recorded numerical differences, and require a meaningful adapter
  effect under both backends.
- The first AISI transfer launch used fixed eager batches and was superseded
  before interpretation by the parity-gated vLLM campaign. Keep any partial
  eager artifacts diagnostic-only and summarize only the vLLM result tree.
