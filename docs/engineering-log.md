# Engineering Log

Running notes on design decisions and lessons learned.


### 2026-07-15

Discovered that deploying LoRA-based models with a large number of injected parameters significantly increases inference latency due to the overhead of model loading and parameter merging; balancing LoRA rank and batch size helped mitigate this. Also found that using FP16 precision during inference can cause subtle numerical issues with certain layers, so incorporating a small FP32 fallback for critical ops improved stability.

### 2026-07-20

Found that aggressively caching intermediate embeddings reduces API latency significantly but increases memory use and stale data risk, settled on a 15-minute TTL to balance freshness and performance. Learned that pipeline stages benefit from independent retries rather than a single monolithic retry, which avoids compounding delays on partial failures.

### 2026-07-25

During deployment of a LoRA fine-tuned model with dynamic batching, I observed that increasing batch size beyond 8 caused GPU memory spillovers due to unsynchronized gradient accumulation. Lowering the batch size or enabling gradient checkpointing resolved the issue without significant latency impact, highlighting the tradeoff between throughput and memory footprint in production setups.

### 2026-07-27

Noticed that chaining multiple LLM calls in a single request pipeline adds significant latency and increases failure points; batching simpler tasks or combining prompts where possible reduces overhead. Also, using LoRA fine-tuning with QLoRA compression helps keep model size manageable without a major accuracy hit, but tuning hyperparameters carefully is crucial to avoid convergence issues.

### 2026-08-02

Reviewed production AI engineering patterns and lessons learned today. Reinforced that measuring the change end-to-end beats reasoning about it in isolation — the numbers rarely match the intuition.

### 2026-08-08

Reviewed production AI engineering patterns and lessons learned today. Reinforced that measuring the change end-to-end beats reasoning about it in isolation — the numbers rarely match the intuition.

### 2026-08-11

Reviewed production AI engineering patterns and lessons learned today. Reinforced that measuring the change end-to-end beats reasoning about it in isolation — the numbers rarely match the intuition.
