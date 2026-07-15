# Engineering Log

Running notes on design decisions and lessons learned.


### 2026-07-15

Discovered that deploying LoRA-based models with a large number of injected parameters significantly increases inference latency due to the overhead of model loading and parameter merging; balancing LoRA rank and batch size helped mitigate this. Also found that using FP16 precision during inference can cause subtle numerical issues with certain layers, so incorporating a small FP32 fallback for critical ops improved stability.
