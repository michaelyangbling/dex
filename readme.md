# Dexmate AI, Data Platform — System Architecture
The data→model→agent flywheel behind Dexmate's robots: a **data plane** (ingest from the public products, label, version), a **model plane** (train / serve voice + motion + LLM models, cloud + on-robot edge), an **agent plane** (LLM agents over a self-updating knowledge base), on a shared **control plane** (WorkOS auth, SpiceDB ReBAC, token CLI, forum). Thesis: a robotics company wins on the data flywheel, so production usage becomes tomorrow's training set.

**Stack:** React 19 + TS (FE); Rust/Axum control plane, Python ML, Go ingest + CLI (BE); S3 + Iceberg lakehouse, Postgres + pgvector, Kafka + Flink, Ray + PyTorch, MLflow registry, Feast; KServe + Triton + vLLM (cloud), ONNX + TensorRT (edge); SpiceDB; AWS EKS-GPU + Fargate.

**1. Auth & Account** — WorkOS AuthKit SSO + TOTP MFA + SCIM; custom React UI (not hosted pages); JWT for web + CLI; session revoke, GDPR export, soft-delete.
**2. ReBAC Access** — one SpiceDB graph; capabilities-at-scope, never role strings; `authz.require()` facade + reverse-query lists; outbox writes + consistency tokens (no drift, no stale reads); caveats for consent / break-glass; PII-reveal is a standalone audited capability.
**3. CLI & Tokens** — `dex` CLI; scoped personal + service tokens (rotate / revoke); resolves through the *same* SpiceDB facade — not a privilege bypass; per-token audit + rate limit.
**4. Data Collection** — every interaction (events, content, telemetry, agent traces) → Protobuf → Kafka → Flink (PII scrub, consent-filter) → Iceberg bronze; provenance + consent stamped at ingest; GDPR erasure propagates to the lake.
**5. Labeling** — per-modality projects (vision / NLP / motion / voice); consensus review + κ quality gate; weak-supervision + model-in-the-loop pre-labels; active learning picks high-value items; labels versioned + lineage-linked.
**6. Data Management** — medallion lakehouse (bronze→silver→gold) on Iceberg; dataset versioning (time-travel, hashes); catalog + column lineage; Feast feature store (online/offline parity); quality gates block promotion; retention + region governance.
**7. Training** — Argo / Temporal pipelines; Ray + PyTorch distributed (FSDP) on GPU; families = LLM / voice / motion; W&B tracking, MLflow registry (staging→prod), Ray Tune HPO; eval promotion-gated (golden + safety).
**8. Serving (cloud)** — KServe / Triton / vLLM; canary + shadow → auto-rollback; autoscale-to-zero; drift + accuracy monitoring auto-triggers retrain (flywheel closes); every inference traced + sampled back.
**9. Voice & Motion (edge)** — quantize / distill → ONNX / TensorRT on-robot (sub-10 ms control loop); signed OTA, cohort canary, instant rollback; deterministic safety layer overrides learned policy; behavior streams back to §4.
**10. Agent Management** — harness (context / tools / memory / lifecycle, deterministic replay); versioned registry; runtime guardrails (schema-validation, injection defense, budgets, kill-switch); per-skill eval + A/B; replayable operator console.
**11. Knowledge Base** — self-updating, not static: CDC from every service → embed (pgvector) + GraphRAG; hybrid retrieval + re-rank as a tool in the loop; answers / corrections written back as memory (compounds); permission-aware retrieval (no leak).
**12. Forum** — categories / topics / threaded replies, accepted-solution, reactions, moderation + audit; doubles as a labeled corpus (QA pairs, preference votes) feeding §11 + §7.
**Cross-cutting** — i18n; sanitize + rate-limit + strict CORS; SOC 2 / GDPR audit + export; OTel + Langfuse tracing; drift / cost dashboards; AWS CDK, EKS + Fargate, blue/gree

---

## Minimal runnable mock — §4 Data Collection flywheel

A stdlib-only stand-in for the ingest path, so the pipeline *shape* is runnable without Kafka / Flink / Iceberg. In-memory queues replace each hop:

```
interaction  ->  [topic]  ->  Flink(consent-filter + PII scrub)  ->  bronze lake
                 (Kafka)       (stream processor)                    (Iceberg)
```

Demonstrates the readme's §4 guarantees: consent-filtering, PII scrub, provenance stamping, and GDPR erasure propagating to the lake — asserting the invariants (no PII leak, no consent violation) that gate promotion downstream to §5/§7.

```bash
python3 mock_flywheel.py   # no dependencies to install
```

See [mock_flywheel.py](mock_flywheel.py). Real system swaps `Topic` → Kafka, `flink()` → Flink jobs, `bronze` list → Iceberg bronze tables.

## Minimal runnable mock — RLHF fine-tuning of the robot reaction model (§6/§7/§9/§10)

Robot reaction to its environment is modeled as a "GPT" — a policy that, given a state prompt, emits an action. It's collected, stored in a feature store, and fine-tuned on human feedback, ReAct-style:

```
ReAct loop (mocked GridWorld env):
  Reason  -> GPTReactionModel scores each action given state features (the "prompt")
  Act     -> apply chosen action to the mocked environment
  Observe -> write (state, action, reward) to the FeatureStore (online + offline)
          -> synthesize human preference feedback (preferred vs rejected action)

RLHF fine-tuning:
  collect preference pairs from the feedback above
  -> DPO-style update: raise score(preferred) - score(rejected) for that state
  -> reaction model improves without a reward-model rollout, same shape as real RLHF
```

Run it — no dependencies to install, pure stdlib:

```bash
python3 mock_rlhf_flywheel.py
```

Sample run: before fine-tuning the policy (small random init) reaches the goal in 1/20 episodes; after 3 epochs of RLHF fine-tuning on ~450 collected human preference pairs, it reaches the goal in 20/20 episodes with 0 collisions, in a third of the steps.

See [mock_rlhf_flywheel.py](mock_rlhf_flywheel.py). Maps to the real system: `GridWorld` → real robot + sensors, `GPTReactionModel` → an actual LLM/policy head fine-tuned via DPO/PPO, `FeatureStore` → Feast online/offline store (§6), `synth_human_feedback` → real operator/forum feedback (§12) captured as preference labels, `fine_tune()` → the training pipeline (§7) gated by eval before promotion (§8).