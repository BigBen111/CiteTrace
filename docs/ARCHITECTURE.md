# CiteTrace Architecture Contract

## Pipeline

1. Deterministic routing: explicit citation-existence questions bypass LLM agents.
2. Citation scan: normalize the named target, scan every extracted chunk and the detected reference region, validate excerpts against source chunks, and report extraction coverage. A negative answer is allowed only with complete page and reference coverage.
3. Hybrid retrieval: for general questions, deterministic entity/BM25 retrieval runs first. A Qwen query planner is called only when direct retrieval is insufficient, with DeepSeek reserved for transport/API or invalid-output fallback.
4. Evidence table: assign stable Evidence IDs and validate every excerpt against its original chunk.
5. Answering: one answer agent consumes only the question and validated evidence table and returns an AnswerDraft with atomic factual claims bound to Evidence IDs.
6. Conditional verification: one verifier judges each atomic claim using only its linked evidence. GLM is primary and DeepSeek is used for transport/API or invalid-output fallback. Claims without valid links are not sent to a model and become UNKNOWN.
7. Aggregation: disagreement is treated as uncertainty rather than naive majority voting. Unsupported, unknown, or unverified claims must not silently enter the final answer.

## Runtime progress contract

The web data-flow view is driven by persisted backend events, not a timer-based
animation. A job's `stage_history` records each executed stage and its measured
duration. Optional stages such as `query_planner`, semantic verification, and
the three provider fallback stages are absent when they did not execute; the UI
renders those nodes as `SKIPPED` after completion.

The normal stage vocabulary is `question_routing`, `citation_scan`,
`direct_retrieval`, `query_planner`, `evidence_validation`,
`answer_generation`, `claim_binding`, `source_validation`,
`semantic_verification`, `aggregation`, and `completed`. Provider failover is
reported as `retrieval_fallback`, `answer_fallback`, or
`verification_fallback`.

## Stable extension points

- New retrieval method -> implement BaseRetriever
- New answer model -> implement BaseAnswerAgent
- New checker -> implement BaseVerifier
- New conflict policy -> implement BaseAggregator

The core pipeline must not depend on provider-specific SDK objects.

## Semantics

UNKNOWN means the current evidence is insufficient to confirm a claim; it does not mean the paper definitively omits the information.

UNSUPPORTED means the available evidence fails to support the claim or conflicts with it.

## First-version non-goals

- General paper summarization
- Research novelty assessment
- Open-ended scientific judgment
- UI-specific logic inside core modules
- Provider-specific logic inside the pipeline
