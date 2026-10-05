# CiteTrace

CiteTrace is a modular multi-agent system for **factual question answering over research papers**.

The system is intentionally narrow: it does not try to be a general paper-reading assistant. Its objective is to reduce **unsupported factual claims** by requiring traceable paper evidence, independent claim verification, and abstention when evidence is insufficient or conflicting.

## Core research question

Can heterogeneous multi-agent cross-checking improve the reliability of factual question answering over research papers compared with a single-model or ordinary RAG pipeline?

## Architecture

```text
Question
  |
  +--> Retrieval Agent A --+
  |                        |
  +--> Retrieval Agent B --+--> Evidence Pool
                                 |
                              Answer Agent
                                 |
                             Atomic Claims
                                 |
                     +-----------+-----------+
                     |                       |
                Verifier A              Verifier B
                     |                       |
                     +-----------+-----------+
                                 |
                         Agreement Resolver
                                 |
                    Answer / Qualify / Abstain
```

### Cross Retrieval

Each retrieval agent independently interprets the same user question and generates its own retrieval queries. Results are merged and de-duplicated.

Purpose: reduce evidence misses caused by one model's query formulation.

### Evidence-grounded Answering

The answer agent only sees the question and the shared Evidence Pool. It returns both a readable answer and atomic factual claims.

### Cross Verification

Each atomic claim is independently checked by multiple verifier agents against the same original paper evidence. First-pass verifiers do not see each other's verdicts.

Verdicts:

- `SUPPORTED`
- `PARTIALLY_SUPPORTED`
- `UNSUPPORTED`
- `UNKNOWN`

### Disagreement-aware Resolution

CiteTrace does not use naive majority voting. Verifier disagreement is treated as an uncertainty signal. The first implementation is deliberately conservative and abstains when material support is disputed or insufficient.

## Default agent layout

The current runnable configuration is:

- Retrieval Agent A: Qwen
- Retrieval Agent B: DeepSeek
- Answer Agent: Qwen
- Verifier A: DeepSeek
- Verifier B: GLM

All roles use stable interfaces, so models can be swapped without changing the pipeline.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
```

Fill the API keys, model names, and OpenAI-compatible base URLs in `.env`.

### Ingest a paper

```bash
citetrace ingest ./paper.pdf --paper-id demo-paper
```

### Ask a factual question

```bash
citetrace ask --paper-id demo-paper "How many robot demonstrations were used?"
```

The result contains:

- final answer
- retrieved evidence with page/section/source
- per-claim verifier verdicts
- abstention state and reason

## Evaluation

The required evaluation categories are:

1. answerable factual questions
2. misleading questions containing false premises
3. unanswerable questions

A starter schema and runner are in `evaluation/`.

The main comparison should separate:

- ordinary/single-path RAG baseline
- evidence-constrained answering
- evidence-constrained answering + heterogeneous cross verification

This is important because a system can appear to hallucinate less simply by refusing more questions.

## Project structure

```text
citetrace/
  paper/           PDF parsing and evidence store
  llm/             provider-agnostic LLM adapters
  retrieval/       cross-retrieval agents
  answer/          evidence-grounded answer agent
  verification/    independent claim verifiers
  aggregation/     disagreement-aware resolver
  pipeline.py      orchestration only
prompts/           reproducible prompt library
evaluation/        hallucination/abstention evaluation
tests/             unit tests
docs/              architecture contract
```

## Design principles

1. Evidence before answer.
2. No model verifies its own claim by default.
3. First-pass verifiers are independent.
4. Retrieval failure is not equivalent to “the paper does not contain this information.”
5. Abstention is a valid output.
6. Provider-specific SDK objects never enter the core pipeline.
7. New methods should implement existing interfaces rather than rewrite orchestration.

See `docs/ARCHITECTURE.md` before modifying the core interfaces.
