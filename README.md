# CiteTrace

**CiteTrace** is a modular multi-agent system for factual question answering over research papers.

Its goal is not to answer every question. Its goal is to reduce **unsupported factual claims** by grounding answers in paper evidence, independently verifying atomic claims, and abstaining when the evidence is insufficient or conflicting.

## Core research question

Can heterogeneous multi-agent cross-checking improve the reliability of factual question answering over research papers compared with a single-model or ordinary RAG pipeline?

## Core pipeline

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

### 1. Cross Retrieval
Multiple retrieval agents independently analyze the same question, generate retrieval queries, and select evidence. Their evidence is merged into a shared evidence pool.

Purpose: reduce evidence misses caused by a single model's query formulation.

### 2. Evidence-grounded Answering
The answer agent may only answer from the supplied evidence pool. It must separate explicit statements, justified inference, and unknown information.

### 3. Atomic Claim Extraction
Generated answers are split into independently checkable factual claims.

### 4. Cross Verification
Multiple verifier agents independently judge the same claim against the same evidence. Verifiers do not see one another's verdicts.

Possible verdicts:
- `SUPPORTED`
- `PARTIALLY_SUPPORTED`
- `UNSUPPORTED`
- `UNKNOWN`

### 5. Disagreement-aware Resolution
Agreement allows an answer to pass. Conflicts trigger conservative qualification, re-checking, or abstention.

Agent disagreement is treated as an uncertainty signal rather than resolved by naive majority voting.

## Design principles

1. **Evidence before answer.**
2. **No model verifies its own claim by default.**
3. **Independent first-pass verification.**
4. **Retrieval failure is not equivalent to “the paper does not contain this information.”**
5. **Abstention is a valid output.**
6. **All major components expose stable interfaces so models or algorithms can be swapped independently.**

## Repository structure

```text
citetrace/
  schemas.py
  pipeline.py
  retrieval/
  answer/
  verification/
  aggregation/
  llm/
prompts/
evaluation/
tests/
```

The initial repository intentionally focuses on the core architecture. UI, deployment, databases, and other product details are secondary and should not leak into the core interfaces.
