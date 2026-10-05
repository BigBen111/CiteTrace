# CiteTrace Architecture Contract

## Pipeline

1. Cross Retrieval: two or more retrieval agents independently return Evidence objects; the pipeline merges and de-duplicates them.
2. Answering: one answer agent consumes only the question and shared evidence pool and returns an AnswerDraft with atomic factual claims.
3. Cross Verification: two or more verifier agents independently judge every atomic claim against the same evidence pool. First-pass verifiers do not see one another's verdicts.
4. Aggregation: disagreement is treated as uncertainty rather than naive majority voting. Unsupported or unknown claims must not silently enter the final answer.

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
