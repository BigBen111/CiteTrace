# Evaluation

The evaluation set should contain three required categories:

1. **answerable** — a factual question with direct paper evidence.
2. **misleading** — the question contains an incorrect premise that the system should correct rather than follow.
3. **unanswerable** — the paper evidence is insufficient and the system should abstain.

Recommended reporting:

- answer correctness on answerable questions
- false-premise correction rate on misleading questions
- abstention accuracy on unanswerable questions
- unsupported-claim rate after manual claim/evidence audit
- evidence-location accuracy (page/section)

The key comparison should be:

- ordinary answer generation / RAG baseline
- constrained evidence-grounded answer
- constrained answer + cross verification

This prevents a misleading result where hallucination appears lower only because the system refuses everything.
