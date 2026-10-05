# CiteTrace: likely questions and design responses

## 1. "Isn't this just ChatPDF / RAG?"

**Risk:** If the system is only PDF -> retrieval -> generation, the criticism is valid.

**Design response:** Retrieval is only evidence acquisition. CiteTrace decomposes the generated answer into atomic factual claims and independently checks whether each claim is supported by the retrieved paper evidence before it may enter the final answer.

Core distinction:

```text
ordinary RAG: Retrieve -> Generate
CiteTrace:    Cross-Retrieve -> Generate -> Claim Split -> Cross-Verify -> Resolve
```

## 2. "The verifier is also an LLM. Why trust it?"

CiteTrace does not claim formal verification or guaranteed correctness. The verifier performs a narrower **claim-evidence consistency** task rather than answering the original research question.

The system hypothesis is empirical: independent heterogeneous checks may expose errors that a single model would pass through. This must be tested against a single-verifier baseline.

## 3. "Why verify after already instructing the answer model to use evidence?"

Prompt constraints are not guarantees. An answer model may still:
- strengthen a weak statement,
- alter a number,
- confuse simulation with real-world evaluation,
- convert an inference into an explicit paper claim.

CiteTrace therefore separates **generation constraint** from **post-generation checking**.

## 4. "What if retrieval misses the correct evidence?"

Retrieval failure is not treated as proof that the paper lacks the information.

CiteTrace uses independent cross-retrieval agents to reduce single-query formulation failures. If sufficient evidence still cannot be found, the proper output is `UNKNOWN` / abstention, not "the paper definitely does not say this."

## 5. "Why is the task limited to factual questions?"

This is deliberate. Evidence tracing and hallucination suppression are objectively testable when claims can be directly supported or contradicted by paper text.

The first version does not claim to solve:
- novelty assessment,
- whether a method is scientifically "good",
- open-ended research judgment,
- formal correctness of mathematical proofs.

A narrow capability boundary is part of the reliability design.

## 6. "How do citations prove that an answer is supported?"

A citation attached to a paragraph is insufficient. CiteTrace splits the answer into **atomic claims** and checks each claim against evidence.

```text
Question
  -> Answer
    -> Atomic Claim
      -> Evidence
        -> Support Verdict
```

The evidence chain is claim-level rather than paragraph-level.

## 7. "How do you prove hallucination is reduced?"

Evaluation must include:
1. normal answerable factual questions,
2. misleading questions containing false premises,
3. unanswerable questions.

The system should correct false premises and abstain when evidence is insufficient.

## 8. "Couldn't you get low hallucination by refusing everything?"

Yes. Therefore hallucination rate alone is not a valid metric.

Report both:
- correctness on answerable questions,
- abstention accuracy on unanswerable questions,

and manually audit unsupported factual claims.

The target behavior is:
- evidence sufficient -> answer,
- evidence contradicts premise -> correct it,
- evidence insufficient -> abstain.

## 9. "Why multiple models?"

Multiple models are not included merely for model comparison. They are part of the reliability mechanism.

Different models independently formulate retrieval queries and independently judge claim-evidence support. The design assumes their errors are not perfectly correlated.

A useful ablation is:
- single model,
- homogeneous multi-agent,
- heterogeneous multi-agent.

## 10. "Is this just spending more API calls for higher accuracy?"

This must be answered experimentally.

Required ablation ladder:

```text
single-path RAG
-> evidence-constrained RAG
-> single verifier
-> heterogeneous multi-verifier
-> cross-retrieval + heterogeneous multi-verifier
```

If an added stage does not improve reliability enough to justify its cost, it should not be claimed as useful.

## 11. "Why not majority vote?"

Support verification is asymmetric. One verifier finding a material contradiction should not be ignored simply because another verifier says "supported."

CiteTrace treats disagreement as an uncertainty signal. The default resolver is intentionally conservative rather than majority-vote based.

## One-sentence project definition

CiteTrace studies unsupported generation in research-paper factual QA by combining independent evidence retrieval, evidence-grounded answering, atomic claim decomposition, heterogeneous claim-evidence verification, and disagreement-aware abstention.
