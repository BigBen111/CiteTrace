from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from collections.abc import Mapping
from functools import partial

from dotenv import load_dotenv

from citetrace.aggregation.conservative import ConservativeAggregator
from citetrace.answer.agent import EvidenceAnswerAgent
from citetrace.llm.factory import build_llm
from citetrace.llm.fallback import FallbackLLM
from citetrace.llm.validation import validate_output
from citetrace.paper.parser import build_pdf_metadata, parse_pdf
from citetrace.paper.store import JsonPaperStore
from citetrace.pipeline import CiteTracePipeline
from citetrace.retrieval.agent import LLMRetrievalAgent
from citetrace.tracing import TraceCollector, TracingLLM
from citetrace.verification.agent import LLMVerifier


def build_pipeline(
    store: JsonPaperStore, config: Mapping[str, str] | None = None
) -> CiteTracePipeline:
    qwen = build_llm("qwen", config)
    deepseek = build_llm("deepseek", config)
    glm = build_llm("glm", config)
    trace = TraceCollector()
    retrieval_llm = FallbackLLM(
        "qwen_deepseek_retrieval_fallback",
        [
            TracingLLM(qwen, "hybrid_retriever.primary", trace),
            TracingLLM(deepseek, "hybrid_retriever.fallback", trace),
        ],
        progress_stage="retrieval_fallback",
        validator=partial(validate_output, kind="retrieval"),
    )
    verifier_llm = FallbackLLM(
        "glm_deepseek_verifier_fallback",
        [
            TracingLLM(glm, "evidence_verifier.primary", trace),
            TracingLLM(deepseek, "evidence_verifier.fallback", trace),
        ],
        progress_stage="verification_fallback",
        validator=partial(validate_output, kind="verification"),
    )

    return CiteTracePipeline(
        retrievers=[
            LLMRetrievalAgent(
                "hybrid_retriever",
                retrieval_llm,
                store,
                trace_collector=trace,
            ),
        ],
        answer_agent=EvidenceAnswerAgent(
            "qwen_answerer",
            FallbackLLM(
                "qwen_deepseek_answer_fallback",
                [
                    TracingLLM(qwen, "qwen_answerer.primary", trace),
                    TracingLLM(deepseek, "qwen_answerer.fallback", trace),
                ],
                progress_stage="answer_fallback",
                validator=partial(validate_output, kind="answer"),
            ),
        ),
        verifiers=[
            LLMVerifier(
                "evidence_verifier",
                verifier_llm,
            ),
        ],
        aggregator=ConservativeAggregator(),
        max_workers=int((config or {}).get("CITETRACE_PIPELINE_MAX_WORKERS", "4")),
        trace_collector=trace,
        citation_scanner=store.scan_citation,
        evidence_validator=store.validate_evidence,
        enforce_evidence_links=True,
    )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="CiteTrace CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest")
    ingest.add_argument("pdf")
    ingest.add_argument("--paper-id", required=True)

    ask = sub.add_parser("ask")
    ask.add_argument("--paper-id", required=True)
    ask.add_argument("question")

    args = parser.parse_args()
    store = JsonPaperStore()

    if args.command == "ingest":
        chunks = parse_pdf(Path(args.pdf), args.paper_id)
        store.save(args.paper_id, chunks, build_pdf_metadata(Path(args.pdf), chunks))
        print(json.dumps({"paper_id": args.paper_id, "chunks": len(chunks)}, ensure_ascii=False))
        return

    result = build_pipeline(store).run(args.question, args.paper_id)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
