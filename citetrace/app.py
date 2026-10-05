from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataclasses import asdict

from citetrace.aggregation.conservative import ConservativeAggregator
from citetrace.answer.agent import EvidenceAnswerAgent
from citetrace.llm.factory import build_llm
from citetrace.paper.parser import parse_pdf
from citetrace.paper.store import JsonPaperStore
from citetrace.pipeline import CiteTracePipeline
from citetrace.retrieval.agent import LLMRetrievalAgent
from citetrace.verification.agent import LLMVerifier


def build_pipeline(store: JsonPaperStore) -> CiteTracePipeline:
    qwen = build_llm("qwen")
    deepseek = build_llm("deepseek")
    glm = build_llm("glm")

    return CiteTracePipeline(
        retrievers=[
            LLMRetrievalAgent("qwen_retriever", qwen, store),
            LLMRetrievalAgent("deepseek_retriever", deepseek, store),
        ],
        answer_agent=EvidenceAnswerAgent("qwen_answerer", qwen),
        verifiers=[
            LLMVerifier("deepseek_verifier", deepseek),
            LLMVerifier("glm_verifier", glm),
        ],
        aggregator=ConservativeAggregator(),
    )


def main() -> None:
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
        store.save(args.paper_id, chunks)
        print(json.dumps({"paper_id": args.paper_id, "chunks": len(chunks)}, ensure_ascii=False))
        return

    result = build_pipeline(store).run(args.question, args.paper_id)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
