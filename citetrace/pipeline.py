from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Callable, Iterable, Sequence

from citetrace.aggregation.base import BaseAggregator
from citetrace.answer.base import BaseAnswerAgent
from citetrace.citation import CitationScanResult, extract_citation_target
from citetrace.retrieval.base import BaseRetriever
from citetrace.schemas import Evidence, FinalResult, VerificationResult, Verdict
from citetrace.tracing import TraceCollector
from citetrace.verification.base import BaseVerifier


logger = logging.getLogger(__name__)


class CiteTracePipeline:
    """Model-agnostic orchestration for CiteTrace's first architecture."""

    def __init__(
        self,
        retrievers: Sequence[BaseRetriever],
        answer_agent: BaseAnswerAgent,
        verifiers: Sequence[BaseVerifier],
        aggregator: BaseAggregator,
        max_evidence: int = 12,
        max_workers: int = 4,
        trace_collector: TraceCollector | None = None,
        citation_scanner: Callable[[str, str], CitationScanResult] | None = None,
        evidence_validator: Callable[[str, Evidence], bool] | None = None,
        enforce_evidence_links: bool = False,
    ) -> None:
        if not retrievers:
            raise ValueError("CiteTrace expects at least one retriever.")
        if not verifiers:
            raise ValueError("CiteTrace expects at least one verifier.")
        self.retrievers = retrievers
        self.answer_agent = answer_agent
        self.verifiers = verifiers
        self.aggregator = aggregator
        self.max_evidence = max_evidence
        self.max_workers = max(1, max_workers)
        self.trace_collector = trace_collector
        self.citation_scanner = citation_scanner
        self.evidence_validator = evidence_validator
        self.enforce_evidence_links = enforce_evidence_links

    def trace_snapshot(self) -> list[dict]:
        return self.trace_collector.snapshot() if self.trace_collector else []

    def run(
        self,
        question: str,
        paper_id: str,
        progress: Callable[[str, str], None] | None = None,
    ) -> FinalResult:
        self._set_progress(progress)
        _emit(progress, "question_routing", "正在判断问题应走规则扫描还是 Agent QA")
        citation_target = extract_citation_target(question)
        self._trace(
            kind="preprocessing",
            agent="question_router",
            stage="routing",
            output={
                "question": question,
                "route": "citation_rule" if citation_target else "agentic_qa",
                "citation_target": citation_target,
            },
        )
        if citation_target and self.citation_scanner is not None:
            _emit(progress, "citation_scan", f"正在全文扫描引用名称 {citation_target}")
            scan = self.citation_scanner(paper_id, citation_target)
            _emit(
                progress,
                "evidence_validation",
                f"正在校验 {len(scan.evidences)} 条引用扫描结果及页面覆盖",
            )
            result = _resolve_citation_scan(scan)
            self._trace(
                kind="preprocessing",
                agent="citation_rule",
                stage="citation_scan",
                output={
                    "target": scan.target,
                    "coverage": scan.coverage,
                    "evidences": [asdict(item) for item in scan.evidences],
                    "decision": asdict(result),
                },
            )
            _emit(progress, "completed", "引用扫描与原文校验已完成")
            return result

        evidence_pool = self._cross_retrieve(question, paper_id, progress)
        _emit(progress, "answer_generation", "Answer Agent 正在基于校验证据生成回答")
        draft = self.answer_agent.answer(question, evidence_pool)
        self._trace(
            kind="agent_output",
            agent=self.answer_agent.name,
            stage="answer",
            output=asdict(draft),
        )
        _emit(
            progress,
            "claim_binding",
            f"正在检查 {len(draft.claims)} 条 Claim 与 Evidence ID 的绑定关系",
        )

        verifications = self._cross_verify(
            draft.claims,
            draft.claim_evidence_ids,
            evidence_pool,
            progress,
        )

        _emit(progress, "aggregation", "正在汇总验证结果并执行保守决策")
        result = self.aggregator.resolve(draft, evidence_pool, verifications)
        self._trace(
            kind="agent_output",
            agent=type(self.aggregator).__name__,
            stage="aggregation",
            output=asdict(result),
        )
        _emit(progress, "completed", "回答、证据与交叉验证已生成")
        return result

    def _cross_retrieve(
        self,
        question: str,
        paper_id: str,
        progress: Callable[[str, str], None] | None = None,
    ) -> list[Evidence]:
        _emit(progress, "direct_retrieval", "正在执行实体、全文与 BM25 直接检索")
        results: list[Sequence[Evidence]] = [[] for _ in self.retrievers]
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(self.retrievers)),
            thread_name_prefix="citetrace-retrieval",
        ) as executor:
            futures = {
                executor.submit(retriever.retrieve, question, paper_id): (index, retriever)
                for index, retriever in enumerate(self.retrievers)
            }
            for future in as_completed(futures):
                index, retriever = futures[future]
                try:
                    results[index] = future.result()
                    self._trace(
                        kind="agent_output",
                        agent=retriever.name,
                        stage="retrieval",
                        output={
                            "evidence_count": len(results[index]),
                            "evidences": [asdict(item) for item in results[index]],
                        },
                    )
                except Exception as exc:
                    logger.warning("Retriever %s failed: %s", retriever.name, exc)
                    results[index] = []
                    self._trace(
                        kind="agent_output",
                        agent=retriever.name,
                        stage="retrieval",
                        status="error",
                        error=str(exc)[:1000],
                        output={"evidence_count": 0, "evidences": []},
                    )

        pooled = [evidence for result in results for evidence in result]
        ranked = _deduplicate_evidence(pooled)
        evidence_pool = _prioritize_for_question(question, ranked)[: self.max_evidence]
        _emit(
            progress,
            "evidence_validation",
            f"正在去重并回查 {len(evidence_pool)} 条候选摘录",
        )
        for index, evidence in enumerate(evidence_pool, 1):
            evidence.evidence_id = f"E{index}"
            evidence.match_type = evidence.match_type or "lexical"
            if self.evidence_validator is not None:
                evidence.validated = self.evidence_validator(paper_id, evidence)
        self._trace(
            kind="preprocessing",
            agent="evidence_table",
            stage="evidence_validation",
            output={
                "pooled_count": len(pooled),
                "deduplicated_count": len(ranked),
                "final_count": len(evidence_pool),
                "evidences": [asdict(item) for item in evidence_pool],
            },
        )
        return evidence_pool

    def _cross_verify(
        self,
        claims: Sequence[str],
        claim_evidence_ids: dict[str, list[str]],
        evidence_pool: Sequence[Evidence],
        progress: Callable[[str, str], None] | None,
    ) -> list[VerificationResult]:
        _emit(progress, "source_validation", "正在校验 Claim 与原文 Evidence ID 的绑定")
        evidence_by_id = {
            item.evidence_id: item for item in evidence_pool if item.evidence_id
        }
        tasks = [
            (
                claim,
                verifier,
                [
                    evidence_by_id[evidence_id]
                    for evidence_id in claim_evidence_ids.get(claim, [])
                    if evidence_id in evidence_by_id
                    and evidence_by_id[evidence_id].validated is not False
                ],
            )
            for claim in claims
            for verifier in self.verifiers
        ]
        if not tasks:
            return []

        ordered: list[VerificationResult | None] = [None] * len(tasks)
        completed = 0
        runnable_tasks: list[
            tuple[int, str, BaseVerifier, list[Evidence]]
        ] = []
        for index, (claim, verifier, linked_evidence) in enumerate(tasks):
            if self.enforce_evidence_links and not linked_evidence:
                ordered[index] = VerificationResult(
                    claim=claim,
                    verdict=Verdict.UNKNOWN,
                    confidence=0.0,
                    reason="The answer agent did not bind this claim to validated evidence IDs.",
                    verifier=verifier.name,
                    evidence_ids=[],
                )
                self._trace(
                    kind="agent_output",
                    agent=verifier.name,
                    stage="verification_skipped",
                    output=asdict(ordered[index]),
                )
                completed += 1
                continue
            runnable_tasks.append(
                (index, claim, verifier, linked_evidence or list(evidence_pool))
            )

        if not runnable_tasks:
            return [result for result in ordered if result is not None]

        _emit(
            progress,
            "semantic_verification",
            f"Evidence Verifier 正在核查 {len(runnable_tasks)} 个 Claim-Evidence 组合",
        )
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(runnable_tasks)),
            thread_name_prefix="citetrace-verification",
        ) as executor:
            futures = {}
            for index, claim, verifier, verification_evidence in runnable_tasks:
                future = executor.submit(verifier.verify, claim, verification_evidence)
                futures[future] = (
                    index,
                    claim,
                    verifier,
                    verification_evidence,
                )
            for future in as_completed(futures):
                index, claim, verifier, verification_evidence = futures[future]
                try:
                    ordered[index] = future.result()
                    self._trace(
                        kind="agent_output",
                        agent=verifier.name,
                        stage="verification",
                        output=asdict(ordered[index]),
                    )
                except Exception as exc:
                    logger.warning(
                        "Verifier %s failed for claim %r: %s",
                        verifier.name,
                        claim,
                        exc,
                    )
                    ordered[index] = VerificationResult(
                        claim=claim,
                        verdict=Verdict.UNKNOWN,
                        confidence=0.0,
                        reason=f"Verifier unavailable: {str(exc)[:160]}",
                        verifier=verifier.name,
                        evidence_ids=[
                            item.evidence_id
                            for item in verification_evidence
                            if item.evidence_id
                        ],
                    )
                    self._trace(
                        kind="agent_output",
                        agent=verifier.name,
                        stage="verification",
                        status="error",
                        error=str(exc)[:1000],
                        output=asdict(ordered[index]),
                    )
                completed += 1
                _emit(
                    progress,
                    "semantic_verification",
                    f"并行核查已完成 {completed}/{len(tasks)} 项",
                )
        return [result for result in ordered if result is not None]

    def _set_progress(
        self, progress: Callable[[str, str], None] | None
    ) -> None:
        for component in [*self.retrievers, self.answer_agent, *self.verifiers]:
            setter = getattr(component, "set_progress", None)
            if callable(setter):
                setter(progress)

    def _trace(self, *, kind: str, agent: str, **details) -> None:
        if self.trace_collector is not None:
            self.trace_collector.record(kind=kind, agent=agent, **details)


def _emit(
    progress: Callable[[str, str], None] | None,
    stage: str,
    message: str,
) -> None:
    if progress is not None:
        progress(stage, message)


def _deduplicate_evidence(items: Iterable[Evidence]) -> list[Evidence]:
    result: dict[tuple[str, int | None], Evidence] = {}
    for item in items:
        key = (item.source or " ".join(item.text.split()), item.page)
        current = result.get(key)
        if current is None:
            result[key] = item
            continue
        current.score = max(current.score or 0.0, item.score or 0.0)
        agents = {
            agent
            for value in (current.agent, item.agent)
            for agent in (value or "").split(",")
            if agent
        }
        current.agent = ",".join(sorted(agents)) or None
    return sorted(
        result.values(), key=lambda item: item.score or 0.0, reverse=True
    )


def _prioritize_for_question(question: str, items: list[Evidence]) -> list[Evidence]:
    """For citation questions, require an exact proper-name anchor when one exists."""
    if not re.search(r"引用|参考文献|\bcit(?:e|ed|es|ation)\b|\breference", question, re.I):
        return items
    entities = re.findall(r"[A-Za-z][A-Za-z0-9._+\-/]{2,}", question)
    if not entities:
        return items

    exact: list[Evidence] = []
    for item in items:
        for entity in entities:
            pattern = rf"(?<![A-Za-z0-9._+\-/]){re.escape(entity)}(?![A-Za-z0-9._+\-/])"
            if re.search(pattern, item.text, re.I):
                exact.append(item)
                break
    return exact or items


def _resolve_citation_scan(scan: CitationScanResult) -> FinalResult:
    citation_hits = [item for item in scan.evidences if item.relation == "cites"]
    mention_hits = [item for item in scan.evidences if item.relation == "mentions"]
    if citation_hits:
        citations = ", ".join(
            f"[{item.evidence_id}]" for item in citation_hits[:3] if item.evidence_id
        )
        pages = sorted({item.page for item in citation_hits if item.page is not None})
        return FinalResult(
            answer=(
                f"是。参考文献中发现了 {scan.target}，位于第 "
                f"{', '.join(map(str, pages))} 页 {citations}。"
            ),
            evidences=scan.evidences,
            verifications=[],
            abstained=False,
            note="由确定性全文扫描和原文摘录校验得出，未调用语言模型。",
            workflow="citation_rule",
            coverage=scan.coverage,
        )
    if mention_hits:
        mentions = ", ".join(
            f"[{item.evidence_id}]" for item in mention_hits[:3] if item.evidence_id
        )
        return FinalResult(
            answer=(
                f"正文中发现了 {scan.target} 的名称 {mentions}，但没有在已识别的"
                "参考文献区域找到对应条目，因此暂不能确认其是否构成正式引用。"
            ),
            evidences=scan.evidences,
            verifications=[],
            abstained=True,
            note="发现正文提及，但缺少经过校验的参考文献条目。",
            workflow="citation_rule",
            coverage=scan.coverage,
        )

    total_pages = scan.coverage.get("total_pages", 0)
    pages_with_text = scan.coverage.get("pages_with_text", 0)
    reference_pages = scan.coverage.get("reference_pages", [])
    if scan.coverage.get("complete_for_absence"):
        return FinalResult(
            answer=(
                f"未发现。已扫描成功解析的 {pages_with_text}/{total_pages} 页正文及"
                f"参考文献区域（第 {', '.join(map(str, reference_pages))} 页），"
                f"没有发现 {scan.target} 的精确或标准化名称匹配。"
            ),
            evidences=[],
            verifications=[],
            abstained=False,
            note="这是带扫描覆盖证明的未命中结果，未调用语言模型。",
            workflow="citation_rule",
            coverage=scan.coverage,
        )
    return FinalResult(
        answer=(
            f"当前已解析 {pages_with_text}/{total_pages} 页，但参考文献或页面覆盖不完整，"
            f"无法可靠判断是否引用 {scan.target}。"
        ),
        evidences=[],
        verifications=[],
        abstained=True,
        note="PDF 文本覆盖不足，不能把未检索到名称解释为未引用。",
        workflow="citation_rule",
        coverage=scan.coverage,
    )
