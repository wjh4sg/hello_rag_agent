from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
import sys
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CASES_PATH = Path(__file__).resolve().parent / "rag_eval_cases.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from hello_rag_agent import get_service
from hello_rag_agent.config import load_settings
from hello_rag_agent.knowledge_base import KnowledgeBase, SearchResult
from hello_rag_agent.llm import SafeHelloAgentsLLM

DEFAULT_JUDGE_MODEL = "qwen3.5-plus"
DEFAULT_JUDGE_TIMEOUT = 180
DEFAULT_JUDGE_MAX_TOKENS = 900
DEFAULT_JUDGE_RETRIES = 3
DEFAULT_EVIDENCE_PREVIEW_CHARS = 220


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    question: str
    goal: str


def load_cases(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for item in raw:
        cases.append(
            EvalCase(
                case_id=str(item["id"]),
                question=str(item["question"]),
                goal=str(item.get("goal", "")),
            )
        )
    return cases


def build_judge_prompt(
    *,
    question: str,
    goal: str,
    answer: str,
    evidence: str,
) -> str:
    return f"""你是一个严格但公正的 RAG 问答评估裁判。

请仅根据“用户问题”“系统回答”“检索证据”进行打分，不要使用外部知识脑补。

评分维度（1-5 分）：
1. groundedness：回答是否被检索证据支持，是否存在明显编造。
2. relevance：回答是否真正回应了问题，而不是泛泛而谈。
3. completeness：回答是否覆盖了问题需要的关键点。
4. clarity：表达是否清楚、结构是否易读。

请输出严格 JSON，不要输出任何额外解释：
{{
  "groundedness": 1-5 的整数,
  "relevance": 1-5 的整数,
  "completeness": 1-5 的整数,
  "clarity": 1-5 的整数,
  "summary": "一句中文总结",
  "strengths": ["优点1", "优点2"],
  "issues": ["问题1", "问题2"],
  "verdict": "pass 或 borderline 或 fail"
}}

用户问题：
{question}

评估目标：
{goal or "评估当前回答质量"}

系统回答：
{answer}

检索证据：
{evidence}
"""


def render_evidence(results: list[SearchResult]) -> str:
    if not results:
        return "没有检索到任何证据。"

    lines: list[str] = []
    for index, item in enumerate(results, start=1):
        preview = item.chunk.content.strip().replace("\r", " ").replace("\n", " ")
        if len(preview) > DEFAULT_EVIDENCE_PREVIEW_CHARS:
            preview = f"{preview[:DEFAULT_EVIDENCE_PREVIEW_CHARS]}..."
        lines.append(f"[Evidence {index}] source={item.chunk.source}")
        lines.append(f"title={item.chunk.title}")
        lines.append(f"score={item.score:.2f}")
        lines.append(f"content={preview}")
        lines.append("")
    return "\n".join(lines).strip()


def parse_judge_payload(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise ValueError("Judge response is empty.")

    if text.startswith("```"):
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("{") and part.endswith("}"):
                text = part
                break
            if "\n" in part:
                candidate = part.split("\n", 1)[1].strip()
                if candidate.startswith("{") and candidate.endswith("}"):
                    text = candidate
                    break

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Judge response is not valid JSON: {text}")
    return json.loads(text[start : end + 1])


def average_score(payload: dict[str, Any]) -> float:
    metrics = [
        int(payload["groundedness"]),
        int(payload["relevance"]),
        int(payload["completeness"]),
        int(payload["clarity"]),
    ]
    return round(mean(metrics), 2)


def invoke_judge_with_retries(
    *,
    judge_llm: SafeHelloAgentsLLM,
    prompt: str,
    retries: int = DEFAULT_JUDGE_RETRIES,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return judge_llm.invoke([{"role": "user", "content": prompt}])
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(2 * attempt, 5))
    assert last_error is not None
    raise last_error


def build_markdown_report(
    *,
    summary: dict[str, Any],
    cases: list[dict[str, Any]],
) -> str:
    lines = [
        "# RAG LLM Judge Evaluation Report",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Model under evaluation: {summary['model_under_test']}",
        f"- Judge model: {summary['judge_model']}",
        f"- Cases: {summary['case_count']}",
        f"- Average score: {summary['average_score']:.2f}/5",
        f"- Pass rate: {summary['pass_rate']:.2%}",
        f"- Borderline rate: {summary['borderline_rate']:.2%}",
        f"- Fail rate: {summary['fail_rate']:.2%}",
        "",
        "## Dimension Averages",
        "",
        f"- Groundedness: {summary['dimension_averages']['groundedness']:.2f}/5",
        f"- Relevance: {summary['dimension_averages']['relevance']:.2f}/5",
        f"- Completeness: {summary['dimension_averages']['completeness']:.2f}/5",
        f"- Clarity: {summary['dimension_averages']['clarity']:.2f}/5",
        "",
        "## Case Details",
        "",
    ]

    for case in cases:
        judge = case["judge"]
        lines.extend(
            [
                f"### {case['id']}",
                "",
                f"- Question: {case['question']}",
                f"- Goal: {case['goal']}",
                f"- Average score: {case['average_score']:.2f}/5",
                f"- Verdict: {judge['verdict']}",
                f"- Summary: {judge['summary']}",
                f"- Groundedness: {judge['groundedness']}/5",
                f"- Relevance: {judge['relevance']}/5",
                f"- Completeness: {judge['completeness']}/5",
                f"- Clarity: {judge['clarity']}/5",
                "- Strengths: " + "；".join(judge.get("strengths", [])),
                "- Issues: " + "；".join(judge.get("issues", [])),
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-judge evaluation for the local RAG agent.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH, help="Path to evaluation cases JSON.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of evidence chunks used for judging.")
    parser.add_argument(
        "--judge-model",
        default="",
        help="Optional judge model override. Defaults to qwen3.5-plus.",
    )
    args = parser.parse_args()

    settings = load_settings()
    service = get_service()
    knowledge_base = KnowledgeBase(
        settings.knowledge_base,
        api_key=settings.resolve_api_key(),
        base_url=settings.llm.base_url,
    )
    judge_model_name = args.judge_model or DEFAULT_JUDGE_MODEL
    judge_llm = SafeHelloAgentsLLM(
        model=judge_model_name,
        api_key=settings.resolve_api_key(),
        base_url=settings.llm.base_url,
        temperature=0.0,
        max_tokens=DEFAULT_JUDGE_MAX_TOKENS,
        timeout=DEFAULT_JUDGE_TIMEOUT,
    )

    cases = load_cases(args.cases)
    results: list[dict[str, Any]] = []

    for case in cases:
        answer, session_id = service.ask(case.question)
        evidence_results = knowledge_base.search(case.question, top_k=args.top_k)
        evidence_text = render_evidence(evidence_results)
        prompt = build_judge_prompt(
            question=case.question,
            goal=case.goal,
            answer=answer,
            evidence=evidence_text,
        )
        response = invoke_judge_with_retries(judge_llm=judge_llm, prompt=prompt)
        payload = parse_judge_payload(getattr(response, "content", ""))
        case_result = {
            "id": case.case_id,
            "question": case.question,
            "goal": case.goal,
            "session_id": session_id,
            "answer": answer,
            "evidence": [
                {
                    "source": item.chunk.source,
                    "title": item.chunk.title,
                    "score": item.score,
                    "content": item.chunk.content,
                }
                for item in evidence_results
            ],
            "judge": payload,
            "average_score": average_score(payload),
        }
        results.append(case_result)

    dimension_averages = {
        "groundedness": mean(item["judge"]["groundedness"] for item in results),
        "relevance": mean(item["judge"]["relevance"] for item in results),
        "completeness": mean(item["judge"]["completeness"] for item in results),
        "clarity": mean(item["judge"]["clarity"] for item in results),
    }
    verdicts = [item["judge"]["verdict"] for item in results]
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_under_test": settings.llm.model,
        "judge_model": judge_model_name,
        "case_count": len(results),
        "average_score": mean(item["average_score"] for item in results),
        "pass_rate": sum(verdict == "pass" for verdict in verdicts) / len(verdicts),
        "borderline_rate": sum(verdict == "borderline" for verdict in verdicts) / len(verdicts),
        "fail_rate": sum(verdict == "fail" for verdict in verdicts) / len(verdicts),
        "dimension_averages": dimension_averages,
    }

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = DEFAULT_OUTPUT_DIR / f"rag_llm_judge_{timestamp}.json"
    md_path = DEFAULT_OUTPUT_DIR / f"rag_llm_judge_{timestamp}.md"

    json_path.write_text(
        json.dumps({"summary": summary, "cases": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(
        build_markdown_report(summary=summary, cases=results),
        encoding="utf-8",
    )

    print(f"Evaluated {len(results)} cases")
    print(f"Average score: {summary['average_score']:.2f}/5")
    print(f"Pass rate: {summary['pass_rate']:.2%}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
