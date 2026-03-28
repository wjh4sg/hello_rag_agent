# 项目评估说明

这套评估不是只看“最后一句答得像不像”，而是按当前项目的真实链路拆成多层：

1. `retrieval`
   评估知识库检索是否命中正确来源、关键内容、可读 snippet 和引用信息。
2. `memory`
   评估 `MemoryTool` 的检索、排序、类型过滤和近期优先能力。
3. `session`
   评估多轮对话后，系统是否还能正确回忆当前会话和用户画像。
4. `fusion`
   评估“记忆是否真的参与了检索”，也就是记忆召回、检索 query 拼接、证据命中和最终回答是否一致。
5. `answer`
   保留 `RAG + LLM judge` 评估，同时加入规则化检查，避免只靠 judge 打分。
6. `answer_smoke`
   轻量回答冒烟评测，适合开发中快速回归。

## 为什么这样设计

当前项目的核心链路是：

- 本地知识库检索
- 用户级 + 会话级记忆
- 上下文拼装
- 直接检索回答 / Agent 回答

如果只看最后回答，很难判断问题出在：

- 检索没命中
- 记忆没召回
- 记忆没有正确影响检索
- 还是生成阶段把证据用坏了

所以评估必须分层。

## 运行方式

开发态快速回归：

```bash
.venv\Scripts\python.exe evaluation\run_project_eval.py --profile dev
```

只跑检索和记忆：

```bash
.venv\Scripts\python.exe evaluation\run_project_eval.py --suites retrieval,memory
```

只跑记忆-检索协同：

```bash
.venv\Scripts\python.exe evaluation\run_project_eval.py --suites fusion
```

只跑会话记忆：

```bash
.venv\Scripts\python.exe evaluation\run_project_eval.py --suites session
```

跑完整评估：

```bash
.venv\Scripts\python.exe evaluation\run_project_eval.py --suites retrieval,memory,session,fusion,answer
```

限制 case 数量做快速检查：

```bash
.venv\Scripts\python.exe evaluation\run_project_eval.py --profile dev --max-cases 1
```

## 输出内容

脚本会在 `evaluation/results/` 下生成：

- `project_eval_时间戳.json`
- `project_eval_时间戳.md`

其中：

- JSON 适合后续做版本对比、自动化接入、画趋势图
- Markdown 适合人工查看和答辩展示

## 当前重点指标

`retrieval`

- `source_hit_rate`
- `avg_keyword_hit_rate`
- `avg_snippet_keyword_hit_rate`
- `avg_citation_coverage_rate`
- `avg_source_diversity`
- `avg_first_hit_rank`

`memory`

- `avg_substring_hit_rate`
- `top1_ok_rate`

`session`

- `avg_memory_hit_rate`
- `avg_answer_hit_rate`
- `trace_leak_rate`

`fusion`

- `avg_memory_hit_rate`
- `avg_query_hit_rate`
- `source_hit_rate`
- `avg_keyword_hit_rate`
- `avg_answer_hit_rate`

`answer`

- `average_score`
- `groundedness / relevance / completeness / clarity`
- `rule_pass_rate`
- `avg_substring_hit_rate`
- `forbidden_hit_rate`
- `trace_leak_rate`

`answer_smoke`

- `avg_substring_hit_rate`
- `forbidden_hit_rate`
- `trace_leak_rate`

## 使用建议

1. 改检索逻辑后，至少跑 `retrieval + fusion + answer`
2. 改记忆逻辑后，至少跑 `memory + session + fusion`
3. 平时开发优先跑 `--profile dev`
4. 如果 `fusion` 退化但 `retrieval` 没退化，优先排查“记忆是否正确参与检索”
5. 如果 `answer` 退化但 `retrieval` 和 `fusion` 正常，优先排查回答生成阶段
