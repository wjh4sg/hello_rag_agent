# 项目评测说明

这套评测不是只看“最后一句答得像不像”，而是按当前项目的真实链路拆成 4 层：

1. `retrieval`
   评估知识库检索是否命中正确来源与关键信号。
   适合回归分块、召回策略、向量库状态变化。

2. `memory`
   评估 `MemoryTool` 的工具级回忆能力。
   适合回归记忆打分、类型过滤、近期优先级。

3. `session`
   评估多轮对话后的会话记忆是否真的能被找回，并进入最终回答。
   适合回归“用户刚说过的话还记不记得”这类体验问题。

4. `answer`
   保留现有 `RAG + LLM judge` 评测。
   适合看最终回答的 groundedness、relevance、completeness、clarity。

## 为什么这样设计

当前项目的核心链路是：

- 本地知识库检索
- 会话级 MemoryTool
- ContextBuilder 拼装上下文
- ReActAgent 作答
- 必要时走 `rag_tool.ask` 兜底

如果只看最后回答，很难判断问题出在：

- 检索没命中
- 记忆没召回
- 会话上下文丢了
- 还是模型回答阶段幻觉

所以评测必须分层，才能真正支撑后续改造。

## 运行方式

开发态快跑，默认只跑轻量回归，不跑 LLM judge：

```bash
.venv\Scripts\python.exe evaluation\run_project_eval.py --profile dev
```

PowerShell 下也可以直接用快捷脚本：

```powershell
.\evaluation\run_dev_eval.ps1
```

如果你只想更快看 1 个样例，也可以再限制 case 数量：

```bash
.venv\Scripts\python.exe evaluation\run_project_eval.py --profile dev --max-cases 1
```

只跑离线能力评测：

```bash
.venv\Scripts\python.exe evaluation\run_project_eval.py --suites retrieval,memory
```

跑会话记忆评测：

```bash
.venv\Scripts\python.exe evaluation\run_project_eval.py --suites session
```

跑完整评测：

```bash
.venv\Scripts\python.exe evaluation\run_project_eval.py --suites retrieval,memory,session,answer
```

自定义 judge 模型：

```bash
.venv\Scripts\python.exe evaluation\run_project_eval.py --suites answer --judge-model qwen3.5-plus
```

## 输出内容

脚本会在 `evaluation/results/` 下生成：

- `project_eval_时间戳.json`
- `project_eval_时间戳.md`

其中：

- JSON 适合后续接 CI、画趋势图、做版本对比
- Markdown 适合人工查看和周报汇总

## 当前重点指标

`retrieval`:

- `source_hit_rate`
- `avg_keyword_hit_rate`
- `avg_first_hit_rank`

`memory`:

- `avg_substring_hit_rate`
- `top1_ok_rate`

`session`:

- `avg_memory_hit_rate`
- `avg_answer_hit_rate`
- `trace_leak_rate`

`answer`:

- `average_score`
- `groundedness / relevance / completeness / clarity`
- `trace_leak_rate`

`answer_smoke`:

- `avg_substring_hit_rate`
- `trace_leak_rate`
- `forbidden_hit_rate`

## 使用建议

1. 改检索前后，至少跑 `retrieval + answer`
2. 改记忆前后，至少跑 `memory + session`
3. 平时开发先跑 `--profile dev`，只有准备提交或看总分时再跑全量 `answer`
4. 如果 `answer` 退化，但 `retrieval` 没退化，优先排查回答阶段
5. 如果 `session` 退化，但 `memory` 正常，优先排查上下文拼装与会话链路
