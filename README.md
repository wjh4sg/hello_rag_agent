# Hello Agents RAG

这是一个基于 `hello_agents` 重构后的精简版 RAG 项目。

项目只保留三条主线：
- `hello_agents` 负责 Agent 和 Tool 编排
- 本地轻量知识库负责文档加载、切分和检索
- `Streamlit` 与 `FastAPI` 共用同一套服务层

## 目录结构

```text
.
├─ api/
├─ config/
├─ data/
├─ hello_rag_agent/
├─ prompts/
├─ app.py
├─ requirements.txt
└─ .env.example
```

## 运行前准备

1. 创建并激活虚拟环境
2. 安装依赖
3. 复制 `.env.example` 为 `.env`
4. 配置模型信息

推荐使用 DashScope 的 OpenAI 兼容接口：

```env
LLM_MODEL_ID=tongyi-xiaomi-analysis-pro
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=你的密钥
```

如果你已经有 `DASHSCOPE_API_KEY`，代码也会自动兜底读取。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动 Streamlit

```bash
streamlit run app.py
```

## 启动 FastAPI

```bash
python -m api.run_api
```

## 一键启动

Windows 下可以直接使用：

```bash
python start_services.py
```

或者：

```bash
start.bat
```

可选参数示例：

```bash
python start_services.py --host 0.0.0.0 --api-port 9000 --streamlit-port 8601
```

默认接口：
- `POST /api/chat`
- `POST /api/query`
- `GET /api/health`
- `POST /api/session/reset`
- `GET /api/session/{session_id}`
- `GET /api/knowledge/stats`

## Docker 启动

项目已经提供 `Dockerfile` 和 `docker-compose.yml`。

构建并启动：

```bash
docker compose up -d --build
```

默认 Docker 访问地址：
- Streamlit: `http://127.0.0.1:18501`
- FastAPI: `http://127.0.0.1:18000`
- FastAPI Docs: `http://127.0.0.1:18000/docs`

停止：

```bash
docker compose down
```

## 设计说明

- 不再依赖 `langchain`、`langgraph`、`chroma`
- 检索层改成本地轻量实现，便于教学和维护
- 所有业务入口统一在 `hello_rag_agent/agent_service.py`
