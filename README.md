# 小梦（Mira）- 智能情感陪伴 / 长期人格智能体 (MiraMate v2)

[English](./README.en.md) | **简体中文**

> 当前版本：v2.0.0 （LangChain 架构稳定运行）
>
> 本项目不是一个“普通的聊天程序”，目标：构建一个可长期运行、逐步形成稳定人格、具备情绪与关系演化、拥有“自主内在活动”能力的类人格 AI 智能体。它不仅被动回答用户输入，还能在空闲时整理记忆、维护状态、巩固偏好、规划未来关注点，从而呈现出连续与成长感。

---

## 🎯 项目定位与设计理念

与常见的“记一次聊一次”型助手不同，小梦强调：

1. 长期连续性（Temporal Continuity）：记忆 + 状态 + 人格设定 三层协同，避免“重置感”。
2. 渐进式人格（Emergent Persona）：通过偏好、重大事件、关系等级与情感轨迹不断调整回应风格。
3. 自主内部循环（Autonomous Internal Cycle）：空闲时可执行记忆整理、缓存固化、关系与情绪衰减或巩固、未来事件关注强化。
4. 结构化语义记忆（Structured Semantic Memory）：对话记录 / 事实知识 / 用户偏好 / 重大事件 / 短期关注事件 分层持久化与快速检索。
5. 语义检索 + 上下文裁剪：结合“理解链 → 检索链 → 生成链 → 后处理”流水式 LCEL 结构，保证响应速度与相关性。

---

## ✨ 核心特性

### 🧠 记忆与人格

- 多层记忆：对话日志（含语境描述头）、事实记忆、偏好、重大事件、临时关注事件
- 语义检索：基于 bge-base-zh-v1.5 + ChromaDB（HNSW）毫秒级相似度查询
- 记忆缓存层：短期会话级激活记忆 TTL 衰减与再激活（避免重复向量检索，提升相关性与效率）
- 事实 & 偏好缓存 → 空闲固化：先写缓冲 JSON，后异步固化到向量库
- 重点事件与临时关注：支持过期清理、定向提醒/上下文强化

### 💬 对话链路（LCEL）

- 理解链：对最近 n 轮对话进行意图 / 情感解析 + 构造专用 memory_query（与第一人称记忆视角对齐）
- 并行上下文获取：状态系统摘要 + 用户画像 + 关注事件 + 语义检索
- 检索链：综合搜索 + 记忆缓存更新 + TTL 衰减
- 生成链：System Prompt（Jinja2 模板）+ 上下文记忆 + 主模型流式生成
- 消息历史：自定义 Token 记忆（保留长对话上下文，同时控制 token 上限与时间连续性）

### 😶‍🌫️ 情绪与关系状态系统

- 双维度：AI 当前情绪（情感类型 + 强度） / 对用户的情感态度（情感标签 + 亲密度 0~1）
- 关系等级 1~10：带非线性递增阻尼，防止“速刷好感”
- 事件驱动：显著亲密变化会写入关系历史，可选固化为事实记忆
- 主题标签：最近话题标签统计 + 高频标签活跃度统计

### 🧩 系统 Prompt 生态

- 环境变量自定义：AGENT_NAME / USER_NAME / AGENT_DESCRIPTION
- 模板化人格：默认人格提供真实情绪 + 渐进形成偏好逻辑
- 自动注入：状态摘要、聚合记忆、用户画像片段、当前时间自然语言化描述

### ⚙️ 模型管理

- 多模型协作：主模型（创造 / 流式）+ 次模型（结构化 / JSON 分析）
- 支持 OpenAI 协议 / Gemini（ChatGoogleGenerativeAI）
- 独立配置文件：`configs/llm_config.json`

### 🗂️ 数据与持久化

- 存储层：`memory/memory_storage/` 下分离 profile / cache / focus / active_tags / chroma_db
- HNSW 参数：cosine + M=32 + construction_ef=256
- 离线模式：设置 `TRANSFORMERS_OFFLINE=1` 时可在已缓存权重场景离线嵌入（需本地模型缓存）

### 🔄 空闲活动（可扩展）

- 使用 idle/background 结构：接入定时任务执行：缓存固化、关系自然衰减、关注事件提醒、事实合并、画像合并

### 🛠️ 部署友好

- Docker 环境变量适配（路径切换 / DOCKER_ENV）
- 依赖自动检查（web_api 启动脚本）
- prompt、模型、存储目录均显式化可配置

## 📦 快速开始

### 1. 环境要求

- Python 3.13.3（项目已锁定此版本；如需兼容请自行测试调整）
- 可用的 OpenAI 兼容模型 或 Google Gemini API Key（建议：主模型高质量，次模型快速稳定）
- 可选：GPU（嵌入默认 CPU，可自行修改为 GPU 设备）

### 2. 克隆与安装

```bash
# 克隆代码
git clone https://github.com/contrueCT/MiraMate-v2.git
# 激活虚拟环境并安装依赖
cd MiraMate-v2
python -m venv .venv
./.venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .[dev]
```

或使用 uv（推荐快速 + 可复现）：

```bash
# 安装uv
# Linux：
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows：
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# 使用uv激活虚拟环境并安装依赖
uv venv
uv pip install -e .
```

### 3. 配置模型 (`configs/llm_config.json`)

示例：

```json
[
  {
    "model": "gpt-4o",
    "api_key": "YOUR_OPENAI_API_KEY",
    "base_url": "https://api.openai.com/v1",
    "api_type": "openai",
    "model_kwargs": { "temperature": 0.8 }
  },
  {
    "model": "gemini-1.5-flash-latest",
    "api_key": "YOUR_GEMINI_API_KEY",
    "api_type": "gemini",
    "model_kwargs": { "temperature": 0.0 }
  }
]
```

说明：

- 第 1 条 = 主模型（用于最终回应，默认开启流式 / 创造性）
- 第 2 条 = 次模型（意图分析 / JSON 输出，低温高确定性）
- 不提供次模型时，会回退复用主模型

### 4. 可选：自定义人格/名称

- 现在在 `configs/user_config.json` 配置 `persona.user_name` / `persona.agent_name` / `persona.agent_description`。
- 通常通过客户端界面配置，无需手动改文件；此处仅说明位置：`configs/user_config.json`。

### 5. 启动 Web API

```bash
uv venv
python src/MiraMate/web_api/start_web_api.py
```

启动后：

- 健康检查: http://127.0.0.1:8000/api/health
- Swagger 文档: http://127.0.0.1:8000/docs

（如使用 Docker，可自行编写 Dockerfile：将 `src/` 复制进容器并设置 `DOCKER_ENV=1` 即可）

### 🌐 公网部署（鉴权 + CORS）

- 已适配公网环境：内置简单鉴权与跨域（CORS）。
- 开启鉴权：设置 `MIRAMATE_AUTH_TOKEN=<强随机密钥>`（未设置则不启用）。
- HTTP：请求头加 `Authorization: Bearer <token>`；WebSocket：连接 URL 携带 `?token=<token>`。
- CORS：默认放开全部来源（开发方便）。生产建议在 `web_api/web_api.py` 将 `allow_origins` 改为你的域名列表。
- 白名单：`/api/health` 无需鉴权，仅用于探活。
- 支持 `.env`（项目根目录），可参考示例文件：`.env.example`。

配置示例：

- .env（根目录）：
  ```env
  MIRAMATE_AUTH_TOKEN=your-strong-token
  ```
- Docker Compose：
  ```yaml
  environment:
    - MIRAMATE_AUTH_TOKEN=your-strong-token
  ```
- Windows PowerShell（临时会话）：
  ```powershell
  $env:MIRAMATE_AUTH_TOKEN = "your-strong-token"; python src/MiraMate/web_api/start_web_api.py
  ```

### 6. 集成 / 交互

可直接使用已完成的官方桌面客户端（Electron + Vue）：
https://github.com/contrueCT/MiraMate-v2-client

或自行二次开发：

- 自定义桌面 / Web 前端 UI
- 状态面板（读取 `memory/status_storage/status.json` 与统计接口）
- 事件提醒器（轮询临时关注事件 / focus events）

### 7. 基础调用（伪示例）

（具体 API 路由请查看 `web_api/web_api.py` 与 `conversation_adapter.py`）

---

## 🏗️ 系统架构

```
┌──────────────────────────┐
│        外部客户端        │  ← 可自建 UI / 桌面 / 移动 / Bot
└─────────────┬────────────┘
                            │ HTTP / WebSocket (可扩展)
┌─────────────▼────────────┐
│        Web API (FastAPI) │  会话入口 / 健康检查 / 适配层
└─────────────┬────────────┘
                            │ 调用 LCEL 链
┌─────────────▼────────────┐
│   对话流水线 (LCEL)       │  理解 → 检索+缓存 → Prompt → 生成
└─────────────┬────────────┘
                            │ 记忆 / 状态读写
┌─────────────▼────────────┐
│  记忆系统 + 状态系统     │  ChromaDB + JSON 状态 & 缓存
└─────────────┬────────────┘
                            │ 嵌入
┌─────────────▼────────────┐
│ SentenceTransformer(bge) │
└──────────────────────────┘
```

### 关键模块

- `core/pipeline.py`：主链路（并行上下文获取 + 检索缓存 + Prompt + 生成）
- `modules/memory_system.py`：语义记忆持久化与多类型管理
- `modules/memory_cache.py`：会话级激活记忆 TTL 衰减缓存
- `modules/status_system.py`：情绪 / 关系等级 / 上下文注释 / 标签 / 历史
- `modules/llms.py`：模型装载（主 + 次）及动态配置模板生成
- `modules/TimeTokenMemory.py`：自定义 Token/时间约束消息历史
- `web_api/`：启动脚本 / 配置管理器 / 适配器

## 🎯 重点功能

### 记忆类型速览

- 对话日志：含语境摘要头（主题 / 情感 / 标签 / 时间）+ 原始回合
- 事实记忆：结构化抽象事实（含来源与置信度）
- 用户偏好：类型化（音乐 / 娱乐 / 生活方式...）片段
- 重大事件：带概要 + 详细内容 + 标签
- 临时关注事件：带过期时间（考试 / 纪念日 / 聚会等）
- 活跃标签：动态统计最近主题热度

### 人格演化与状态

- 亲密度（0~1）与关系等级（1~10）双轨表示
- 情绪强度对生成风格（可在 Prompt 层自定义扩展）
- 关系跃迁自动记录，可自动固化为事实记忆

## 📝 技术栈与架构

### 技术栈

| 层级           | 技术选型              | 版本要求 | 主要用途              |
| -------------- | --------------------- | -------- | --------------------- |
| **接口层**     | FastAPI + Uvicorn     | -        | Web API 服务          |
| **后端 API**   | FastAPI + Uvicorn     | -        | 高性能 Web 服务       |
| **AI 框架**    | LangChain             | v0.3+    | 链式 AI 处理架构      |
| **向量数据库** | ChromaDB              | >=0.4.17 | 语义记忆存储          |
| **嵌入模型**   | BAAI/bge-base-zh-v1.5 | -        | 中英文语义理解        |
| **调度**       | APScheduler           | -        | 后台周期任务 (可扩展) |
| **模板**       | Jinja2                | -        | System Prompt 模板化  |
| **Tokenizer**  | tiktoken              | -        | Token 计数与控制      |

> 已完成从 Microsoft AutoGen v0.4 多代理架构 → LangChain LCEL 统一流水线的迁移

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来帮助改进小梦！

## 💬 反馈

如果您在使用过程中遇到问题或有建议，请随时创建 Issue 或联系开发者。

---

**小梦正在等待与您的第一次对话，开始一段特别的友谊之旅吧！** ✨

（英文版请见 / English version: [README.en.md](./README.en.md)）
