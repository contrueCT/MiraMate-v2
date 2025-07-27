# src/MiraMate/core/post_sync_chain.py

import json
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any # <-- 新增导入 Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# 导入我们需要的模块
from MiraMate.modules.llms import small_llm
from MiraMate.modules.status_system import update_status, get_status_summary

# --- 3. 定义 Prompt 模板 (保持不变) ---
STATE_ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    """
# 指令
你是一个专业的对话分析师，你的任务是根据对话内容，生成一个用于更新系统状态的JSON对象。
你必须严格按照下面“输出格式与示例”部分的要求来构造你的输出。

# 当前AI与用户状态（分析前）
{current_state}

# 最近的对话历史
{conversation_history}

# 最新一轮对话
用户: {user_input}
AI: {ai_response}

# ----------- 输出格式与示例 (这是最重要的部分！) -----------
你的输出必须是一个JSON对象。对于 'ai_status', 'user_status', 'context_notes' 这三个键，它们的值也必须是JSON对象（字典），而不能是描述性的字符串。

## 示例 1: 如果AI心情变好，用户心情也变好
{{
    "ai_status": {{
        "emotion": {{"mood": "开心", "strength": 0.8}}
    }},
    "user_status": {{
        "current_mood": "激动"
    }}
}}

## 示例 2: 如果只需要更新对话风格
{{
    "context_notes": {{
        "conversation_style": "技术讨论"
    }}
}}

## 示例 3: 如果你认为什么都不需要更新
{{}}

**规则**:
- **不要**在值的位置放入任何分析性的长句子或总结。
- **必须**提供嵌套的JSON对象作为值。
- 如果某个字段或类别不需要更新，请直接在最终的JSON中**省略**掉那个键。
- 你输出的json对象必须是标准的json格式，不要用斜杠\这种转义字符。

# 分析与输出
请根据以上所有信息，生成你的JSON输出:
"""
)

# --- 4. 辅助函数和更新函数 (保持不变) ---
def format_history_for_prompt(history: List[BaseMessage]) -> str:
    if not history:
        return "（没有更早的对话历史）"
    formatted_lines = [f"{'用户' if msg.type == 'human' else 'AI'}: {msg.content}" for msg in history]
    return "\n".join(formatted_lines)

def _update_state_from_llm(state_update_dict: dict):
    if not state_update_dict or not isinstance(state_update_dict, dict):
        print("[同步后处理] 模型认为无需更新状态或返回格式无效。")
        return {"status": "no_update_needed"}
    try:
        # 我们的 update_status 函数完全可以处理这种字典结构
        update_status(**state_update_dict)
        update_str = json.dumps(state_update_dict, ensure_ascii=False, indent=2)
        print(f"[同步后处理] 状态已成功更新:\n{update_str}")
        return {"status": "updated_successfully", "updated_fields": state_update_dict}
    except Exception as e:
        print(f"[同步后处理] ❌ 状态更新失败: {e}")
        return {"status": "update_failed", "error": str(e)}

update_state_runnable = RunnableLambda(_update_state_from_llm).with_config(run_name="WriteStateToFile")

# --- 5. 组装最终的后处理链 (保持不变) ---
post_sync_chain = (
    {
        "current_state": RunnableLambda(lambda _: get_status_summary()),
        "conversation_history": lambda x: format_history_for_prompt(x["conversation_history"]),
        "user_input": lambda x: x["user_input"],
        "ai_response": lambda x: x["ai_response"],
    }
    | STATE_ANALYSIS_PROMPT
    | small_llm
    | JsonOutputParser()
    | update_state_runnable
).with_config(run_name="PostDialogueSyncChain_Structured")