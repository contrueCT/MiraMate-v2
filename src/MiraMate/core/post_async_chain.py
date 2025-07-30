# src/MiraMate/core/post_async_chain.py

import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser

# 导入我们需要的模块
from MiraMate.modules.llms import small_model
from MiraMate.modules.memory_system import memory_system

# --- 1. 定义 Prompt 模板 ---
ASYNC_ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    """
# 指令
你是一个高度智能的AI助理，任务是深入分析一段对话，并提取出所有有价值的信息用于记忆和状态更新。
你必须严格按照指定的JSON格式输出你的分析结果。

# 对话上下文
## 最近的对话历史
{conversation_history}

## 最新一轮对话
用户: {user_input}
AI: {ai_response}

# ----------- 输出格式与任务要求 (这是最重要的部分！) -----------
你的输出必须是一个JSON对象，包含以下所有键。如果某个键没有可提取的内容，请返回一个空列表 `[]` 或对应的默认值。

{{
  "facts_to_cache": [
    {{
      "content": "从对话中提取的客观事实陈述。",
      "tags": ["相关的标签", "例如：工作", "项目"],
      "confidence": "你对这条事实准确性的置信度 (0.0 到 1.0 之间的浮点数)。"
    }}
  ],
  "preferences_to_cache": [
    {{
      "content": "从对话中提取的明确的用户偏好、喜好或厌恶。",
      "type": "偏好类型 (例如：音乐偏好, 食物偏好, 观点)",
      "tags": ["相关的标签"],
      "confidence": "你对这条偏好总结准确性的置信度 (0.0 到 1.0 之间的浮点数)。"
    }}
  ],
  "profile_updates_to_cache": [
    {{
      "key": "可以直接更新用户画像的键 (例如: dream, job, birthday)",
      "value": "对应的值"
    }}
  ],
  "temp_focus_events_to_add": [
    {{
      "content": "对话中提到的、未来会发生的、需要关注的短期事件。",
      "event_time_iso": "事件发生的ISO 8601格式日期 (例如: '2024-08-15')",
      "expire_time_iso": "此关注事件的过期时间的ISO 8601格式 (通常是事件发生后一两天)",
      "tags": ["相关的标签"]
    }}
  ],
  "dialog_log_metadata": {{
    "topic": "对本次对话核心主题的简短概括。",
    "sentiment": "本次对话的整体情感基调 (例如: 开心, 伤感, 激动, 平淡)。",
    "importance": "本次对话的重要性评分 (0.0 到 1.0，浮点数)。",
    "tags": ["描述本次对话内容的关键词标签"],
    "is_potential_major_event": "本次对话是否揭示了一个潜在的长期重大事件 (true/false)。"
  }}
}}

当前日期是: {current_date}

请开始你的分析，并生成JSON输出:
/no_think
"""
)

# --- 2. 辅助函数 ---
def format_history_for_prompt(history: list[BaseMessage]) -> str:
    """将 LangChain 的消息对象列表格式化为对 LLM 更友好的字符串。"""
    if not history:
        return "（没有更早的对话历史）"
    
    # 只取最近的几轮对话，避免上下文过长
    recent_history = history[-10:] # 最多取最近5轮对话（10条消息）
    
    formatted_lines = [f"{'用户' if msg.type == 'human' else 'AI'}: {msg.content}" for msg in recent_history]
    return "\n".join(formatted_lines)

# --- 3. 核心处理逻辑函数 ---
# 这个函数将作为链的最后一步，接收LLM的输出并执行所有数据库和文件写入操作。
def _process_analysis_result(analysis_result: dict) -> dict:
    """
    接收小模型分析出的JSON对象，并严格按照memory_system的接口进行处理。
    """
    print("\n[异步后处理] 开始处理分析结果...")
    processed_summary = {}

    try:
        # a. 缓存事实记忆 (修正：传入 confidence)
        facts_to_cache = analysis_result.get("facts_to_cache", [])
        if facts_to_cache:
            for fact in facts_to_cache:
                memory_system.cache_fact_memory(
                    content=fact.get("content", ""), 
                    tags=fact.get("tags", []),
                    confidence=float(fact.get("confidence", 1.0)) # 确保是浮点数
                )
            processed_summary["facts_cached"] = len(facts_to_cache)

        # b. 缓存用户偏好 (修正：传入 confidence)
        preferences_to_cache = analysis_result.get("preferences_to_cache", [])
        if preferences_to_cache:
            for pref in preferences_to_cache:
                memory_system.cache_user_preference(
                    content=pref.get("content", ""), 
                    preference_type=pref.get("type", "未知类型"), 
                    tags=pref.get("tags", []),
                    confidence=float(pref.get("confidence", 1.0)) # 确保是浮点数
                )
            processed_summary["preferences_cached"] = len(preferences_to_cache)

        # c. 缓存用户画像更新 (保持不变，接口原本就匹配)
        profile_updates_to_cache = analysis_result.get("profile_updates_to_cache", [])
        if profile_updates_to_cache:
            profile_dict = {item['key']: item['value'] for item in profile_updates_to_cache if 'key' in item and 'value' in item}
            if profile_dict:
                memory_system.cache_profile_update(profile_dict)
                processed_summary["profile_updates_cached"] = len(profile_dict)

        # d. 添加临时关注事件 (保持不变，接口原本就匹配)
        temp_events_to_add = analysis_result.get("temp_focus_events_to_add", [])
        if temp_events_to_add:
            for event in temp_events_to_add:
                memory_system.save_temp_focus_event(
                    content=event.get("content", ""),
                    event_time=event.get("event_time_iso", ""),
                    expire_time=event.get("expire_time_iso", ""),
                    tags=event.get("tags", [])
                )
            processed_summary["temp_events_added"] = len(temp_events_to_add)

        # e. 保存对话记录 (保持不变，接口原本就匹配)
        user_input = analysis_result.get("_original_input", {}).get("user_input", "")
        ai_response = analysis_result.get("_original_input", {}).get("ai_response", "")
        dialog_meta = analysis_result.get("dialog_log_metadata", {})
        
        if user_input and ai_response and dialog_meta:
            memory_system.save_dialog_log(
                user_input=user_input,
                ai_response=ai_response,
                topic=dialog_meta.get("topic", "未知主题"),
                sentiment=dialog_meta.get("sentiment", "未知"),
                importance=float(dialog_meta.get("importance", 0.5)),
                tags=dialog_meta.get("tags", []),
                additional_metadata={"is_potential_major_event": dialog_meta.get("is_potential_major_event", False)}
            )
            processed_summary["dialog_log_saved"] = True
        
        print(f"[异步后处理] 处理完成: {processed_summary}")
        return {"status": "async_processing_complete", "summary": processed_summary}

    except Exception as e:
        print(f"[异步后处理] ❌ 处理分析结果时出错: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "async_processing_failed", "error": str(e)}

# --- 4. 组装完整的异步后处理链 ---

# 这个链接收 user_input, ai_response, conversation_history
post_async_chain = (
    # 步骤 1: 准备Prompt的输入字典
    {
        "conversation_history": lambda x: format_history_for_prompt(x["conversation_history"]),
        "user_input": lambda x: x["user_input"],
        "ai_response": lambda x: x["ai_response"],
        "current_date": lambda x: datetime.now().strftime("%Y-%m-%d"),
        # 使用一个技巧，将原始输入也传递下去，方便最后一步使用
        "_original_input": lambda x: x
    }
    # 步骤 2: 将字典送入模板
    | ASYNC_ANALYSIS_PROMPT
    # 步骤 3: 调用小模型生成JSON字符串
    | small_model
    # 步骤 4: 解析JSON字符串为Python字典
    | JsonOutputParser()
    # 步骤 5: 将原始输入合并到解析结果中，以便后续处理
    | RunnableLambda(lambda parsed_json, config: {**parsed_json, "_original_input": config["run_manager"].get_context()["input"].get("_original_input")})
    # 步骤 6: 调用核心处理函数，执行所有副作用
    | RunnableLambda(_process_analysis_result)
).with_config(run_name="PostDialogueAsyncChain")