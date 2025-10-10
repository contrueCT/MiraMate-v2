# src/MiraMate/core/post_async_chain.py

import os
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser

# 导入我们需要的模块
from MiraMate.modules.llms import small_llm
from MiraMate.modules.memory_system import memory_system
from MiraMate.modules.settings import get_persona

# --- 1. 定义 Prompt 模板 ---
ASYNC_ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    """
# 指令
你是一个高度智能的AI助理，任务是深入分析一段对话，并提取出所有有价值的信息用于记忆和状态更新。
你必须严格按照指定的JSON格式输出你的分析结果。

# 当前对话信息
用户: {USER_NAME}
AI: {AGENT_NAME}

# 对话上下文
## 最近的对话历史
{conversation_history}

## 最新一轮对话
{USER_NAME}: {user_input}
{AGENT_NAME}: {ai_response}

# ----------- 输出格式与任务要求 (这是最重要的部分！) -----------
你的输出必须是一个JSON对象，包含以下所有键。如果某个键没有可提取的内容，请返回一个空列表 `[]` 或对应的默认值。

{{
  "facts_to_cache": [
    {{
      "content": "从对话中提取的客观事实陈述。",
      "tags": ["相关的标签", "例如：工作", "项目"],
      "confidence": "你对这条事实准确性的置信度 (0.0 到 1.0 之间的浮点数)。",
      "source": "事实来源 (例如： '用户自述', '外部链接', '学习外部知识库')"
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
      "key": "可以直接更新用户画像的结构化键 (例如: dream, job, birthday)",
      "value": "对应的值 (例如: '成为一名宇航员', '软件工程师', '1998-10-05')",
      "source": "信息来源 (例如: '用户自述', 'AI推断')"
    }}
  ],
  "temp_focus_events_to_add": [
    {{
      "content": "对话中提到的、未来会发生的、需要关注的短期事件。",
      "event_time_iso": "事件发生的ISO 8601格式日期 (格式YYYY-MM-DD，例如: '2024-08-15')",
      "expire_time_iso": "此关注事件的过期时间的ISO 8601格式日期，格式YYYY-MM-DD (通常是事件发生后一两天)",
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

关于这些键的说明：
- `facts_to_cache`: 包含对话中用户提到的客观事实，只记录从用户说的话中能获得的信息，适用于长期记忆，不是指所有客观事实，只需要与用户相关的、重要的、在长期都有意义的事实，比如说用户告诉ai他家是别墅，这是一个长期存在的事实，不需要记录的例子：用户今天吃了什么，但是可以记录用户吃过什么，因为是否吃过什么是一个永久存在的事实。
- `preferences_to_cache`: 包含对话中提到的用户偏好、喜好或厌恶，适用于长期记忆。
- `profile_updates_to_cache`: 包含对话中提到的用户画像更新信息，适用于长期记忆。
- `temp_focus_events_to_add`: 包含对话中提到的最近需要关注的事件（预计时间在两周以内的），比如某些活动、节日、工作，注意和偏好、画像、普通记忆这些内容作出区分，这种记录必须包含event_time_iso和expire_time_iso的ISO 8601格式日期，即YYYY-MM-DD。

类似、相关的内容最好整合记录到同一条记录中，除非他们在语义上有明显的区分。
你只需要提取最新一轮对话中的信息，前面的对话历史仅供参考，帮助你分析。
对于提取的信息，你需要考虑的是其重要性和长期价值，不要记录一些临时性的比如“今天的天气”、“正在做某事”这样的内容。
如果你要记录有时间信息的内容，请在content部分用自然语言写清楚具体时间，而不是使用模糊的时间描述。

# 重点注意事项！！
- 你记录的记忆内容必须是以AI的第一人称视角来描述事件和事实。例如“梦醒和我分享了他开发jingzhuan的进展，我给予了建议。”，在这个例子中梦醒是用户，“我”指的是AI自己。
- 如果对话中提到“之前提到”、“曾经说过”这样的话，你应该判断这是否包含这一次提出的新内容，如果只是单纯的重复之前的内容，则不需要记录。
- 你应该尽量避免记录模糊或不确定的信息，除非它们对理解用户的需求或情感状态至关重要。
- 对于ai回答中的内容要谨慎对待，除非用户明确表示认同，否则不要将其作为事实或偏好记录。
- 记忆的内容应当简洁明了，避免冗长的描述，确保未来检索时的效率和准确性。

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
    recent_history = history.copy()
    
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
        # a. 缓存事实记忆
        facts_to_cache = analysis_result.get("facts_to_cache", [])
        if facts_to_cache:
            for fact in facts_to_cache:
                memory_system.cache_fact_memory(
                    content=fact.get("content", ""), 
                    tags=fact.get("tags", []),
                    confidence=float(fact.get("confidence", 1.0)) 
                )
            processed_summary["facts_cached"] = len(facts_to_cache)

        # b. 缓存用户偏好
        preferences_to_cache = analysis_result.get("preferences_to_cache", [])
        if preferences_to_cache:
            for pref in preferences_to_cache:
                memory_system.cache_user_preference(
                    content=pref.get("content", ""), 
                    preference_type=pref.get("type", "未知类型"), 
                    tags=pref.get("tags", []),
                    confidence=float(pref.get("confidence", 1.0)) 
                )
            processed_summary["preferences_cached"] = len(preferences_to_cache)

        # c. 缓存用户画像更新
        profile_updates_to_cache = analysis_result.get("profile_updates_to_cache", [])
        if profile_updates_to_cache:
            # 步骤1: 将LLM返回的列表聚合成一个单一的更新字典
            profile_dict = {
                item['key']: item['value'] 
                for item in profile_updates_to_cache 
                if 'key' in item and 'value' in item
            }
            
            # 步骤2: 收集所有信息来源，并去重
            # 使用 set 来自动去重，然后用 ', '.join 连接成一个字符串
            sources = list(set(
                item.get('source', 'ai_inference') for item in profile_updates_to_cache
            ))
            source_str = ", ".join(sources)

            # 步骤3: 如果聚合后的字典不为空，则调用缓存函数
            if profile_dict:
                # 现在传递的参数类型完全正确：一个字典 和 一个字符串
                memory_system.cache_profile_update(profile_dict, source=source_str)
                processed_summary["profile_updates_cached"] = len(profile_dict)

        # d. 添加临时关注事件
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

# 这个链条将接收用户输入、AI响应和对话历史，并进行分析和处理。
analysis_parser_chain = ASYNC_ANALYSIS_PROMPT | small_llm | JsonOutputParser()

# 这个链接收 user_input, ai_response, conversation_history
post_async_chain = (
    # 一个包含所有初始信息的字典，包括 _original_input
    {
        "conversation_history": lambda x: format_history_for_prompt(x["conversation_history"]),
        "user_input": lambda x: x["user_input"],
        "ai_response": lambda x: x["ai_response"],
        "current_date": lambda x: datetime.now().strftime("%Y-%m-%d"),
        "_original_input": lambda x: x,
        "USER_NAME": lambda x: get_persona().get("USER_NAME", "小伙伴"),
        "AGENT_NAME": lambda x: get_persona().get("AGENT_NAME", "小梦")
    }
    # 接收上面准备好的字典，运行 analysis_parser_chain，
    # 然后将结果以 "analysis_json" 为键，添加到原始字典中。
    | RunnablePassthrough.assign(analysis_json=analysis_parser_chain)
    # 现在的输入是一个大字典，例如：{ "user_input": ..., "_original_input": ..., "analysis_json": ... }
    | RunnableLambda(
        lambda combined_data: _process_analysis_result({
            # 提取分析出的 JSON 内容
            **combined_data["analysis_json"], 
            "_original_input": combined_data["_original_input"] 
        })
    )
).with_config(run_name="PostDialogueAsyncChain")