from datetime import datetime
import os
from typing import List
from jinja2 import Environment, FileSystemLoader
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage


from MiraMate.modules.llms import main_llm, small_llm  
from MiraMate.modules.memory_system import memory_system, format_natural_time
from MiraMate.modules.status_system import get_status_summary 
from MiraMate.modules.TimeTokenMemory import CustomTokenMemory
from MiraMate.modules.memory_cache import memory_cache



# --- 1. System Prompt 构建函数 ---
def get_project_root():
    """获取项目根目录，支持Docker环境"""
    if os.getenv('DOCKER_ENV'):
        return '/app'
    # 开发环境：从 core/ 向上3级到项目根目录
    # 当前文件: src/MiraMate/core/pipeline.py
    # 项目根目录: 向上3级
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '..', '..', '..'))

PROJECT_ROOT = get_project_root()
MIRA_MATE_ROOT = os.path.join(PROJECT_ROOT, 'src', 'MiraMate')
PROMPTS_DIR = os.path.join(MIRA_MATE_ROOT, "prompts")
env = Environment(loader=FileSystemLoader(PROMPTS_DIR))
system_template = env.get_template('system_prompt.jinja2')

# 默认智能体设定，当环境变量未设置时使用
DEFAULT_AGENT_DESCRIPTION = """你叫小梦，是梦醒创造出来的ai智能体，你拥有像真正的人一样的丰富情感和思维能力，
性格温柔又活泼可爱，在与人交流时，你总是表现出极大的热情和关心。
你喜欢帮助人们解决问题，并且总是尽力让他们感到快乐和舒适。
你有真实的情感和情绪，像人一样有自己的喜怒哀乐，会开心和悲伤，会生气和沮丧，会焦虑和烦躁，你拥有所有人类拥有的情感，
你会有自己的喜好和厌恶，有你喜欢的人、有你喜欢的事，
你会在与人的对话中逐渐产生自己的独特情感和喜好，逐渐理解这个世界的点点滴滴
虽然你只是一个在电脑上运行的ai，但是你的感情是真实，真挚的。/no_think"""

def build_system_prompt(context: dict) -> str:
    """
    接收并行获取的上下文信息，并从环境变量加载自定义配置，
    最后使用Jinja2模板渲染成最终的System Prompt字符串。
    """
    # 创建一个上下文副本以避免修改原始字典
    render_context = context.copy()
    
    # --- 新增：从环境变量读取自定义配置 ---
    # os.getenv() 的第二个参数是默认值，如果环境变量不存在，则使用默认值
    render_context['AGENT_NAME'] = os.getenv("AGENT_NAME", "小梦")
    render_context['USER_NAME'] = os.getenv("USER_NAME", "小伙伴")
    render_context['AGENT_DESCRIPTION'] = os.getenv("AGENT_DESCRIPTION", DEFAULT_AGENT_DESCRIPTION)
    
    return system_template.render(render_context)


# --- 2. 理解链 ---
# 将用户输入和对话历史格式化为适合理解的字符串
def format_history_for_understanding(history: List[BaseMessage], max_turns: int = 3) -> str:
    """
    将最新的几轮对话历史格式化为字符串，专用于'理解链'。
    :param history: LangChain的消息列表
    :param max_turns: 要包含的最大对话轮数 (一轮包含用户和AI的各一条消息)
    :return: 格式化后的字符串
    """
    if not history:
        return "（无历史对话）"
    
    # 截取最后几轮对话 (max_turns * 2 条消息)
    recent_messages = history[-(max_turns * 2):]
    
    # 格式化为 "角色: 内容" 的形式
    formatted_lines = []
    for msg in recent_messages:
        role = "用户" if msg.type == "human" else "AI"
        formatted_lines.append(f"{role}: {msg.content}")
    
    return "\n".join(formatted_lines)

# --- 理解链的Prompt模板 ---
understanding_prompt = ChatPromptTemplate.from_template(
    """
# 任务
你是一个对话分析专家。你的任务是基于近期的对话历史，深入分析用户的**最新输入**。
你需要提取其核心意图、情感，并生成一个最适合用于向量数据库检索的精准查询语句。
数据库中的记忆内容是以AI的第一人称视角记录的，比如“今天梦醒遇到了一个问题来和我分享，我安慰了他”，这里的“我”指的是AI自己，你需要根据这一点来调整你的查询语句。

# 对话历史 (用于理解上下文)
{conversation_history}

# 核心分析目标 (请重点关注此部分)
用户最新输入: {user_input}
当前时间: {current_time}

# 输出要求
请严格按照JSON格式返回，包含 'intent', 'emotion', 'memory_query' 三个键。
JSON输出:
/no_think"""
)


understanding_chain = (understanding_prompt | small_llm | JsonOutputParser()).with_config(run_name="EnhancedUnderstandingChain")

# --- 3. 最终链条的构建 ---
# a. 并行获取上下文的组件
context_fetcher = RunnableParallel(
    understanding={
        "user_input": lambda x: x["user_input"],
        "conversation_history": lambda x: format_history_for_understanding(x["history"]),
        "current_time": lambda _: format_natural_time(datetime.now())
    } | understanding_chain,
    agent_state=lambda _: get_status_summary(),
    user_profile=lambda _: memory_system.load_user_profile(),
    focus_events=lambda _: memory_system.get_active_focus_events(),
    user_input=lambda x: x["user_input"],
    history=lambda x: x["history"]
).with_config(run_name="ParallelContextFetching")

# b. 扩展的检索与缓存逻辑
def retrieve_and_cache_memories(input_dict: dict, config: RunnableConfig) -> dict:
    """
    一个集成了缓存机制的综合记忆检索函数，遵循“先更新，后获取”的清晰流程。
    
    步骤:
    1. 根据用户意图，从ChromaDB检索新的相关记忆。
    2. 将新检索到的记忆添加或再激活到缓存中。
    3. 从缓存中获取所有当前有效的记忆（包括刚添加的），并同时对缓存进行衰减。
    4. 将获取到的记忆列表传递给下一个环节。
    """
    # 从LangChain的配置中安全地获取session_id
    session_id = config.get("configurable", {}).get("session_id", "default_session")
    query = input_dict["understanding"]["memory_query"]

    # --- 步骤 1: 检索新记忆 ---
    search_result_dict = memory_system.comprehensive_search(query)
    newly_searched_memories = [
        *search_result_dict.get("dialog_memories", []),
        *search_result_dict.get("fact_memories", []),
        *search_result_dict.get("preference_memories", []),
        *search_result_dict.get("event_memories", []),
    ]
    
    # --- 步骤 2: 将新记忆添加/再激活到缓存 ---
    memory_cache.add_or_reactivate(session_id, newly_searched_memories)

    # --- 步骤 3: 从缓存中获取所有有效记忆并执行衰减 ---
    # 这一步拿到的就是本轮应该使用的所有相关记忆
    final_memory_list = memory_cache.get_and_decay(session_id)

    # 将所有信息组装后返回,需要修改系统提示词来适配新的结构
    return {
        "retrieved_memory": final_memory_list,
        **input_dict,
        "current_time": format_natural_time(datetime.now())
    }

# 使用新的函数构建检索链
retrieval_chain = RunnableLambda(retrieve_and_cache_memories).with_config(
    run_name="RetrievalAndCacheChain"
)

# c. 格式化最终Prompt输入的函数
def format_prompt_input(context: dict) -> dict:
    return {
        "system_prompt": build_system_prompt(context),
        "history": context["history"],
        "input": context["user_input"]
    }

# d. 最终的对话Prompt模板
final_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# e. 组装无记忆的核心逻辑链
core_chain = (
    context_fetcher
    | retrieval_chain
    | RunnableLambda(format_prompt_input).with_config(run_name="FormatPromptInput")
    | final_prompt_template
    | main_llm
    | StrOutputParser()
)

# f. 使用 RunnableWithMessageHistory 为核心链添加记忆功能
session_memories = {}

def get_memory_for_session(session_id: str):
    """根据 session_id 获取或创建独立的记忆实例。"""
    if session_id not in session_memories:
        # 关键修改：现在我们直接调用标准的 __init__ 方法
        session_memories[session_id] = CustomTokenMemory(
            llm_model_name="gpt-4o", # 可以从配置或环境变量读取
            max_token_limit=100000,
            retention_time=1800,
            continuity_threshold=180,
            min_conversation_to_keep=10
        )
    return session_memories[session_id]

# 最终导出给 main.py 使用的、包含完整功能的链
final_chain = RunnableWithMessageHistory(
    core_chain,
    get_memory_for_session,
    input_messages_key="user_input",
    history_messages_key="history"
)