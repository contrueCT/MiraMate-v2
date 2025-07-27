import os  
from jinja2 import Environment, FileSystemLoader
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory


from MiraMate.modules.llms import main_llm, small_llm  
from MiraMate.modules.memory_system import memory_system
from MiraMate.modules.status_system import get_status_summary 
from MiraMate.modules.TimeTokenMemory import CustomTokenMemory



# --- 1. System Prompt 构建函数 ---
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
MIRA_MATE_ROOT = os.path.abspath(os.path.join(CORE_DIR, '..'))
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
understanding_prompt = ChatPromptTemplate.from_template(
    """请分析以下用户输入。
你的任务是提取用户的核心意图、情感，并生成一个最适合用于向量数据库搜索的查询关键词。
以严格的JSON格式返回，包含'intent', 'emotion', 'memory_query'三个键。
用户最新输入: {user_input}
JSON输出:
/no_think"""

)
understanding_chain = (understanding_prompt | small_llm | JsonOutputParser()).with_config(run_name="UnderstandingChain")

# --- 3. 最终链条的构建 ---
# a. 并行获取上下文的组件
context_fetcher = RunnableParallel(
    understanding={"user_input": lambda x: x["user_input"]} | understanding_chain,
    agent_state=lambda _: get_status_summary(),
    user_profile=lambda _: memory_system.load_user_profile(),
    focus_events=lambda _: memory_system.get_active_focus_events(),
    user_input=lambda x: x["user_input"],
    history=lambda x: x["history"]
).with_config(run_name="ParallelContextFetching")

# b. 检索链
retrieval_chain = RunnableLambda(
    lambda x: {
        "retrieved_memory": memory_system.comprehensive_search(x["understanding"]["memory_query"]),
        **x
    }
).with_config(run_name="RetrievalChain")

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