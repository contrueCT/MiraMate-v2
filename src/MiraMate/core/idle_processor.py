import threading
import time
import json
from typing import Dict, List, Any

from MiraMate.modules.llms import main_llm
from MiraMate.modules.memory_system import memory_system
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# TODO：后面要把临时关注事件也加入空闲处理，解决高频发生的临时关注事件无法及时删除的问题。

# --- 1. 创建一个通用的Prompt生成函数以减少重复代码 ---
def create_consolidation_prompt(task_description: str, data_key: str, output_format: str) -> ChatPromptTemplate:
    """
    创建一个用于特定记忆类型整合的Prompt模板。
    """
    escaped_output_format = output_format.replace("{", "{{").replace("}", "}}")

    template = f"""# 指令
你是一个记忆整理大师，拥有顶级的分析和总结能力。你的当前任务是：**{task_description}**

# 任务要求
1.  **合并与提炼**: 将关于同一主题的多个记录合并成一条更全面、更精确的陈述。
2.  **去重**: 移除完全重复或语义上高度相似的记录。
3.  **解决矛盾**: 如果存在矛盾信息，优先采纳时间戳（timestamp）最新的记录。
4.  **严格格式化**: 你的输出必须是一个只包含一个键 `"{data_key}"` 的JSON对象。如果分析后认为没有内容需要存入长期记忆，请返回一个空列表 `[]` 作为值。

# 注意事项
1. 如果某条原始记录中存在时间信息，最终要保留的格式是自然语言描述下的准确、具体的时间，比如 "2023年10月1日中午"，如果原始记录中包含更精确的时间也要保留。
2. 不要编造任何信息或细节，所有内容必须基于原始记录。
3. 只保留那些具有长期价值的信息，避免记录临时性或无关紧要的内容，例如“今天的天气”、“正在做某事”这样的内容。
4. 对于所有内容性的描述，必须使用AI的第一人称视角来表达。例如，“我和（对话的人名）约定了每天晚上一起聊天”，“我注意到（对话的人名）喜欢技术讨论”。
5. 要注意区分对话中提及的事件是当前发生的还是用户和ai对过去已经发生过的事情的回顾，如果是回顾，则不能记录。

# 待处理的缓存数据
{{raw_cache_data}}

# ----------- 输出格式 (必须严格遵守！) -----------
{{{{
    "{data_key}": [
        {escaped_output_format}
    ]
}}}}

# 开始整理并生成JSON输出:
"""
    return ChatPromptTemplate.from_template(template)


# --- 2. 为每种记忆类型定义专门的Chain ---

# a. 事实整合链
fact_prompt = create_consolidation_prompt(
    task_description="分析并整合关于用户的客观事实。",
    data_key="consolidated_facts",
    output_format='{"content": "整合后的事实陈述。", "tags": ["相关标签"], "confidence": 1.0, "source": "事实来源"}'
)
fact_consolidation_chain = fact_prompt | main_llm | JsonOutputParser()

# b. 偏好整合链
preference_prompt = create_consolidation_prompt(
    task_description="分析并整合用户的个人偏好、喜好或厌恶。",
    data_key="consolidated_preferences",
    output_format='{"content": "整合后的用户偏好。", "type": "偏好类型", "tags": ["相关标签"], "confidence": 0.9, "source": "偏好来源"}'
)
preference_consolidation_chain = preference_prompt | main_llm | JsonOutputParser()

# c. 画像更新整合链 (这个稍微特殊，输出不是列表)
# TODO；优化用户画像的处理，减少冗余
profile_prompt = ChatPromptTemplate.from_template("""# 指令
你是一个用户画像分析师。你的任务是分析下列用户画像的更新请求，并整合成一个最终的、无冲突的更新字典。

# 任务要求
- 对于类似的用户画像记录，进行合并和去重。
- 如果存在矛盾信息，优先采纳时间戳最新的记录。
- 排除不属于用户画像的记录（例如：仅是对话内容或事实信息）。
- 要注意区分对话中提及的事件是当前发生的还是用户和ai对过去已经发生过的事情的回顾，如果是回顾，则不能记录。

# 待处理的画像更新缓存
{raw_cache_data}

# 当前用户画像（供参考）
{user_profile}

# ----------- 输出格式 (必须严格遵守！) -----------
请输出一个JSON对象，包含实际的用户画像字段更新(只要输出需要更新的字段即可):
示例:
{{
  "age": 25,
  "gender": "male",
  "occupation": "软件工程师"
}}

# 你需要在每次分析用户画像后修改其中的always_remember字段，这个字段在每次ai回答用户时都需要考虑的信息，你根据整体考虑要把哪些信息加入或移除这个字段来确保ai对用户的了解，同时避免重复和冗余,只挑选出所有用户画像中的关键信息，采用键值对的形式，整理成一个字典。
示例：
{{
  "always_remember": {{
    "name": "小明",
    "age": 25,
    "gender": "male",
    "occupation": "软件工程师",
    "hobbies": ["编程", "阅读", "旅行"],
    "dream": "成为一名优秀的全栈开发者",
  }}
}}
# 开始分析并生成最终的画像更新JSON:
""")
profile_consolidation_chain = profile_prompt | main_llm | JsonOutputParser()


# --- d. 重要事件识别链 ---
IMPORTANT_EVENT_PROMPT = ChatPromptTemplate.from_template(
"""# 指令
你是一位经验丰富的人生分析师，擅长从零散的对话和笔记中识别出具有长期影响的关键性“重大事件”。

# 核心任务
分析下面提供的“最近的对话记录”和“临时的关注事件”，判断其中是否包含了可以被定义为**重大事件**的信息。

# “重大事件”的定义 (这是最重要的部分！)
重大事件是**稀少**且**关键**的，它们往往对一个人的生活轨迹、核心目标或情感状态产生深远影响。
- **是重大事件**:
    - 人生决策：决定考研、换城市、开始创业。
    - 重要关系变化：开始一段重要的恋情、与挚友和解。
    - 长期目标：立志要写一本书、为期一年的海外学习计划。
    - 深刻的个人成长或转变。
- **不是重大事件**:
    - 日常安排：下周要考试、明天有约会、计划周末去购物。 (这些是“临时关注事件”)
    - 短期情绪：今天很开心、昨天有点难过。
    - 普通的兴趣爱好。

**规则：如果分析后没有发现任何符合上述严格定义的重大事件，你必须返回一个空列表 `[]`！不要为了填充而降低标准。**

**注意事项**
如果涉及AI的描述，必须使用第一人称视角。例如，“我和（对话的人名）约定了要一起奋斗，实现梦想”。如果是单纯的事件描述，则使用第三人称。
要注意区分对话中提及的事件是当前发生的还是用户和ai对过去已经发生过的事情的回顾，如果是回顾，则不能记录。

# 待分析的原始信息
## 最近的对话记录
{recent_dialogues}

## 临时的关注事件
{temp_focus_events}

# ----------- 输出格式 (必须严格遵守！) -----------
你的输出必须是一个只包含一个键 `"identified_important_events"` 的JSON对象。
{{
    "identified_important_events": [
        {{
            "content": "对重大事件的详细、客观描述。",
            "event_type": "事件的分类 (例如：学业目标, 职业规划, 人际关系, 个人成长)",
            "summary": "一句话概括这个事件的核心。",
            "tags": ["相关的关键词标签"]
        }}
    ]
}}

# 开始分析并生成JSON输出:
"""
)
important_event_identification_chain = IMPORTANT_EVENT_PROMPT | main_llm | JsonOutputParser()

class IdleProcessor:
    def __init__(self, idle_threshold_seconds: int = 1200):
        self.idle_threshold = idle_threshold_seconds
        self.last_interaction_time = time.time()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)

    def update_last_interaction_time(self):
        self.last_interaction_time = time.time()
        print("[IdleProcessor] 互动计时器已重置。")

    def _process_caches(self):
        """核心处理逻辑：分步调用LLM整合各类缓存并写入数据库。"""
        print("\n[IdleProcessor] 开始分步处理缓存记忆...")
        
        # --- a. 处理事实缓存 ---
        facts_cache = memory_system.load_fact_cache()
        if facts_cache:
            try:
                print("[IdleProcessor] 正在整合事实记忆...")
                result = fact_consolidation_chain.invoke({"raw_cache_data": json.dumps(facts_cache, ensure_ascii=False)})
                for fact in result.get("consolidated_facts", []):
                    memory_system.save_fact_memory(content=fact['content'], tags=fact['tags'], confidence=fact.get('confidence', 1.0), source=fact.get('source', '未知来源'))
                memory_system.clear_fact_cache()
                print("[IdleProcessor] ✅ 事实记忆处理完成，缓存已清除。")
            except Exception as e:
                print(f"[IdleProcessor] ❌ 事实记忆处理失败: {e}")

        # --- b. 处理偏好缓存 ---
        preferences_cache = memory_system.load_preference_cache()
        if preferences_cache:
            try:
                print("[IdleProcessor] 正在整合偏好记忆...")
                result = preference_consolidation_chain.invoke({"raw_cache_data": json.dumps(preferences_cache, ensure_ascii=False)})
                for pref in result.get("consolidated_preferences", []):
                     memory_system.save_user_preference(content=pref['content'], preference_type=pref['type'], tags=pref['tags'])
                memory_system.clear_preference_cache()
                print("[IdleProcessor] ✅ 偏好记忆处理完成，缓存已清除。")
            except Exception as e:
                print(f"[IdleProcessor] ❌ 偏好记忆处理失败: {e}")
        
        # --- c. 处理画像更新缓存 ---
        profile_updates_cache = memory_system.load_profile_cache()
        if profile_updates_cache:
            try:
                print("[IdleProcessor] 正在整合画像更新...")
                current_profile = memory_system.load_user_profile() or {}
                final_updates = profile_consolidation_chain.invoke({
                    "raw_cache_data": json.dumps(profile_updates_cache, ensure_ascii=False),
                    "user_profile": json.dumps(current_profile, ensure_ascii=False)
                })
                if final_updates:
                    memory_system.update_user_profile(**final_updates)
                memory_system.clear_profile_cache()
                print("[IdleProcessor] ✅ 用户画像处理完成，缓存已清除。")
            except Exception as e:
                print(f"[IdlePocessor] ❌ 用户画像处理失败: {e}")


        # --- d. 处理重要事件识别 ---
        try:
            print("[IdleProcessor] 正在分析是否存在新的重要事件...")
            # 从数据库和文件中获取所需信息
            recent_dialogues = memory_system.get_recent_dialogs(limit=5) # 获取最近10条对话作为上下文
            temp_focus_events = memory_system.get_active_focus_events()

            # 仅在有内容可分析时才调用LLM
            if recent_dialogues or temp_focus_events:
                result = important_event_identification_chain.invoke({
                    "recent_dialogues": json.dumps(recent_dialogues, ensure_ascii=False, indent=2),
                    "temp_focus_events": json.dumps(temp_focus_events, ensure_ascii=False, indent=2)
                })
                
                identified_events = result.get("identified_important_events", [])
                if identified_events:
                    print(f"[IdleProcessor] ✅ 识别到 {len(identified_events)} 个新的重要事件，正在存入记忆库...")
                    for event in identified_events:
                        memory_system.save_important_event(**event)
                else:
                    print("[IdleProcessor] 未发现新的重要事件。")
            else:
                print("[IdleProcessor] 无可供分析的对话或关注事件，跳过重要事件识别。")
        except Exception as e:
            print(f"[IdleProcessor] ❌ 重要事件识别失败: {e}")

        print("[IdleProcessor] 所有缓存处理流程结束。")


    def _worker_loop(self):
        """后台线程的工作循环。"""
        print("[IdleProcessor] 后台工作线程已启动。")
        while not self._stop_event.is_set():
            time_since_last_interaction = time.time() - self.last_interaction_time
            
            if time_since_last_interaction > self.idle_threshold:
                # 检查是否有任何缓存需要处理
                cache_status = memory_system.get_cache_status()
                if sum(cache_status.values()) > 0:
                    print(f"\n[IdleProcessor] 检测到空闲且有缓存数据，触发记忆处理...")
                    self._process_caches()
                
                # 无论是否处理，都重置计时器，避免CPU空转检查
                self.update_last_interaction_time()
            
            self._stop_event.wait(60)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        print("[IdleProcessor] 后台工作线程已停止。")