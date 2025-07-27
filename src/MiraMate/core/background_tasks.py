import asyncio
from MiraMate.modules.memory_system import memory_system, cache_fact_memory, cache_user_preference
from MiraMate.modules.llms import small_llm

async def process_conversation_asynchronously(user_input, ai_response, session_id):
    """
    模拟对话后的异步处理任务。
    在真实应用中，这会是一个由 Celery, RQ, or Dramatiq 调度的后台任务。
    """
    print(f"\n[后台任务 {session_id}] 开始处理对话 '{user_input[:20]}...'")
    # 模拟网络I/O或LLM分析的耗时
    await asyncio.sleep(5) 

    # 1. 调用小模型提取事实、偏好等信息
    analysis_prompt = f"""从以下对话中提取事实、用户偏好和用户信息更新。
    用户: {user_input}
    AI: {ai_response}
    以JSON格式返回，包含 'facts', 'preferences', 'profile_updates' 三个键，如果没有则返回空列表或空字典。"""
    
    # 实际应用中会用真实调用，这里我们用模拟数据代替以节省token和时间
    # extracted_info = await small_model.ainvoke(analysis_prompt)
    
    # 模拟提取结果
    extracted_info = {
        "facts": [{"content": f"用户对'{user_input[:10]}'的看法是'{ai_response[:10]}'", "tags": ["对话摘要"]}],
        "preferences": [{"content": f"用户似乎对'{user_input[:10]}'话题感兴趣", "type": "兴趣偏好", "tags": ["新话题"]}]
    }
    
    # 2. 将提取的信息存入缓存文件
    if extracted_info.get("facts"):
        for fact in extracted_info["facts"]:
            cache_fact_memory(fact["content"], fact["tags"])
            
    if extracted_info.get("preferences"):
        for pref in extracted_info["preferences"]:
            cache_user_preference(pref["content"], pref["type"], pref["tags"])
            
    # 3. 将原始对话存入数据库
    # memory_system.save_dialog_log(...)

    print(f"[后台任务 {session_id}] 处理完成，新的事实和偏好已缓存。")


def run_idle_maintenance():
    """
    模拟智能体空闲时的维护任务。
    在真实应用中，这会是一个由 Cron job 或 APScheduler 调度的定时任务。
    """
    print("\n[空闲时维护] 检查并开始整理缓存文件...")
    
    facts_cache = memory_system.load_fact_cache()
    if not facts_cache:
        print("[空闲时维护] 没有需要处理的缓存。")
        return
        
    print(f"[空闲时维护] 发现 {len(facts_cache)} 条事实缓存，正在处理...")
    # 2. 调用模型对缓存信息进行整理、去重、合并
    # summarized_facts = main_llm.invoke(f"请整理这些事实: {facts_cache}")
    
    # 3. 将整理后的信息存入 ChromaDB
    # for fact in summarized_facts:
    #     memory_system.save_fact_memory(...)
        
    # 4. 清空缓存文件
    memory_system.clear_fact_cache()
    
    print("[空闲时维护] 缓存已处理并存入长期记忆。")