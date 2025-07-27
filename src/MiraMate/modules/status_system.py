import json
import os
from datetime import datetime
from typing import Dict, Any, List

# 加载状态信息已接入对话链，后续需要在对话完后的步骤中更新各种状态，可以将会话理解模块生成的信息作为更新内容

# 路径配置
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, '..', '..', '..'))
STATUS_DIR = os.path.join(PROJECT_ROOT, "memory", "status_storage")
STATUS_FILE = os.path.join(STATUS_DIR, "status.json")
RELATIONSHIP_HISTORY_FILE = os.path.join(STATUS_DIR, "relationship_history.json")

# 创建必要的目录
os.makedirs(STATUS_DIR, exist_ok=True)

# 获取当前时间戳
def get_timestamp() -> str:
    return datetime.now().isoformat()

# 获取可读时间戳
def get_readable_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 加载状态
def load_status() -> Dict[str, Any]:
    if not os.path.exists(STATUS_FILE):
        return {
            "timestamp": get_timestamp(),
            "ai_status": {
                "emotion": {"mood": "平静", "strength": 0.5},
                "user_attitude": {"emotional_feeling": "中立", "intimacy": 0.5},
                "relationship_level": 1.0,  # 新增：关系亲密度 1-10
                "recent_topic_tags": []
            },
            "user_status": {
                "last_emotion": "未知",
                "last_topic": "无",
                "current_mood": "未知",
                "energy_level": 0.5  # 新增：用户活跃度
            },
            "context_notes": {
                "thinking_focus": "无",
                "intent": "无",
                "conversation_style": "正常",  # 新增：对话风格
                "session_context": ""  # 新增：会话上下文
            },
            "session_stats": {  # 新增：会话统计
                "message_count": 0,
                "session_start": get_timestamp(),
                "last_interaction": get_timestamp()
            }
        }
    with open(STATUS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# 保存状态
def save_status(status: Dict[str, Any]):
    status["timestamp"] = get_timestamp()
    if "session_stats" in status:
        status["session_stats"]["last_interaction"] = get_timestamp()
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)

# 部分更新状态（自动合并）
def update_status(**kwargs):
    status = load_status()
    for key, value in kwargs.items():
        if isinstance(value, dict) and key in status:
            status[key].update(value)
        else:
            status[key] = value
    save_status(status)

# 添加标签（带时间戳，避免重复）
def add_tag(tag: str):
    status = load_status()
    tags = status.get("ai_status", {}).get("recent_topic_tags", [])
    if not any(t["name"] == tag for t in tags):
        tags.append({"name": tag, "timestamp": get_timestamp()})
        status["ai_status"]["recent_topic_tags"] = tags
        save_status(status)
        print(f"✅ 标签已添加: {tag}")
    else:
        print(f"⚠️ 标签已存在: {tag}")

# 删除标签
def remove_tag(tag: str):
    status = load_status()
    tags = status.get("ai_status", {}).get("recent_topic_tags", [])
    original_count = len(tags)
    tags = [t for t in tags if t["name"] != tag]
    
    if len(tags) < original_count:
        status["ai_status"]["recent_topic_tags"] = tags
        save_status(status)
        print(f"✅ 标签已删除: {tag}")
    else:
        print(f"⚠️ 标签不存在: {tag}")

# 修改标签（保持时间戳不变）
def edit_tag(old_tag: str, new_tag: str):
    status = load_status()
    tags = status.get("ai_status", {}).get("recent_topic_tags", [])
    for t in tags:
        if t["name"] == old_tag:
            t["name"] = new_tag
            save_status(status)
            print(f"✅ 标签已修改: {old_tag} → {new_tag}")
            return
    print(f"⚠️ 未找到标签: {old_tag}")

# 更新AI情绪状态
def update_ai_emotion(mood: str, strength: float):
    """更新AI的情绪状态"""
    update_status(ai_status={
        "emotion": {"mood": mood, "strength": max(0.0, min(1.0, strength))}
    })
    print(f"✅ AI情绪已更新: {mood} (强度: {strength})")

# 更新对用户的态度
def update_user_attitude(emotional_feeling: str, intimacy_change: float = 0.0):
    """更新对用户的情感态度"""
    status = load_status()
    current_intimacy = status["ai_status"]["user_attitude"]["intimacy"]
    
    # 计算新的亲密度（非线性变化）
    if intimacy_change > 0:
        # 正向变化随亲密度提高而减小
        adjusted_change = intimacy_change * (1 - current_intimacy * 0.5)
    else:
        # 负向变化影响较大
        adjusted_change = intimacy_change
    
    new_intimacy = max(0.0, min(1.0, current_intimacy + adjusted_change))
    
    update_status(ai_status={
        "user_attitude": {
            "emotional_feeling": emotional_feeling,
            "intimacy": new_intimacy
        }
    })
    
    # 记录重要的关系变化
    if abs(intimacy_change) >= 0.1:
        save_relationship_change(emotional_feeling, current_intimacy, new_intimacy, intimacy_change)
    
    print(f"✅ 用户态度已更新: {emotional_feeling} (亲密度: {new_intimacy:.2f})")

# 更新关系等级
def update_relationship_level(change: float):
    """更新关系亲密度等级 (1-10)"""
    status = load_status()
    current_level = status["ai_status"].get("relationship_level", 1.0)
    
    # 非线性变化计算
    if change > 0:
        adjusted_change = change * (1 - current_level/12)
    else:
        adjusted_change = change
    
    new_level = max(1.0, min(10.0, current_level + adjusted_change))
    
    update_status(ai_status={"relationship_level": new_level})
    
    # 记录重要关系变化
    if abs(new_level - current_level) >= 0.5:
        save_relationship_change("关系等级变化", current_level, new_level, change)
    
    print(f"✅ 关系等级已更新: {current_level:.1f} → {new_level:.1f}")
    return new_level

# 记录关系变化历史
def save_relationship_change(reason: str, old_value: float, new_value: float, change: float):
    """保存关系变化记录"""
    if not os.path.exists(RELATIONSHIP_HISTORY_FILE):
        history = []
    else:
        with open(RELATIONSHIP_HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    
    change_record = {
        "timestamp": get_timestamp(),
        "readable_time": get_readable_timestamp(),
        "reason": reason,
        "old_value": old_value,
        "new_value": new_value,
        "change": change,
        "magnitude": abs(change)
    }
    
    history.append(change_record)
    
    # 只保留最近50条记录
    if len(history) > 50:
        history = history[-50:]
    
    with open(RELATIONSHIP_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# 获取关系变化历史
def get_relationship_history(limit: int = 10) -> List[Dict]:
    """获取最近的关系变化记录"""
    if not os.path.exists(RELATIONSHIP_HISTORY_FILE):
        return []
    
    with open(RELATIONSHIP_HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    
    return history[-limit:] if history else []

# 更新用户状态
def update_user_status(last_emotion: str = None, last_topic: str = None, 
                     current_mood: str = None, energy_level: float = None):
    """更新用户状态信息"""
    updates = {}
    if last_emotion is not None:
        updates["last_emotion"] = last_emotion
    if last_topic is not None:
        updates["last_topic"] = last_topic
    if current_mood is not None:
        updates["current_mood"] = current_mood
    if energy_level is not None:
        updates["energy_level"] = max(0.0, min(1.0, energy_level))
    
    if updates:
        update_status(user_status=updates)
        print(f"✅ 用户状态已更新: {updates}")

# 更新上下文注释
def update_context_notes(thinking_focus: str = None, intent: str = None, 
                        conversation_style: str = None, session_context: str = None):
    """更新上下文注释"""
    updates = {}
    if thinking_focus is not None:
        updates["thinking_focus"] = thinking_focus
    if intent is not None:
        updates["intent"] = intent
    if conversation_style is not None:
        updates["conversation_style"] = conversation_style
    if session_context is not None:
        updates["session_context"] = session_context
    
    if updates:
        update_status(context_notes=updates)
        print(f"✅ 上下文注释已更新: {updates}")

# 增加会话计数
def increment_message_count():
    """增加消息计数"""
    status = load_status()
    current_count = status.get("session_stats", {}).get("message_count", 0)
    update_status(session_stats={"message_count": current_count + 1})

# 获取当前状态摘要
def get_status_summary() -> Dict[str, Any]:
    """获取状态系统摘要，用于AI上下文"""
    status = load_status()
    
    # 获取关系等级描述
    relationship_level = status["ai_status"].get("relationship_level", 1.0)
    relationship_descriptions = {
        (1, 2): "初期接触阶段，关系较为礼貌但疏远",
        (3, 4): "初步熟悉阶段，开始建立基本信任", 
        (5, 6): "熟悉阶段，相互了解并形成稳定互动模式",
        (7, 8): "亲密阶段，交流更加开放和情感化",
        (9, 10): "深度亲密阶段，关系非常紧密和私人化"
    }
    
    relationship_desc = ""
    for range_key, desc in relationship_descriptions.items():
        if range_key[0] <= relationship_level <= range_key[1]:
            relationship_desc = desc
            break
    
    return {
        "ai_emotion": status["ai_status"]["emotion"],
        "user_attitude": status["ai_status"]["user_attitude"], 
        "relationship_level": relationship_level,
        "relationship_description": relationship_desc,
        "user_current": status["user_status"],
        "context": status["context_notes"],
        "session_info": status["session_stats"],
        "recent_topic_tags": [t["name"] for t in status["ai_status"].get("recent_topic_tags", [])],
        "last_updated": status["timestamp"]
    }

# 重置会话状态
def reset_session():
    """重置会话相关状态，保留长期状态"""
    update_status(
        context_notes={
            "thinking_focus": "无",
            "intent": "无", 
            "conversation_style": "正常",
            "session_context": ""
        },
        session_stats={
            "message_count": 0,
            "session_start": get_timestamp(),
            "last_interaction": get_timestamp()
        }
    )
    print("✅ 会话状态已重置")

# 与记忆系统集成的便捷函数
def sync_with_memory_system(memory_system):
    """与记忆系统同步状态信息"""
    status_summary = get_status_summary()
    
    # 可以将重要状态变化作为事实记忆保存
    if status_summary["relationship_level"] >= 7:
        recent_changes = get_relationship_history(3)
        if recent_changes:
            latest_change = recent_changes[-1]
            if latest_change["magnitude"] >= 0.5:
                memory_system.save_fact_memory(
                    content=f"关系发生重要变化：{latest_change['reason']}，亲密度从{latest_change['old_value']:.1f}变为{latest_change['new_value']:.1f}",
                    tags=["关系变化", "重要事件"],
                    source="state_system"
                )
    
    return status_summary

# 示例用法（测试）
if __name__ == "__main__":
    print("🧠 状态系统测试开始...")
    
    # 测试基本状态更新
    update_status(ai_status={
        "emotion": {"mood": "开心", "strength": 0.9},
        "user_attitude": {"emotional_feeling": "想念", "intimacy": 0.87},
    })
    
    # 测试用户状态更新
    update_user_status(
        last_emotion="专注",
        last_topic="状态系统的设计",
        current_mood="积极",
        energy_level=0.8
    )
    
    # 测试上下文注释更新
    update_context_notes(
        thinking_focus="是否将上下文保存为状态",
        intent="进一步完善AI状态系统",
        conversation_style="技术讨论",
        session_context="正在开发情感陪伴智能体"
    )
    
    # 测试标签管理
    add_tag("用户昨天没有说晚安")
    add_tag("亲密")
    edit_tag("亲密", "超级亲密")
    
    # 测试情绪和关系管理
    update_ai_emotion("兴奋", 0.95)
    update_user_attitude("喜欢", 0.1)
    update_relationship_level(0.3)
    
    # 测试会话统计
    increment_message_count()
    increment_message_count()
    
    # 获取状态摘要
    summary = get_status_summary()
    print(f"\n📊 当前关系等级: {summary['relationship_level']:.1f}")
    print(f"📝 关系描述: {summary['relationship_description']}")
    
    # 显示完整状态
    print("\n✅ 状态更新完成！当前状态如下：")
    print(json.dumps(load_status(), ensure_ascii=False, indent=2))
    
    # 显示关系变化历史
    history = get_relationship_history(3)
    if history:
        print(f"\n📈 最近的关系变化记录:")
        for record in history:
            print(f"  {record['readable_time']}: {record['reason']} (变化: {record['change']:.2f})")
    
    print("\n🎉 状态系统测试完成！")
