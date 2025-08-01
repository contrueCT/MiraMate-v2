from typing import Dict, List, Any
from collections import defaultdict

class MemoryCache:
    """
    一个基于会话的、采用“轮次衰减与再激活”策略的记忆缓存。
    它被设计为支持“先更新，后获取并衰减”的清晰工作流。
    """
    def __init__(self, default_ttl_turns: int = 5):
        """
        :param default_ttl_turns: 记忆在缓存中的默认存活轮次。
        """
        self.default_ttl_turns = default_ttl_turns
        # 缓存结构: { session_id: { memory_id: {"memory": {...}, "ttl_turns": N} } }
        self.caches: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    def get_and_decay(self, session_id: str) -> List[Dict]:
        """
        核心方法：获取当前会话的所有有效记忆，并对所有记忆的生命周期执行一次“衰减”。
        此方法应该在将新记忆添加到缓存 *之后* 调用。
        """
        if session_id not in self.caches:
            return []

        session_cache = self.caches[session_id]
        active_memories = []
        next_turn_cache = {}

        # 遍历当前缓存，筛选有效记忆，并准备下一轮的缓存
        for mem_id, cache_item in session_cache.items():
            # 步骤1: 判断本轮是否有效。只要TTL大于0，就将其视为有效记忆。
            if cache_item["ttl_turns"] > 0:
                active_memories.append(cache_item["memory"])
            
            # 步骤2: 为下一轮准备，对TTL进行衰减。
            new_ttl = cache_item["ttl_turns"] - 1
            if new_ttl > 0:
                # 如果衰减后生命周期仍然大于0，则保留到下一轮的缓存中。
                next_turn_cache[mem_id] = {
                    "memory": cache_item["memory"],
                    "ttl_turns": new_ttl
                }

        # 步骤3: 用衰减后的新缓存替换旧缓存。
        self.caches[session_id] = next_turn_cache
        
        print(f"[MemoryCache] Session {session_id[:8]}: 返回 {len(active_memories)} 条有效记忆, 衰减后 {len(next_turn_cache)} 条将留存至下一轮。")
        return active_memories

    def add_or_reactivate(self, session_id: str, new_memories: List[Dict]):
        """
        将新检索到的记忆加入缓存，或重置已存在记忆的生命周期（再激活）。
        """
        session_cache = self.caches[session_id]
        if not new_memories:
            return

        print(f"[MemoryCache] Session {session_id[:8]}: 添加/再激活 {len(new_memories)} 条记忆。")
        for memory in new_memories:
            mem_id = memory.get("id")
            if not mem_id:
                continue
            
            # 直接用满额的生命周期覆盖或创建条目。
            session_cache[mem_id] = {
                "memory": memory,
                "ttl_turns": self.default_ttl_turns
            }

# 创建一个全局的缓存实例
memory_cache = MemoryCache()