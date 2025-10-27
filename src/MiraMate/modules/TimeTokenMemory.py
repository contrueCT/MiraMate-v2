import time
from datetime import datetime
from collections import deque
from typing import List, Dict, Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory

from MiraMate.modules.memory_system import format_natural_time

# 导入tiktoken用于自动计算token
import tiktoken

class CustomTokenMemory(BaseChatMessageHistory):
    """
    一个高性能、自管理的聊天历史记录类，实现了 BaseChatMessageHistory 接口。
    """

    # TODO 计算token数量的分词器选择需后续优化实现
    def __init__(self, 
                 llm_model_name: str = "gpt-4o",
                 max_token_limit: int = 100000, 
                 retention_time: int = 3600, 
                 continuity_threshold: int = 600, 
                 min_conversation_to_keep: int = 40):
        """
        标准的Python构造函数，直接初始化所有属性。
        """
        self.llm_model_name = llm_model_name
        self.max_token_limit = max_token_limit
        self.retention_time = retention_time
        self.continuity_threshold = continuity_threshold
        self.min_conversation_to_keep = min_conversation_to_keep

        # 内部状态
        self.memory: deque[Dict[str, Any]] = deque()
        self.total_token_count: int = 0
        
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.llm_model_name)
        except KeyError:
            print(f"Warning: model '{self.llm_model_name}' not found. Using 'cl100k_base'.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    # --- BaseChatMessageHistory 接口实现 ---

    @property
    def messages(self) -> List[BaseMessage]:
        """以属性形式返回当前有效的消息列表。"""
        self._manage_memory()
        return [item['message'] for item in self.memory]

    @messages.setter
    def messages(self, messages: List[BaseMessage]) -> None:
        """允许外部直接设置消息历史。"""
        self.clear()
        self.add_messages(messages)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """同步添加消息。"""
        for message in messages:
            if not isinstance(message, (HumanMessage, AIMessage)):
                continue
            token_count = len(self.tokenizer.encode(message.content))
            self._add_message(message, token_count)

    async def aadd_messages(self, messages: List[BaseMessage]) -> None:
        """异步添加消息。对于内存操作，其逻辑与同步版本相同。"""
        self.add_messages(messages)

    def clear(self) -> None:
        """清空所有记忆。"""
        self.memory.clear()
        self.total_token_count = 0

    # --- 内部辅助方法 ---

    def _add_message(self, message: BaseMessage, token_count: int) -> None:
        ts = time.time()
        new_item = {
            "message": f"{message.content} 【{format_natural_time(datetime.fromtimestamp(ts))}】",
            "timestamp": ts,
            "token_count": token_count
        }
        self.memory.append(new_item)
        self.total_token_count += token_count
        
    def _manage_memory(self) -> None:
        current_time = time.time()
        current_dt = datetime.fromtimestamp(current_time)
        min_messages_to_keep = self.min_conversation_to_keep * 2

        # 规则一：当总token超限时，按先进先出丢弃，保留至少min_messages_to_keep条
        while self.total_token_count > self.max_token_limit and len(self.memory) > min_messages_to_keep:
            removed = self.memory.popleft()
            self.total_token_count -= removed["token_count"]

        # 规则二（新增）：跨日且超过8小时的历史消息，强制删除（即便低于最低保留数）
        eight_hours = 8 * 3600
        if self.memory:
            survivors_forced = deque()
            removed_tokens = 0
            for item in self.memory:
                ts = item["timestamp"]
                item_dt = datetime.fromtimestamp(ts)
                is_cross_day = (item_dt.date() != current_dt.date())
                is_older_than_8h = (current_time - ts) > eight_hours
                if is_cross_day and is_older_than_8h:
                    removed_tokens += item["token_count"]
                    continue  # 强制删除
                survivors_forced.append(item)
            if len(survivors_forced) != len(self.memory):
                self.memory = survivors_forced
                self.total_token_count -= removed_tokens

        if len(self.memory) <= min_messages_to_keep:
            return

        survivors = deque()
        for i, item in enumerate(self.memory):
            if (current_time - item["timestamp"]) <= self.retention_time:
                survivors.append(item)
                continue

            is_part_of_continuity = False
            if i + 1 < len(self.memory):
                if (self.memory[i+1]["timestamp"] - item["timestamp"]) <= self.continuity_threshold:
                    is_part_of_continuity = True
            
            if survivors:
                if (item["timestamp"] - survivors[-1]["timestamp"]) <= self.continuity_threshold:
                    is_part_of_continuity = True
            
            if is_part_of_continuity:
                survivors.append(item)

        if len(survivors) < min_messages_to_keep:
            original_items = list(self.memory)
            survivors = deque(original_items[-min_messages_to_keep:])

        self.memory = survivors
        self.total_token_count = sum(item["token_count"] for item in self.memory)