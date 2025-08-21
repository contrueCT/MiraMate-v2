"""
Web API 数据模型定义
"""

from token import OP
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from datetime import datetime


class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str
    enable_timing: bool = True
    stream: bool = False  # 新增：是否使用流式输出


class ChatResponse(BaseModel):
    """聊天响应模型"""
    response: str
    timestamp: datetime
    emotional_state: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    commands: Optional[List[Dict[str, Any]]] = None  # 新增：视觉效果指令列表


class StreamChunk(BaseModel):
    """流式响应数据块模型"""
    type: str  # "content", "metadata", "end", "error"
    content: Optional[str] = None  # 内容块（type="content"时使用）
    chunk_id: Optional[int] = None  # 块ID
    emotional_state: Optional[Dict[str, Any]] = None  # 情感状态（type="metadata"时使用）
    commands: Optional[List[Dict[str, Any]]] = None  # 视觉效果指令
    processing_time: Optional[float] = None  # 处理时间
    full_response: Optional[str] = None  # 完整响应（type="metadata"时使用）
    total_chunks: Optional[int] = None  # 总块数
    error: Optional[str] = None  # 错误类型（type="error"时使用）
    message: Optional[str] = None  # 错误消息（type="error"时使用）
    timestamp: str


class EmotionalState(BaseModel):
    """情感状态模型"""
    current_emotion: str
    emotion_intensity: float
    relationship_level: int
    last_updated: Optional[datetime] = None


class ChatHistoryItem(BaseModel):
    """聊天历史项目"""
    id: str
    user_message: str
    ai_response: str
    timestamp: datetime
    emotional_state: Optional[Dict[str, Any]] = None


class ChatHistory(BaseModel):
    """聊天历史模型"""
    items: List[ChatHistoryItem]
    total_count: int
    has_more: bool


# ===== 配置管理相关模型 =====

class LLMConfig(BaseModel):
    """LLM配置模型"""
    model: str
    api_key: str
    base_url: Optional[str] = ""
    api_type: str = "openai"
    model_kwargs: Optional[Dict[str, Any]] = None


class EnvironmentConfig(BaseModel):
    """环境配置模型"""
    user_name: str = "小伙伴"
    agent_name: str = "小梦"
    agent_description: str


class UserPreferences(BaseModel):
    """用户偏好配置"""
    theme: str = "default"
    visual_effects_enabled: bool = True
    notification_enabled: bool = True
    language: str = "zh-CN"


class SystemConfig(BaseModel):
    """系统配置模型"""
    llm_configs: List[LLMConfig]
    environment: EnvironmentConfig
    user_preferences: UserPreferences
    last_updated: Optional[datetime] = None


class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    config_type: str  # "llm", "environment", "preferences"
    config_data: Dict[str, Any]


class ConfigResponse(BaseModel):
    """配置响应模型"""
    success: bool
    message: str
    config: Optional[Dict[str, Any]] = None


class HealthStatus(BaseModel):
    """健康检查模型"""
    status: str
    timestamp: datetime
    version: str = "1.0.0"
    uptime: float
    services: Dict[str, str]


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str
    message: str
    timestamp: datetime
    status_code: int
