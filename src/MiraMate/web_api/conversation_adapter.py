"""
对话处理器适配器
连接重构后的LangChain架构和原有的API接口
"""

import asyncio
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

# 导入重构后的核心组件
from MiraMate.core.pipeline import final_chain, get_memory_for_session
from MiraMate.core.post_sync_chain import post_sync_chain
from MiraMate.core.post_async_chain import post_async_chain
from MiraMate.core.idle_processor import IdleProcessor
from MiraMate.modules.status_system import get_status_summary


class ConversationHandlerAdapter:
    """对话处理器适配器，提供与原有API兼容的接口"""
    
    def __init__(self, config_path: str = None):
        """
        初始化适配器
        
        Args:
            config_path: 配置文件路径（保持兼容性，实际不使用）
        """
        self.session_id = str(uuid.uuid4())  # 为Web API创建一个固定的session
        self.background_tasks_running = False
        self.idle_processor = None  # 后台任务处理器
        print(f"✅ ConversationHandlerAdapter 初始化完成，Session ID: {self.session_id}")
        
    async def get_response_with_commands(self, user_message: str, enable_timing: bool = False) -> Dict[str, Any]:
        """
        获取AI回复和视觉效果指令（兼容原有API接口，非流式）
        
        Args:
            user_message: 用户消息
            enable_timing: 是否启用时间统计
            
        Returns:
            包含回复文本和指令的字典
        """
        try:
            # 更新交互时间（用于IdleProcessor）
            self.update_interaction_time()
            
            start_time = datetime.now() if enable_timing else None
            
            # 获取当前对话历史（用于后处理）
            memory_instance = get_memory_for_session(self.session_id)
            history_before_turn = memory_instance.messages.copy()
            
            # 调用重构后的主对话链，获取AI回复
            full_response = ""
            async for chunk in final_chain.astream(
                {"user_input": user_message},
                config={"configurable": {"session_id": self.session_id}}
            ):
                full_response += chunk
            
            # 执行同步后处理（状态更新）
            try:
                sync_result = await post_sync_chain.ainvoke({
                    "conversation_history": history_before_turn,
                    "user_input": user_message,
                    "ai_response": full_response
                })
                print(f"[同步后处理] 状态更新: {sync_result.get('status', 'unknown')}")
            except Exception as e:
                print(f"[同步后处理] 状态更新失败: {e}")
            
            # 启动异步后处理（记忆处理）
            asyncio.create_task(
                post_async_chain.ainvoke({
                    "conversation_history": history_before_turn,
                    "user_input": user_message,
                    "ai_response": full_response
                })
            )
            
            # TODO: 视觉效果指令生成（暂时返回空列表）
            # 等视觉效果功能完成后再实现
            commands = []
            
            # 计算处理时间
            processing_time = None
            if enable_timing and start_time:
                processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "response": full_response,
                "commands": commands,
                "processing_time": processing_time
            }
            
        except Exception as e:
            print(f"❌ 对话处理失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回错误回复
            return {
                "response": "抱歉，我刚才走神了...能再说一遍吗？ 😅",
                "commands": [],
                "processing_time": None
            }
    
    async def get_response_stream(self, user_message: str, enable_timing: bool = False):
        """
        获取AI回复流式输出（新增流式接口）
        
        Args:
            user_message: 用户消息
            enable_timing: 是否启用时间统计
            
        Yields:
            流式响应数据块
        """
        try:
            # 更新交互时间（用于IdleProcessor）
            self.update_interaction_time()
            
            start_time = datetime.now() if enable_timing else None
            
            # 获取当前对话历史（用于后处理）
            memory_instance = get_memory_for_session(self.session_id)
            history_before_turn = memory_instance.messages.copy()
            
            full_response = ""
            chunk_count = 0
            
            # 流式输出AI回复
            async for chunk in final_chain.astream(
                {"user_input": user_message},
                config={"configurable": {"session_id": self.session_id}}
            ):
                chunk_count += 1
                print(f"[DEBUG] 收到流式块 {chunk_count}: '{chunk}' (类型: {type(chunk)}, 长度: {len(str(chunk))})")
                
                # 确保 chunk 是字符串
                chunk_str = str(chunk) if chunk is not None else ""
                full_response += chunk_str
                
                # 只有当块非空时才发送
                if chunk_str:
                    yield {
                        "type": "content",
                        "content": chunk_str,
                        "chunk_id": chunk_count,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    print(f"[DEBUG] 跳过空块 {chunk_count}")
            
            print(f"[DEBUG] 流式处理完成，总共 {chunk_count} 块，完整回复长度: {len(full_response)}")
            
            # 执行同步后处理（状态更新）
            try:
                sync_result = await post_sync_chain.ainvoke({
                    "conversation_history": history_before_turn,
                    "user_input": user_message,
                    "ai_response": full_response
                })
                print(f"[同步后处理] 状态更新: {sync_result.get('status', 'unknown')}")
            except Exception as e:
                print(f"[同步后处理] 状态更新失败: {e}")
            
            # 启动异步后处理（记忆处理）
            asyncio.create_task(
                post_async_chain.ainvoke({
                    "conversation_history": history_before_turn,
                    "user_input": user_message,
                    "ai_response": full_response
                })
            )
            
            # 获取情感状态
            emotional_state = self.get_current_emotional_state()
            
            # TODO: 视觉效果指令生成（暂时返回空列表）
            commands = []
            
            # 计算处理时间
            processing_time = None
            if enable_timing and start_time:
                processing_time = (datetime.now() - start_time).total_seconds()
            
            # 发送结束信号和元数据
            yield {
                "type": "metadata",
                "emotional_state": emotional_state,
                "commands": commands,
                "processing_time": processing_time,
                "full_response": full_response,
                "total_chunks": chunk_count,
                "timestamp": datetime.now().isoformat()
            }
            
            # 发送流结束信号
            yield {
                "type": "end",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ 流式对话处理失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 发送错误信息
            yield {
                "type": "error",
                "error": "对话处理失败",
                "message": "抱歉，我刚才走神了...能再说一遍吗？ 😅",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_current_emotional_state(self) -> Dict[str, Any]:
        """
        获取当前情感状态（兼容原有API接口）
        """
        try:
            # 从重构后的状态系统获取状态
            status_summary = get_status_summary()
            
            # 解析状态信息，提取情感相关数据
            ai_status = status_summary.get("ai_status", {})
            emotion_info = ai_status.get("emotion", {})
            
            # 转换为API期望的格式
            return {
                "current_emotion": emotion_info.get("mood", "neutral"),
                "emotion_intensity": float(emotion_info.get("strength", 0.5)),
                "relationship_level": ai_status.get("relationship_level", 5)
            }
            
        except Exception as e:
            print(f"❌ 获取情感状态失败: {e}")
            # 返回默认状态
            return {
                "current_emotion": "neutral",
                "emotion_intensity": 0.5,
                "relationship_level": 5
            }
    
    def start_background_tasks(self):
        """启动后台任务（包括IdleProcessor）"""
        if not self.background_tasks_running:
            self.background_tasks_running = True
            
            # 启动IdleProcessor - 20分钟空闲后处理记忆缓存
            self.idle_processor = IdleProcessor(idle_threshold_seconds=1200)
            self.idle_processor.start()
            
            print("✅ 后台任务启动完成 - IdleProcessor已启动（20分钟空闲阈值）")
        else:
            print("⚠️  后台任务已在运行中")
    
    def stop_background_tasks(self):
        """停止后台任务（包括IdleProcessor）"""
        if self.background_tasks_running:
            self.background_tasks_running = False
            
            # 停止IdleProcessor
            if self.idle_processor:
                self.idle_processor.stop()
                self.idle_processor = None
                print("✅ IdleProcessor已停止")
            
            print("✅ 后台任务停止完成")
        else:
            print("⚠️  后台任务未在运行")
    
    def update_interaction_time(self):
        """更新最后交互时间（用于IdleProcessor计时）"""
        if self.idle_processor and self.background_tasks_running:
            self.idle_processor.update_last_interaction_time()
