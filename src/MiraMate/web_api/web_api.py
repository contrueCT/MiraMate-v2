"""
情感陪伴AI系统 Web API 服务器
使用FastAPI提供RESTful API接口，连接前端Web界面与后端ConversationHandler
"""

import os
import sys
import time
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

# 添加项目根目录到Python路径
def get_project_root():
    """基于项目结构自动推断项目根目录（包含 pyproject.toml 且有 src/MiraMate）。"""
    current = os.path.abspath(__file__)
    p = os.path.dirname(current)
    # 往上查找带 pyproject.toml 且包含 src/MiraMate 的目录
    for _ in range(6):
        candidate = p
        if (os.path.exists(os.path.join(candidate, 'pyproject.toml')) and
                os.path.exists(os.path.join(candidate, 'src', 'MiraMate'))):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        p = parent
    # 兜底：返回四级上层（与原注释等价）
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current))))

project_root = get_project_root()
sys.path.insert(0, project_root)

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
import logging

from MiraMate.web_api.conversation_adapter import ConversationHandlerAdapter
from MiraMate.web_api.config_manager import ConfigManager
from MiraMate.web_api.websocket_handler import ws_manager, proactive_service, start_proactive_service
from MiraMate.web_api import auth
from MiraMate.web_api.models import (
    ChatRequest, ChatResponse, EmotionalState, 
    ChatHistory, ChatHistoryItem, HealthStatus, ErrorResponse,
    LLMConfig, EnvironmentConfig, UserPreferences, SystemConfig,
    ConfigResponse, StreamChunk
)


class WebAPIServer:
    """Web API 服务器类"""
    
    def __init__(self):
        self.conversation_handler: Optional[ConversationHandlerAdapter] = None
        self.start_time = time.time()
        self.chat_history: List[ChatHistoryItem] = []
        self.max_history_size = 1000
        # 传递项目根目录给配置管理器
        self.config_manager = ConfigManager(project_root)
        
    async def initialize(self):
        """初始化ConversationHandler和WebSocket服务"""
        try:
            # 使用新的配置文件名称和路径
            config_path = os.path.join(
                os.getenv('CONFIG_DIR', os.path.join(project_root, "configs")), 
                "llm_config.json"  # 重构后使用新的配置文件名
            )
            
            # 检查配置文件是否有有效的API密钥
            if self._has_valid_api_keys(config_path):
                # 使用新的适配器替代原有的ConversationHandler
                self.conversation_handler = ConversationHandlerAdapter(config_path)
                
                # 启动后台任务
                self.conversation_handler.start_background_tasks()
                
                print(f"✅ ConversationHandlerAdapter初始化成功")
                print(f"✅ 配置文件: {config_path}")
            else:
                print(f"⚠️  API配置不完整，ConversationHandler暂未初始化")
                print(f"💡 可通过Web界面配置API密钥后重启服务")
                self.conversation_handler = None
            
            # TODO: 主动消息功能未完成，暂时禁用
            # # 启动WebSocket主动消息服务
            # await start_proactive_service()
            # print(f"✅ WebSocket主动消息服务启动成功")
            
        except Exception as e:
            print(f"⚠️  服务初始化失败: {e}")
            print(f"💡 Web服务器仍将启动，可通过界面配置后重启")
            self.conversation_handler = None
    
    def _has_valid_api_keys(self, config_path: str) -> bool:
        """检查是否有有效的API密钥（适配新的配置文件格式）"""
        try:
            if not os.path.exists(config_path):
                print(f"❌ 配置文件不存在: {config_path}")
                return False
                
            with open(config_path, 'r', encoding='utf-8') as f:
                import json
                configs = json.load(f)
                
            if not configs:
                print(f"❌ 配置文件为空")
                return False
                
            # 检查是否有有效的API密钥
            valid_configs = 0
            for config in configs:
                api_key = config.get("api_key", "").strip()
                if api_key and api_key != "":
                    valid_configs += 1
            
            if valid_configs >= 2:  # 至少需要主模型和小模型的配置
                print(f"✅ 找到 {valid_configs} 个有效的API配置")
                return True
            else:
                print(f"❌ 需要至少2个有效的API配置，当前只有 {valid_configs} 个")
                return False
                
        except Exception as e:
            print(f"❌ 检查配置时出错: {e}")
            return False
    
    async def cleanup(self):
        """清理资源"""
        if self.conversation_handler:
            self.conversation_handler.stop_background_tasks()
            print("✅ 后台任务已停止")
        
        # 停止WebSocket主动消息服务
        from MiraMate.web_api.websocket_handler import proactive_service
        await proactive_service.stop()
        print("✅ WebSocket服务已停止")


# 创建全局服务器实例
server = WebAPIServer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    await server.initialize()
    yield
    # 关闭时清理
    await server.cleanup()


# 创建FastAPI应用
app = FastAPI(
    title="情感陪伴AI Web API",
    description="小梦情感陪伴AI系统的Web API接口",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP 鉴权中间件（仅当设置 MIRAMATE_AUTH_TOKEN 时启用；白名单仅 /api/health）
@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    try:
        auth.verify_http_request(request)
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail},
            headers=getattr(e, "headers", None),
        )
    return await call_next(request)

# 注意：此应用使用Electron客户端，不直接提供web前端
# mira-desktop/web 目录中的文件是给Electron客户端使用的
print("💡 此应用使用Electron客户端，不提供直接的web前端访问")
print(f"🖥️  Electron客户端文件位于: {os.path.join(project_root, 'mira-desktop')}")


# ===== WebSocket端点 =====

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket连接端点"""
    # 在建立连接前进行 WebSocket 鉴权（仅当设置 MIRAMATE_AUTH_TOKEN）
    if auth.is_auth_enabled():
        token = websocket.query_params.get("token")
        if not auth.verify_token_value(token):
            await websocket.close(code=1008, reason="Unauthorized")
            return
    # 建立连接
    if not await ws_manager.connect(websocket):
        return
    
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            message = ws_manager.validate_message(data)
            
            if not message:
                await ws_manager.send_message(websocket, {
                    "type": "error",
                    "data": "消息格式错误",
                    "timestamp": time.time()
                })
                continue
            
            # 处理不同类型的消息
            await handle_websocket_message(websocket, message)
                
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logging.info("WebSocket客户端主动断开连接")
    except Exception as e:
        logging.error(f"WebSocket处理异常: {e}")
        ws_manager.disconnect(websocket)


async def handle_websocket_message(websocket: WebSocket, message: dict):
    """处理WebSocket消息"""
    message_type = message.get("type")
    message_data = message.get("data", "")
    
    try:
        if message_type == "chat":
            # 处理聊天消息
            await handle_chat_message(websocket, message_data)
            
        elif message_type == "ping":
            # 处理心跳检测
            await ws_manager.send_message(websocket, {
                "type": "pong",
                "timestamp": time.time()
            })
            
        elif message_type == "get_emotional_state":
            # 获取情感状态
            await handle_emotional_state_request(websocket)
            
        else:
            # 未知消息类型
            await ws_manager.send_message(websocket, {
                "type": "error",
                "data": f"未知的消息类型: {message_type}",
                "timestamp": time.time()
            })
            
    except Exception as e:
        logging.error(f"处理消息失败 [{message_type}]: {e}")
        await ws_manager.send_message(websocket, {
            "type": "error",
            "data": "处理消息时发生错误",
            "timestamp": time.time()
        })


async def handle_chat_message(websocket: WebSocket, user_message: str):
    """处理聊天消息 - 使用流式输出"""
    if not user_message.strip():
        await ws_manager.send_message(websocket, {
            "type": "chat_response",
            "data": {
                "response": "消息不能为空哦～",
                "emotional_state": None,
                "commands": []
            },
            "timestamp": time.time()
        })
        return
    
    # 更新最后消息时间（用于主动消息服务）
    proactive_service.update_last_message_time()
    
    if server.conversation_handler:
        try:
            # 使用流式处理
            full_response = ""
            emotional_state = None
            commands = []
            start_time = datetime.now()
            
            # 发送开始流式传输消息
            await ws_manager.send_message(websocket, {
                "type": "chat_stream_start",
                "timestamp": time.time()
            })
            
            async for chunk_data in server.conversation_handler.get_response_stream(user_message, enable_timing=True):
                chunk_type = chunk_data.get('type')
                
                if chunk_type == "content":
                    # 累积完整响应
                    content = chunk_data.get('content', '')
                    full_response += content
                    
                    # 发送内容块
                    await ws_manager.send_message(websocket, {
                        "type": "chat_stream_chunk",
                        "data": {
                            "content": content,
                            "chunk_id": chunk_data.get('chunk_id')
                        },
                        "timestamp": time.time()
                    })
                    
                elif chunk_type == "metadata":
                    emotional_state = chunk_data.get('emotional_state')
                    commands = chunk_data.get('commands', [])
                    
                elif chunk_type == "end":
                    # 发送完整响应（兼容原有客户端）
                    await ws_manager.send_message(websocket, {
                        "type": "chat_response",
                        "data": {
                            "response": full_response,
                            "emotional_state": emotional_state,
                            "commands": commands
                        },
                        "timestamp": time.time()
                    })
                    
                    # 发送流结束消息
                    await ws_manager.send_message(websocket, {
                        "type": "chat_stream_end",
                        "data": {
                            "total_response": full_response,
                            "processing_complete": True
                        },
                        "timestamp": time.time()
                    })
                    break
                    
                elif chunk_type == "error":
                    await ws_manager.send_message(websocket, {
                        "type": "error",
                        "data": chunk_data.get('message', '处理消息时发生错误'),
                        "timestamp": time.time()
                    })
                    break
            
            # 添加到聊天历史
            if full_response:
                history_item = ChatHistoryItem(
                    id=str(uuid.uuid4()),
                    user_message=user_message,
                    ai_response=full_response,
                    timestamp=start_time,
                    emotional_state=emotional_state
                )
                
                server.chat_history.append(history_item)
                
                # 限制历史记录数量
                if len(server.chat_history) > server.max_history_size:
                    server.chat_history = server.chat_history[-server.max_history_size:]
                
        except Exception as e:
            logging.error(f"WebSocket聊天处理失败: {e}")
            await ws_manager.send_message(websocket, {
                "type": "error",
                "data": "处理消息时发生错误",
                "timestamp": time.time()
            })
    else:
        # AI系统未初始化时的回复
        await ws_manager.send_message(websocket, {
            "type": "chat_response",
            "data": {
                "response": "系统正在初始化中，请稍候再试。或者你可以通过设置页面配置API密钥后重启服务～",
                "emotional_state": {
                    "current_emotion": "neutral",
                    "emotion_intensity": 0.5,
                    "relationship_level": 1
                },
                "commands": []
            },
            "timestamp": time.time()
        })


async def handle_emotional_state_request(websocket: WebSocket):
    """处理获取情感状态请求"""
    if server.conversation_handler:
        try:
            emotional_state = server.conversation_handler.get_current_emotional_state()
            await ws_manager.send_message(websocket, {
                "type": "emotional_state",
                "data": emotional_state,
                "timestamp": time.time()
            })
        except Exception as e:
            logging.error(f"获取情感状态失败: {e}")
            await ws_manager.send_message(websocket, {
                "type": "error",
                "data": "获取情感状态失败",
                "timestamp": time.time()
            })
    else:
        await ws_manager.send_message(websocket, {
            "type": "emotional_state",
            "data": {
                "current_emotion": "neutral",
                "emotion_intensity": 0.5,
                "relationship_level": 1
            },
            "timestamp": time.time()
        })


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    error_response = ErrorResponse(
        error="Internal Server Error",
        message=str(exc),
        timestamp=datetime.now(),
        status_code=500
    )
    return JSONResponse(
        status_code=500,
        content=jsonable_encoder(error_response)
    )


@app.get("/", response_model=dict)
async def root():
    """根路径 - 返回API信息"""
    return {
        "name": "情感陪伴AI Web API",
        "version": "1.0.0",
        "description": "小梦情感陪伴AI系统的Web API接口 - 为Electron客户端提供服务",
        "docs_url": "/docs",
        "client_type": "electron",
        "endpoints": {
            "chat": "/api/chat",
            "emotional_state": "/api/emotional-state",
            "chat_history": "/api/chat/history",
            "health": "/api/health",
            "config": "/api/config"
        },
        "features": {
            "streaming": "支持流式聊天输出，设置 stream=true",
            "timing": "支持处理时间统计，设置 enable_timing=true",
            "emotional_state": "实时情感状态跟踪",
            "visual_effects": "视觉效果指令生成（开发中）"
        }
    }


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    聊天接口 - 支持流式和非流式输出
    """
    if not server.conversation_handler:
        error_detail = {
            "error": "ConversationHandler未初始化",
            "message": "请先配置API密钥后重启服务",
            "config_url": "/api/config",
            "suggestions": [
                "1. 通过Electron客户端配置界面设置API密钥",
                "2. 直接编辑配置文件后重启服务",
                "3. 检查API密钥是否正确填写"
            ]
        }
        
        if request.stream:
            # 流式模式下返回SSE格式的错误
            async def error_stream():
                error_chunk = StreamChunk(
                    type="error",
                    error="service_unavailable",
                    message="ConversationHandler未初始化，请先配置API密钥",
                    timestamp=datetime.now().isoformat()
                )
                yield f"data: {error_chunk.model_dump_json()}\n\n"
            
            return StreamingResponse(
                error_stream(),
                media_type="text/event-stream",
                status_code=503
            )
        else:
            raise HTTPException(status_code=503, detail=error_detail)
    
    if request.stream:
        # 流式响应
        return StreamingResponse(
            generate_chat_stream(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用nginx缓冲
            }
        )
    else:
        # 非流式响应（保持原有逻辑）
        return await handle_non_stream_chat(request)


async def generate_chat_stream(request: ChatRequest):
    """生成流式聊天响应"""
    try:
        # 记录开始时间用于历史记录
        start_time = datetime.now()
        full_response = ""
        emotional_state = None
        commands = []
        processing_time = None
        
        print(f"[DEBUG] 开始流式处理消息: '{request.message}'")
        
        # 发送流式数据
        async for chunk_data in server.conversation_handler.get_response_stream(
            request.message,
            enable_timing=request.enable_timing
        ):
            print(f"[DEBUG] 从适配器收到数据块: {chunk_data}")
            
            # 直接使用字典创建 StreamChunk，避免 Pydantic 序列化问题
            chunk_type = chunk_data.get('type')
            
            # 累积完整响应用于历史记录
            if chunk_type == "content":
                content = chunk_data.get('content', '')
                full_response += content
                
            elif chunk_type == "metadata":
                emotional_state = chunk_data.get('emotional_state')
                commands = chunk_data.get('commands', [])
                processing_time = chunk_data.get('processing_time')
            
            # 直接序列化字典而不是通过 Pydantic 模型
            try:
                json_data = json.dumps(chunk_data, ensure_ascii=False)
                sse_line = f"data: {json_data}\n\n"
                print(f"[DEBUG] 发送 SSE 数据: {sse_line.strip()}")
                yield sse_line
            except Exception as e:
                print(f"[DEBUG] JSON序列化失败: {e}, 数据: {chunk_data}")
                continue
        
        print(f"[DEBUG] 流式处理完成，完整回复: '{full_response}'")
        
        # 添加到聊天历史（在流结束后）
        if full_response:
            chat_id = str(uuid.uuid4())
            chat_item = ChatHistoryItem(
                id=chat_id,
                user_message=request.message,
                ai_response=full_response,
                timestamp=start_time,
                emotional_state=emotional_state
            )
            
            server.chat_history.append(chat_item)
            # 限制历史记录大小
            if len(server.chat_history) > server.max_history_size:
                server.chat_history = server.chat_history[-server.max_history_size:]
        
    except Exception as e:
        print(f"[DEBUG] 流式处理异常: {e}")
        import traceback
        traceback.print_exc()
        
        # 发送错误信息
        error_data = {
            "type": "error",
            "error": "processing_error", 
            "message": f"处理聊天消息时发生错误: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        try:
            json_data = json.dumps(error_data, ensure_ascii=False)
            yield f"data: {json_data}\n\n"
        except:
            yield f"data: {{\"type\": \"error\", \"message\": \"Unknown error\"}}\n\n"


async def handle_non_stream_chat(request: ChatRequest) -> JSONResponse:
    """处理非流式聊天请求（原有逻辑）"""
    try:
        start_time = time.time()
        
        # 获取AI回复（包含视觉效果指令）
        response_data = await server.conversation_handler.get_response_with_commands(
            request.message, 
            enable_timing=request.enable_timing
        )
        
        processing_time = time.time() - start_time
        
        # 从响应数据中提取回复文本和指令
        ai_response = response_data.get("response", "")
        commands = response_data.get("commands", [])
        
        # 为指令添加时间戳
        for command in commands:
            command["timestamp"] = datetime.now().isoformat()
        
        # 获取当前情感状态
        emotional_state = server.conversation_handler.get_current_emotional_state()
        
        # 生成聊天记录ID
        chat_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # 添加到聊天历史
        chat_item = ChatHistoryItem(
            id=chat_id,
            user_message=request.message,
            ai_response=ai_response,
            timestamp=timestamp,
            emotional_state=emotional_state
        )
        
        server.chat_history.append(chat_item)
        # 限制历史记录大小
        if len(server.chat_history) > server.max_history_size:
            server.chat_history = server.chat_history[-server.max_history_size:]

        chat_response = ChatResponse(
            response=ai_response,
            timestamp=timestamp,
            emotional_state=emotional_state,
            processing_time=processing_time if request.enable_timing else None,
            commands=commands if commands else None
        )
        
        return JSONResponse(content=jsonable_encoder(chat_response))
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"处理聊天消息时发生错误: {str(e)}"
        )


@app.get("/api/emotional-state", response_model=EmotionalState)
async def get_emotional_state():
    """
    获取当前情感状态
    """
    if not server.conversation_handler:
        raise HTTPException(
            status_code=503, 
            detail="ConversationHandler未初始化"
        )
    
    try:
        state = server.conversation_handler.get_current_emotional_state()
        
        return EmotionalState(
            current_emotion=state.get('current_emotion', 'neutral'),
            emotion_intensity=state.get('emotion_intensity', 0.5),
            relationship_level=state.get('relationship_level', 1),
            last_updated=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取情感状态时发生错误: {str(e)}"
        )


@app.get("/api/chat/history", response_model=ChatHistory)
async def get_chat_history(
    limit: int = 50,
    offset: int = 0,
    reverse: bool = True
):
    """
    获取聊天历史记录
    
    Args:
        limit: 返回记录数量限制 (默认50)
        offset: 偏移量 (默认0)
        reverse: 是否倒序返回 (默认True，最新的在前)
    """
    try:
        total_count = len(server.chat_history)
        
        # 处理倒序
        history = list(reversed(server.chat_history)) if reverse else server.chat_history
        
        # 应用分页
        start_idx = offset
        end_idx = offset + limit
        
        items = history[start_idx:end_idx]
        has_more = end_idx < total_count
        
        return ChatHistory(
            items=items,
            total_count=total_count,
            has_more=has_more
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取聊天历史时发生错误: {str(e)}"
        )


@app.delete("/api/chat/history")
async def clear_chat_history():
    """
    清空聊天历史记录
    """
    try:
        server.chat_history.clear()
        response_data = {"message": "聊天历史已清空", "timestamp": datetime.now()}
        return JSONResponse(content=jsonable_encoder(response_data))
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"清空聊天历史时发生错误: {str(e)}"
        )


@app.get("/api/health", response_model=HealthStatus)
async def health_check():
    """
    健康检查接口 (包含WebSocket状态)
    """
    uptime = time.time() - server.start_time
    # 检查API配置状态 - 使用新的配置文件路径
    config_path = os.path.join(
        os.getenv('CONFIG_DIR', os.path.join(project_root, "configs")), 
        "llm_config.json"  # 使用新的配置文件名
    )
    has_valid_keys = server._has_valid_api_keys(config_path)
    
    services = {
        "conversation_handler": "healthy" if server.conversation_handler else "not_configured",
        "chat_history": "healthy",
        "api_server": "healthy",
        "api_config": "healthy" if has_valid_keys else "needs_configuration",
        "websocket_service": "healthy",
        "websocket_connections": str(ws_manager.get_connection_count()),
        "proactive_service": "running" if proactive_service.is_running else "stopped",
        "background_tasks": "running" if (server.conversation_handler and server.conversation_handler.background_tasks_running) else "stopped",
        "idle_processor": "active" if (server.conversation_handler and server.conversation_handler.idle_processor) else "inactive"
    }
      # 如果ConversationHandler未初始化但是服务器运行正常，仍然返回部分可用状态
    overall_status = "healthy" if server.conversation_handler else "partial"
    
    health_status = HealthStatus(
        status=overall_status,
        timestamp=datetime.now(),
        version="1.0.0",
        uptime=uptime,
        services=services
    )
    
    # 使用jsonable_encoder确保datetime对象正确序列化
    return JSONResponse(content=jsonable_encoder(health_status))


@app.get("/api/stats")
async def get_stats():
    """
    获取系统统计信息 (包含WebSocket状态)
    """
    try:
        uptime = time.time() - server.start_time
        
        stats_data = {
            "uptime_seconds": uptime,
            "uptime_formatted": str(timedelta(seconds=int(uptime))),
            "chat_history_count": len(server.chat_history),
            "max_history_size": server.max_history_size,
            "conversation_handler_status": "initialized" if server.conversation_handler else "not_initialized",
            "websocket_connections": ws_manager.get_connection_count(),
            "proactive_service_running": proactive_service.is_running,
            "proactive_last_message": proactive_service.last_message_time.isoformat() if hasattr(proactive_service, 'last_message_time') else None,
            "timestamp": datetime.now()
        }
        
        return JSONResponse(content=jsonable_encoder(stats_data))
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取统计信息时发生错误: {str(e)}"
        )


# 新增WebSocket状态查询接口
@app.get("/api/websocket/status")
async def get_websocket_status():
    """
    获取WebSocket服务状态详细信息
    """
    try:
        status_data = {
            "service_running": True,
            "active_connections": ws_manager.get_connection_count(),
            "max_connections": ws_manager.max_connections,
            "proactive_service": {
                "running": proactive_service.is_running,
                "check_interval": proactive_service.check_interval,
                "idle_threshold": proactive_service.idle_threshold,
                "last_message_time": proactive_service.last_message_time.isoformat(),
                "total_messages": len(proactive_service.proactive_messages)
            },
            "timestamp": datetime.now()
        }
        
        return JSONResponse(content=jsonable_encoder(status_data))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取WebSocket状态失败: {str(e)}"
        )


# ===== 配置管理接口 =====

@app.get("/api/config", response_model=SystemConfig)
async def get_system_config():
    """
    获取完整的系统配置
    """
    try:
        config = server.config_manager.get_system_config()
        return config
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取系统配置失败: {str(e)}"
        )


@app.get("/api/config/llm", response_model=List[LLMConfig])
async def get_llm_configs():
    """
    获取LLM配置列表
    """
    try:
        configs = server.config_manager.get_llm_configs()
        return configs
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取LLM配置失败: {str(e)}"
        )


@app.post("/api/config/llm", response_model=ConfigResponse)
async def update_llm_configs(configs: List[LLMConfig]):
    """
    更新LLM配置列表
    """
    try:
        # 验证配置
        for config in configs:
            is_valid, message = server.config_manager.validate_llm_config(config)
            if not is_valid:
                return ConfigResponse(
                    success=False,
                    message=f"配置验证失败: {message}",
                    config=None
                )
        
        # 保存配置
        success = server.config_manager.save_llm_configs(configs)
        
        if success:
            # 重新初始化ConversationHandler以使用新配置
            try:
                await server.initialize()
                print("✅ 配置更新后ConversationHandler重新初始化成功")
            except Exception as e:
                print(f"⚠️ 重新初始化ConversationHandler失败: {e}")
            
            return ConfigResponse(
                success=True,
                message="LLM配置更新成功，系统已重新初始化",
                config={"configs": [config.dict() for config in configs]}
            )
        else:
            return ConfigResponse(
                success=False,
                message="LLM配置保存失败",
                config=None
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"更新LLM配置失败: {str(e)}"
        )


@app.post("/api/config/llm/test", response_model=dict)
async def test_llm_config(config: LLMConfig):
    """
    测试LLM配置连接
    """
    try:
        is_valid, message = server.config_manager.validate_llm_config(config)
        if not is_valid:
            return {
                "success": False,
                "message": f"配置验证失败: {message}"
            }
        
        success, test_message = server.config_manager.test_llm_connection(config)
        return {
            "success": success,
            "message": test_message
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"测试连接失败: {str(e)}"
        }


@app.get("/api/config/environment", response_model=EnvironmentConfig)
async def get_environment_config():
    """
    获取环境配置
    """
    try:
        config = server.config_manager.get_environment_config()
        return config
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取环境配置失败: {str(e)}"
        )


@app.post("/api/config/environment", response_model=ConfigResponse)
async def update_environment_config(config: EnvironmentConfig):
    """
    更新环境配置
    """
    try:
        success = server.config_manager.save_environment_config(config)
        
        if success:
            # 保存成功后，立即尝试重新初始化系统，使配置即时生效
            try:
                await server.initialize()
                init_msg = "系统已重新初始化"
            except Exception as e:
                init_msg = f"重新初始化失败: {e}"
            return ConfigResponse(
                success=True,
                message=f"环境配置更新成功，{init_msg}",
                config=config.dict()
            )
        else:
            return ConfigResponse(
                success=False,
                message="环境配置保存失败",
                config=None
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"更新环境配置失败: {str(e)}"
        )


@app.get("/api/config/preferences", response_model=UserPreferences)
async def get_user_preferences():
    """
    获取用户偏好配置
    """
    try:
        preferences = server.config_manager.get_user_preferences()
        return preferences
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取用户偏好失败: {str(e)}"
        )


@app.post("/api/config/preferences", response_model=ConfigResponse)
async def update_user_preferences(preferences: UserPreferences):
    """
    更新用户偏好配置
    """
    try:
        success = server.config_manager.save_user_preferences(preferences)
        
        if success:
            return ConfigResponse(
                success=True,
                message="用户偏好更新成功",
                config=preferences.dict()
            )
        else:
            return ConfigResponse(
                success=False,
                message="用户偏好保存失败",
                config=None
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"更新用户偏好失败: {str(e)}"
        )


@app.post("/api/config/backup", response_model=dict)
async def backup_configs():
    """
    备份所有配置文件
    """
    try:
        backup_path = server.config_manager.backup_configs()
        
        if backup_path:
            return {
                "success": True,
                "message": "配置备份成功",
                "backup_path": backup_path
            }
        else:
            return {
                "success": False,
                "message": "配置备份失败"
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"备份配置失败: {str(e)}"
        )


@app.post("/api/config/restore", response_model=ConfigResponse)
async def restore_configs(backup_path: str):
    """
    从备份恢复配置
    """
    try:
        success = server.config_manager.restore_configs(backup_path)
        
        if success:
            # 重新初始化ConversationHandler以使用恢复的配置
            try:
                await server.initialize()
                print("✅ 配置恢复后ConversationHandler重新初始化成功")
            except Exception as e:
                print(f"⚠️ 重新初始化ConversationHandler失败: {e}")
            
            return ConfigResponse(
                success=True,
                message="配置恢复成功，系统已重新初始化",
                config=None
            )
        else:
            return ConfigResponse(
                success=False,
                message="配置恢复失败",
                config=None
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"恢复配置失败: {str(e)}"
        )


@app.post("/api/system/reinitialize", response_model=dict)
async def reinitialize_system():
    """
    手动重新初始化系统（在配置更新后使用）
    """
    try:
        await server.initialize()
        return {
            "success": True,
            "message": "系统重新初始化成功",
            "conversation_handler_status": "initialized" if server.conversation_handler else "not_initialized"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"系统重新初始化失败: {str(e)}"
        }


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 启动情感陪伴AI Web API服务器...")
    print("📖 API文档: http://localhost:8000/docs")
    print("🌐 前端界面: http://localhost:8000/static/index.html")
    print("💡 健康检查: http://localhost:8000/api/health")
    
    uvicorn.run(
        "web_api.web_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
