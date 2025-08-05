# 流式聊天API使用指南

## 概述

后端现在支持流式聊天输出，可以实时接收AI回复的每个字符块，提供更流畅的用户体验。

## API 端点

### HTTP 流式接口

**端点**: `POST /api/chat`

**请求体**:
```json
{
    "message": "用户消息内容",
    "stream": true,          // 启用流式输出
    "enable_timing": true    // 启用性能统计（可选）
}
```

**响应格式**: Server-Sent Events (SSE)

流式响应包含多个数据块，每个块的格式：
```
data: {"type": "content", "content": "文字块", "chunk_id": 1, "timestamp": "2025-01-01T12:00:00Z"}

data: {"type": "metadata", "emotional_state": {...}, "commands": [...], "processing_time": 1.23, "timestamp": "2025-01-01T12:00:00Z"}

data: {"type": "end", "timestamp": "2025-01-01T12:00:00Z"}
```

**数据块类型**:
- `content`: 包含AI回复的文字内容
- `metadata`: 包含情感状态、视觉效果指令等元数据
- `end`: 表示流式传输结束
- `error`: 表示发生错误

### WebSocket 流式接口

**端点**: `ws://localhost:8000/ws`

**发送消息**:
```json
{
    "type": "chat",
    "data": "用户消息内容"
}
```

**接收消息**:
```json
// 开始流式传输
{"type": "chat_stream_start", "timestamp": 1234567890}

// 内容块
{"type": "chat_stream_chunk", "data": {"content": "文字块", "chunk_id": 1}, "timestamp": 1234567890}

// 完整响应（兼容性）
{"type": "chat_response", "data": {"response": "完整回复", "emotional_state": {...}, "commands": [...]}, "timestamp": 1234567890}

// 流结束
{"type": "chat_stream_end", "data": {"total_response": "完整回复", "processing_complete": true}, "timestamp": 1234567890}
```

## 兼容性

### 非流式模式
设置 `stream: false` 或不设置 `stream` 参数，API 将返回传统的完整响应：

```json
{
    "response": "AI的完整回复",
    "timestamp": "2025-01-01T12:00:00Z",
    "emotional_state": {...},
    "processing_time": 1.23,
    "commands": [...]
}
```

### 客户端适配
- **新客户端**: 可以监听流式数据块，实现打字机效果
- **旧客户端**: 仍会收到完整的 `chat_response` 消息，保持兼容性

## 性能优势

1. **更快的首字响应**: 用户可以立即看到AI开始回复
2. **更好的用户体验**: 模拟真实对话的逐字显示效果
3. **降低感知延迟**: 即使总处理时间相同，用户感觉更快

## 测试工具

使用提供的测试脚本：
```bash
python test_stream_api.py
```

该脚本会测试：
1. 流式HTTP API
2. 非流式HTTP API
3. 服务健康状态

## 错误处理

流式传输中的错误会以特殊的错误块形式发送：
```json
{
    "type": "error",
    "error": "processing_error",
    "message": "详细错误信息",
    "timestamp": "2025-01-01T12:00:00Z"
}
```

## 实现细节

### 后端架构
1. `ConversationHandlerAdapter.get_response_stream()` - 提供流式输出
2. `final_chain.astream()` - LangChain 的流式处理
3. FastAPI StreamingResponse - HTTP 流式响应
4. WebSocket 实时推送 - WebSocket 流式消息

### 数据流程
```
用户输入 → LangChain astream() → 适配器处理 → FastAPI/WebSocket → 客户端
```

每个数据块都包含时间戳和类型信息，便于客户端正确处理和显示。
