"""
集中配置读取模块
- 统一从 configs/user_config.json 读取可变配置（persona/server/storage）
- 支持 mtime 检测以便热读，无需重启即可生效
- 对 legacy 字段 environment 兼容（新写入使用 persona）
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def _detect_project_root() -> Path:
    """根据项目结构探测根目录（包含 pyproject.toml 且存在 src/MiraMate）。"""
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src" / "MiraMate").exists():
            return parent
    # 兜底：从当前文件向上四级
    return p.parent.parent.parent.parent


PROJECT_ROOT = _detect_project_root()
CONFIG_DIR = PROJECT_ROOT / "configs"
USER_CONFIG_FILE = CONFIG_DIR / "user_config.json"

_cache_mtime: float = -1.0
_cache: Dict[str, Any] = {}


def _ensure_user_config_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    """确保 user_config.json 包含所需字段；如缺失则补齐默认值并回写到磁盘。

    规则：
    - persona: {user_name, agent_name, agent_description}
    - environment: 与 persona 同步以兼容旧逻辑（若一方缺失，用另一方填充）
    - server: {host, port}
    - storage: {memory_dir}
    """
    os.makedirs(CONFIG_DIR, exist_ok=True)

    updated = False
    data = dict(data) if isinstance(data, dict) else {}

    # 默认块
    default_persona = {
        "user_name": "小伙伴",
        "agent_name": "小梦",
        "agent_description": "你是一个可爱的AI助手",
    }
    default_server = {"host": "0.0.0.0", "port": 8000}
    default_storage = {"memory_dir": str(PROJECT_ROOT / "memory")}

    persona = data.get("persona")
    environment = data.get("environment")

    # 1) persona / environment 双向兼容与补齐
    if not isinstance(persona, dict) and isinstance(environment, dict):
        data["persona"] = environment.copy()
        persona = data["persona"]
        updated = True
    if not isinstance(environment, dict) and isinstance(persona, dict):
        data["environment"] = persona.copy()
        environment = data["environment"]
        updated = True
    if not isinstance(persona, dict):
        data["persona"] = default_persona.copy()
        persona = data["persona"]
        updated = True
    # persona 子键补齐
    for k, v in default_persona.items():
        if persona.get(k) is None:
            persona[k] = v
            updated = True
    # environment 跟随 persona（保持同步）
    if not isinstance(environment, dict):
        data["environment"] = persona.copy()
        environment = data["environment"]
        updated = True
    else:
        # 合并 persona 到 environment，避免缺键
        for k, v in persona.items():
            if environment.get(k) is None:
                environment[k] = v
                updated = True

    # 2) server 补齐
    server = data.get("server")
    if not isinstance(server, dict):
        data["server"] = default_server.copy()
        server = data["server"]
        updated = True
    else:
        if server.get("host") is None:
            server["host"] = default_server["host"]
            updated = True
        if server.get("port") is None:
            server["port"] = default_server["port"]
            updated = True

    # 3) storage 补齐
    storage = data.get("storage")
    if not isinstance(storage, dict):
        data["storage"] = default_storage.copy()
        updated = True
    else:
        if storage.get("memory_dir") is None:
            storage["memory_dir"] = default_storage["memory_dir"]
            updated = True

    if updated:
        # 写回磁盘，便于用户直接编辑
        try:
            data["updated_at"] = data.get("updated_at") or data.get("created_at") or ""
            with USER_CONFIG_FILE.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception:
            # 回写失败时忽略，保持内存中的数据可用
            pass
    return data


def _load_user_config() -> Dict[str, Any]:
    """读取 user_config.json，基于 mtime 缓存。文件不存在时返回空字典。"""
    global _cache_mtime, _cache
    try:
        raw: Dict[str, Any] = {}
        if USER_CONFIG_FILE.exists():
            stat = USER_CONFIG_FILE.stat()
            # 始终读取一次（以便进行自愈与回写），再根据 mtime 更新缓存
            with USER_CONFIG_FILE.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            data = _ensure_user_config_defaults(raw)
            # 如果文件发生了回写，mtime 会变化；重新获取 mtime
            new_stat = USER_CONFIG_FILE.stat()
            _cache_mtime = new_stat.st_mtime
            _cache = data
        else:
            # 文件不存在：以默认结构创建并落盘
            data = _ensure_user_config_defaults({})
            stat = USER_CONFIG_FILE.stat()
            _cache_mtime = stat.st_mtime
            _cache = data
    except Exception:
        # 解析失败：生成默认并尝试写回
        data = _ensure_user_config_defaults({})
        _cache = data
        try:
            stat = USER_CONFIG_FILE.stat()
            _cache_mtime = stat.st_mtime
        except Exception:
            _cache_mtime = -1.0
    return _cache or {}


def get_persona() -> Dict[str, str]:
    """获取对话人格相关配置（用户名/智能体名/描述）。"""
    data = _load_user_config()
    # 新字段 persona，兼容旧字段 environment
    src = data.get("persona") or data.get("environment") or {}
    # 最终默认值
    return {
        "USER_NAME": src.get("user_name", "小伙伴"),
        "AGENT_NAME": src.get("agent_name", "小梦"),
        "AGENT_DESCRIPTION": src.get("agent_description", "你是一个可爱的AI助手"),
    }


def get_server() -> Dict[str, Any]:
    """获取服务端口配置。env 可作为兜底覆盖（兼容旧部署方式）。"""
    data = _load_user_config()
    s = data.get("server", {})
    host = s.get("host", "0.0.0.0")
    port = int(s.get("port", 8000))
    # 允许通过环境变量覆盖，保留兼容性（不推荐文档化）。
    host = os.getenv("HOST", host)
    try:
        port = int(os.getenv("PORT", str(port)))
    except Exception:
        pass
    return {"HOST": host, "PORT": port}


def get_storage() -> Dict[str, Any]:
    """获取存储相关配置（目前仅 memory_dir，可选）。"""
    data = _load_user_config()
    storage = data.get("storage", {})
    # 默认使用标准相对路径，容器内 WORKDIR=/app 时命中卷挂载
    mem_dir = storage.get("memory_dir") or str(PROJECT_ROOT / "memory")
    return {"MEMORY_DIR": mem_dir}
