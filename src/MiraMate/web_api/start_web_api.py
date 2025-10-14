"""
Web API服务器启动脚本 - 支持Docker容器化
"""
import os
import sys
from pathlib import Path


import shutil
import importlib

# Docker环境适配
def get_project_root():
    """基于项目结构自动推断项目根目录（包含 pyproject.toml 且有 src/MiraMate）。"""
    p = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = p
        if (candidate / 'pyproject.toml').exists() and (candidate / 'src' / 'MiraMate').exists():
            return candidate
        if candidate.parent == candidate:
            break
        p = candidate.parent
    return Path(__file__).parent.parent.parent.parent

project_root = get_project_root()
# 确保优先从源码导入（避免命中 site-packages 的已安装版本）
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 现在再导入 settings，确保命中源码版本
from MiraMate.modules.settings import get_server

def check_dependencies():
    """轻量依赖检查：通过导入探测所需模块，提供安装指引，不在运行时安装。"""
    # Docker 环境中跳过依赖检查（镜像构建阶段已安装）
    if os.getenv('DOCKER_ENV'):
        print("📦 Docker环境，跳过依赖检查…")
        return True

    print("📦 检查 Web API 运行依赖…")

    core_modules = [
        ("fastapi", None),
        ("uvicorn", None),
    ]
    optional_modules = [
        ("langchain_openai", None),
        ("langchain_google_genai", None),
        ("chromadb", None),
        ("sentence_transformers", None),
    ]

    missing_core = []
    missing_optional = []

    def _try_import(mod: str) -> bool:
        try:
            importlib.import_module(mod)
            return True
        except Exception:
            return False

    for mod, _ in core_modules:
        if not _try_import(mod):
            missing_core.append(mod)
    for mod, _ in optional_modules:
        if not _try_import(mod):
            missing_optional.append(mod)

    # 安装建议
    has_uv = shutil.which('uv') is not None
    pip_cmd = f"{sys.executable} -m pip install -e ."
    uv_cmd = "uv pip install -e ."

    # 提示虚拟环境
    venv_path = os.getenv('VIRTUAL_ENV')
    if venv_path:
        print(f"✅ 检测到虚拟环境: {venv_path}")
    else:
        print("⚠️ 未检测到已激活的虚拟环境（建议先创建并激活虚拟环境再安装依赖）")

    if missing_core:
        print(f"❌ 缺少核心依赖: {', '.join(missing_core)}")
        print("➡️  请先安装项目依赖（基于 pyproject.toml）：")
        if has_uv:
            print(f"   - {uv_cmd}")
        else:
            print(f"   - {pip_cmd}")
        print("💡 Windows PowerShell 请先激活虚拟环境后再执行以上命令。")
        return False

    if missing_optional:
        print(f"⚠️ 缺少可选依赖（部分功能可能受限）: {', '.join(missing_optional)}")
        print("➡️  安装项目依赖可一并解决：")
        if has_uv:
            print(f"   - {uv_cmd}")
        else:
            print(f"   - {pip_cmd}")

    print("✅ 依赖检查完成")
    return True

def check_config():
    """检查配置文件状态"""
    print("🔍 检查配置文件...")
    
    # 导入配置管理器
    from MiraMate.web_api.config_manager import ConfigManager
    
    try:
        config_manager = ConfigManager(str(project_root))
        
        # 检查重构后的LLM配置文件
        llm_config_file = project_root / "configs" / "llm_config.json"
        
        if not llm_config_file.exists():
            print("⚠️  未找到LLM配置文件，但服务仍将启动以便通过客户端配置")
            return True
            
        import json
        with open(llm_config_file, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        
        if not configs:
            print("⚠️  LLM配置文件为空")
            return True
            
        # 检查是否有空的API密钥
        empty_keys = 0
        for i, config in enumerate(configs, 1):
            api_key = config.get("api_key", "").strip()
            if not api_key or api_key == "":
                empty_keys += 1
                
        if empty_keys > 0:
            print(f"⚠️  发现 {empty_keys} 个API配置缺少密钥")
            print("📝 可以通过以下方式配置API密钥:")
            print(f"   1. 直接编辑配置文件: {llm_config_file}")
            print(f"   2. 通过Electron客户端配置界面")
            print("💡 服务将正常启动，配置完成后功能即可使用")
            return True
        else:
            print(f"✅ 找到 {len(configs)} 个有效的API配置")
            return True
            
    except Exception as e:
        print(f"⚠️  检查配置时发生错误: {e}")
        print("💡 服务仍将启动以便通过客户端配置")
        return True

def start_server():
    """启动Web API服务器"""
    try:
        import uvicorn
        from MiraMate.web_api.web_api import app
        
        # 从环境变量获取配置
        srv = get_server()
        host = srv.get('HOST', '0.0.0.0')
        port = int(srv.get('PORT', 8000))
        
        print(f"\n🚀 启动情感陪伴AI Web API服务器...")
        print("=" * 50)
        print(f"📡 API服务地址: http://{host}:{port}")
        print(f"📖 API文档: http://{host}:{port}/docs")
        print(f"💡 健康检查: http://{host}:{port}/api/health")
        print(f"�️  此服务为Electron客户端提供API支持")
        print("=" * 50)
        print("按 Ctrl+C 停止服务器\n")
        
        # 直接传递app对象
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动服务器失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🎯 情感陪伴AI Web API 启动器")
    print("=" * 40)
    
    # 环境信息提示（可选）
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查配置（但不阻止启动）
    check_config()
    
    # 启动服务器
    start_server()

if __name__ == "__main__":
    main()
