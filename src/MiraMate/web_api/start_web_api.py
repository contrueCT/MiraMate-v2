"""
Web API服务器启动脚本 - 支持Docker容器化
"""

# 尽早禁用遥测功能，避免PostHog等服务的SSL错误
import os
import sys
from pathlib import Path


import subprocess
import shutil

# Docker环境适配
def get_project_root():
    """获取项目根目录，支持容器环境"""
    if os.getenv('DOCKER_ENV'):
        return Path('/app')
    # 从当前文件位置向上追溯到项目根目录
    # 当前文件: src/MiraMate/web_api/start_web_api.py
    # 项目根目录: 向上3级
    return Path(__file__).parent.parent.parent.parent

project_root = get_project_root()
sys.path.insert(0, str(project_root))

def check_dependencies():
    """检查并安装依赖"""
    # Docker环境中跳过依赖检查
    if os.getenv('DOCKER_ENV'):
        print("📦 Docker环境，跳过依赖检查...")
        return True
        
    requirements_file = project_root / "src" / "MiraMate" / "web_api" / "requirements-web.txt"
    
    print("📦 检查Web API依赖...")
    
    # 检查是否在虚拟环境中（通过VIRTUAL_ENV环境变量）
    venv_path = os.getenv('VIRTUAL_ENV')
    is_in_venv = venv_path is not None
    
    if is_in_venv:
        print(f"✅ 检测到虚拟环境: {venv_path}")
    else:
        print("⚠️  未检测到激活的虚拟环境")
    
    try:
        # 优先使用 uv，如果不可用则回退到 pip
        if shutil.which('uv'):
            print("✅ 使用 uv 安装依赖...")
            if requirements_file.exists():
                subprocess.run(['uv', 'pip', 'install', '-r', str(requirements_file)], check=True)
            else:
                print(f"⚠️  requirements文件不存在: {requirements_file}")
                print("💡 跳过依赖安装，假设依赖已经安装")
        elif shutil.which('pip'):
            print("✅ 使用 pip 安装依赖...")
            if requirements_file.exists():
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], check=True)
            else:
                print(f"⚠️  requirements文件不存在: {requirements_file}")
                print("💡 跳过依赖安装，假设依赖已经安装")
        else:
            print("⚠️  未找到 uv 或 pip，跳过依赖检查")
            print("💡 请确保所需的依赖已经手动安装")
            
        print("✅ 依赖检查完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        print("💡 请手动检查并安装依赖")
        return False
    except Exception as e:
        print(f"❌ 检查依赖时发生错误: {e}")
        return False
    
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
        host = os.getenv('HOST', '0.0.0.0')  # Docker中使用0.0.0.0
        port = int(os.getenv('PORT', '8000'))
        
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
    
    # Docker环境检查
    if os.getenv('DOCKER_ENV'):
        print("🐳 Docker环境检测到")
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查配置（但不阻止启动）
    check_config()
    
    # 启动服务器
    start_server()

if __name__ == "__main__":
    main()
