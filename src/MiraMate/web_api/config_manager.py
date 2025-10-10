"""
配置管理器模块 - 支持Docker容器化
负责处理系统配置的读取、保存和更新
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv, set_key

from MiraMate.web_api.models import LLMConfig, EnvironmentConfig, UserPreferences, SystemConfig

class ConfigManager:
    """配置管理器 - 支持容器环境"""
    
    def __init__(self, project_root: Optional[str] = None):
        """初始化配置管理器"""
        if project_root is None:
            # 基于项目结构自动推断（包含 pyproject.toml 且有 src/MiraMate）
            p = Path(__file__).resolve().parent
            detected = None
            for _ in range(6):
                candidate = p
                if (candidate / 'pyproject.toml').exists() and (candidate / 'src' / 'MiraMate').exists():
                    detected = candidate
                    break
                if candidate.parent == candidate:
                    break
                p = candidate.parent
            # 兜底：回退到四级上层（与其他模块一致）
            self.project_root = detected or Path(__file__).parent.parent.parent.parent
        else:
            self.project_root = Path(project_root)

        # 配置目录路径（固定到项目根下的 configs，避免依赖环境变量）
        self.config_dir = Path(str(self.project_root / "configs"))
        self.env_file = self.project_root / ".env"

        # 重构后的配置文件名
        self.llm_config_file = self.config_dir / "llm_config.json"
        # 保持向后兼容
        self.legacy_llm_config_file = self.config_dir / "OAI_CONFIG_LIST.json"

        self.user_config_file = self.config_dir / "user_config.json"

        # 确保配置目录存在
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # 自动创建配置文件
        self._ensure_config_files_exist()

        # 加载环境变量（仅当 .env 存在时）
        if self.env_file.exists():
            load_dotenv(self.env_file)

    def _ensure_config_files_exist(self):
        """确保配置文件存在，如果不存在则创建默认配置"""
        # 优先检查新的配置文件，如果不存在但旧的存在，则迁移
        if not self.llm_config_file.exists():
            if self.legacy_llm_config_file.exists():
                # 迁移旧配置到新格式
                self._migrate_legacy_config()
            else:
                # 创建新的默认配置
                self._create_default_llm_config()
            
        # 环境变量文件仅用于向后兼容，不再强制创建
            
        # 创建用户配置文件
        if not self.user_config_file.exists():
            self._create_default_user_config()

    def _create_default_llm_config(self):
        """创建默认的LLM配置文件"""
        default_config = [
            {
                "model": "Qwen/Qwen2.5-72B-Instruct",
                "api_key": "",
                "base_url": "https://api.siliconflow.cn/v1",
                "api_type": "openai",
                "model_kwargs": {"temperature": 0.8}
            },
            {
                "model": "Qwen/Qwen2.5-7B-Instruct", 
                "api_key": "",
                "base_url": "https://api.siliconflow.cn/v1",
                "api_type": "openai",
                "model_kwargs": {"temperature": 0.0, "response_format": {"type": "json_object"}}
            }
        ]
        
        try:
            with open(self.llm_config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            print(f"✅ 已创建默认LLM配置文件: {self.llm_config_file}")
            print(f"📝 请编辑配置文件并填入你的API密钥")
        except Exception as e:
            print(f"❌ 创建LLM配置文件失败: {e}")

    def _migrate_legacy_config(self):
        """迁移旧的配置文件格式到新格式"""
        try:
            print(f"🔄 发现旧配置文件，正在迁移到新格式...")
            
            # 复制旧配置文件内容到新位置
            shutil.copy2(self.legacy_llm_config_file, self.llm_config_file)
            
            print(f"✅ 配置文件迁移完成: {self.llm_config_file}")
            print(f"💡 旧配置文件已保留: {self.legacy_llm_config_file}")
            
        except Exception as e:
            print(f"❌ 配置文件迁移失败: {e}")
            # 迁移失败时创建默认配置
            self._create_default_llm_config()

    def _create_default_env_config(self):
        """弃用：保留函数名以兼容旧逻辑，但不再创建 .env。"""
        pass

    def _create_default_user_config(self):
        """创建默认的用户配置文件"""
        default_user_config = {
            "persona": {
                "user_name": "小伙伴",
                "agent_name": "小梦",
                "agent_description": "你是一个可爱的AI助手"
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8000
            },
            "preferences": {
                "theme": "light",
                "language": "zh-CN",
                "chat_history_limit": 1000,
                "auto_save": True
            },
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
        try:
            with open(self.user_config_file, 'w', encoding='utf-8') as f:
                json.dump(default_user_config, f, indent=4, ensure_ascii=False)
            print(f"✅ 已创建默认用户配置文件: {self.user_config_file}")
        except Exception as e:
            print(f"❌ 创建用户配置文件失败: {e}")

    def get_llm_configs(self) -> List[LLMConfig]:
        """获取LLM配置列表"""
        try:
            with open(self.llm_config_file, 'r', encoding='utf-8') as f:
                configs_data = json.load(f)
                
            configs = []
            for config_data in configs_data:
                config = LLMConfig(**config_data)
                configs.append(config)
            return configs
        except Exception as e:
            print(f"❌ 读取LLM配置失败: {e}")
            return []

    def save_llm_configs(self, configs: List[LLMConfig]) -> bool:
        """保存LLM配置列表"""
        try:
            # 转换配置为字典
            configs_data = [config.dict() for config in configs]
            
            with open(self.llm_config_file, 'w', encoding='utf-8') as f:
                json.dump(configs_data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ 保存LLM配置失败: {e}")
            return False

    def get_environment_config(self) -> EnvironmentConfig:
        """获取环境配置（从 user_config.json 读取，兼容 legacy environment 字段）"""
        try:
            if self.user_config_file.exists():
                with open(self.user_config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                env = data.get('persona') or data.get('environment') or {}
                return EnvironmentConfig(
                    user_name=env.get('user_name', '小伙伴'),
                    agent_name=env.get('agent_name', '小梦'),
                    agent_description=env.get('agent_description', '你是一个可爱的AI助手')
                )
        except Exception as e:
            print(f"❌ 读取环境配置失败: {e}")
        return EnvironmentConfig(user_name="小伙伴", agent_name="小梦", agent_description="你是一个可爱的AI助手")

    def save_environment_config(self, config: EnvironmentConfig) -> bool:
        """保存环境配置（写入 user_config.json 的 persona/environment 字段）"""
        try:
            existing_data = {}
            if self.user_config_file.exists():
                with open(self.user_config_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

            persona = {
                "user_name": config.user_name,
                "agent_name": config.agent_name,
                "agent_description": config.agent_description,
            }
            existing_data["persona"] = persona
            # 同步 legacy 字段，便于历史代码读取
            existing_data["environment"] = persona
            existing_data["updated_at"] = datetime.now().isoformat()

            with open(self.user_config_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ 保存环境配置失败: {e}")
            return False

    def get_user_preferences(self) -> UserPreferences:
        """获取用户偏好设置"""
        try:
            with open(self.user_config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return UserPreferences(**data.get('preferences', {}))
        except Exception as e:
            print(f"❌ 读取用户配置失败: {e}")
            return UserPreferences()

    def save_user_preferences(self, preferences: UserPreferences) -> bool:
        """保存用户偏好设置"""
        try:
            # 读取现有配置
            existing_data = {}
            if self.user_config_file.exists():
                with open(self.user_config_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            
            # 更新偏好设置
            existing_data['preferences'] = preferences.dict()
            existing_data['updated_at'] = datetime.now().isoformat()
            
            with open(self.user_config_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ 保存用户配置失败: {e}")
            return False

    def get_system_config(self) -> SystemConfig:
        """获取系统配置"""
        return SystemConfig(
            llm_configs=self.get_llm_configs(),
            environment=self.get_environment_config(),
            user_preferences=self.get_user_preferences()
        )

    def update_config(self, config_type: str, config_data: Dict[str, Any]) -> bool:
        """更新配置"""
        try:
            if config_type == "llm":
                configs = [LLMConfig(**item) for item in config_data.get('configs', [])]
                return self.save_llm_configs(configs)
            elif config_type == "environment":
                config = EnvironmentConfig(**config_data)
                return self.save_environment_config(config)
            elif config_type == "user":
                preferences = UserPreferences(**config_data)
                return self.save_user_preferences(preferences)
            else:
                return False
        except Exception as e:
            print(f"❌ 更新配置失败: {e}")
            return False

    def backup_configs(self) -> str:
        """备份配置文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.config_dir / f"backup_{timestamp}"
            backup_dir.mkdir(exist_ok=True)
            
            # 备份所有配置文件
            if self.llm_config_file.exists():
                shutil.copy2(self.llm_config_file, backup_dir / "llm_config.json")
            if self.env_file.exists():
                shutil.copy2(self.env_file, backup_dir / ".env")
            if self.user_config_file.exists():
                shutil.copy2(self.user_config_file, backup_dir / "user_config.json")
                
            return str(backup_dir)
        except Exception as e:
            print(f"❌ 备份配置失败: {e}")
            return ""

    def restore_configs(self, backup_path: str) -> bool:
        """恢复配置文件"""
        try:
            backup_dir = Path(backup_path)
            if not backup_dir.exists():
                return False
                
            # 恢复配置文件
            # 优先恢复新格式的LLM配置文件
            llm_restored = False
            for llm_file_name in ["llm_config.json", "OAI_CONFIG_LIST.json"]:
                source_file = backup_dir / llm_file_name
                if source_file.exists() and not llm_restored:
                    shutil.copy2(source_file, self.llm_config_file)
                    llm_restored = True
                    break
            
            # 恢复其他配置文件
            for file_name, target_file in [
                (".env", self.env_file),
                ("user_config.json", self.user_config_file)
            ]:
                source_file = backup_dir / file_name
                if source_file.exists():
                    shutil.copy2(source_file, target_file)
                    
            return True
        except Exception as e:
            print(f"❌ 恢复配置失败: {e}")
            return False

    def validate_llm_config(self, config: LLMConfig) -> tuple[bool, str]:
        """验证LLM配置"""
        if not config.api_key or config.api_key.strip() == "":
            return False, "API密钥不能为空"
        if not config.model or config.model.strip() == "":
            return False, "模型名称不能为空"
        
        # 对于gemini类型的API，不需要base_url
        if config.api_type != "gemini":
            if not config.base_url or config.base_url.strip() == "":
                return False, "API地址不能为空"
        
        # 验证 model_kwargs 如果存在的话应该是字典类型
        if config.model_kwargs is not None and not isinstance(config.model_kwargs, dict):
            return False, "model_kwargs 必须是字典类型"
        
        return True, "配置有效"

    def test_llm_connection(self, config: LLMConfig) -> tuple[bool, str]:
        """测试LLM连接"""
        try:
            import requests
            
            # 简单的连接测试（可以根据具体API调整）
            test_url = f"{config.base_url.rstrip('/')}/models"
            headers = {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(test_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return True, "连接测试成功"
            else:
                return False, f"连接测试失败: HTTP {response.status_code}"
                
        except Exception as e:
            return False, f"连接测试失败: {str(e)}"

    def get_langchain_llm_configs(self) -> List[Dict[str, Any]]:
        """获取适用于重构后LangChain架构的配置"""
        try:
            if self.llm_config_file.exists():
                with open(self.llm_config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"❌ 读取LangChain配置失败: {e}")
            return []
