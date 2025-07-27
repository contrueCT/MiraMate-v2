# 已经接入了对话链，主要负责用户画像、对话记忆、事实记忆等的存储和检索，后续需要在对话完后的步骤中更新记忆内容：对话记忆、事实记忆、用户偏好、重大事件等。

import json
import os
# 在导入任何模型前设置环境变量
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import chromadb
from datetime import datetime
from typing import List, Dict, Optional
from uuid import uuid4
from chromadb.utils import embedding_functions

# === 🧠 一、通用结构定义 ===

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 可读时间戳

def get_iso_timestamp():
    return datetime.now().isoformat()  # ISO格式时间戳，用于ChromaDB

def format_natural_time(dt: datetime) -> str:
    """将时间格式化为自然语言形式"""
    weekdays = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
    weekday = weekdays[dt.weekday()]
    
    # 判断时间段
    hour = dt.hour
    if 5 <= hour < 12:
        time_period = "上午"
    elif 12 <= hour < 14:
        time_period = "中午"
    elif 14 <= hour < 18:
        time_period = "下午"
    elif 18 <= hour < 22:
        time_period = "晚上"
    else:
        time_period = "深夜"
    
    return f"{dt.year}年{dt.month}月{dt.day}日{weekday}{time_period}"

# === 📦 二、存储目录设置 ===
# 使用基于 __file__ 的健壮路径构建方法
# 1. 获取当前文件(memory_system.py)所在的目录: .../src/MiraMate/modules/
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. 从模块目录回溯三层，到达项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, '..', '..', '..'))
BASE_DIR = os.path.join(PROJECT_ROOT, "memory", "memory_storage")
PROFILE_PATH = os.path.join(BASE_DIR, "user_profile.json")
ACTIVE_TAGS_PATH = os.path.join(BASE_DIR, "active_tags.json")
TEMP_FOCUS_EVENTS_PATH = os.path.join(BASE_DIR, "temp_focus_events.json")

# 缓存文件路径
PREFERENCE_CACHE_PATH = os.path.join(BASE_DIR, "preference_cache.json")
FACT_CACHE_PATH = os.path.join(BASE_DIR, "fact_cache.json")
PROFILE_CACHE_PATH = os.path.join(BASE_DIR, "profile_cache.json")

# ChromaDB 存储目录
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# 创建必要的目录
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# === 🔧 三、ChromaDB 初始化 ===

class MemorySystem:
    def __init__(self, persist_directory=None):
        """初始化记忆系统"""
        if persist_directory is None:
            persist_directory = CHROMA_DB_DIR
            
        # 处理相对路径，支持容器环境
        if not os.path.isabs(persist_directory):
            if os.getenv('DOCKER_ENV'):
                # Docker环境中使用绝对路径
                persist_directory = f"/app/{persist_directory}"
            else:
                # 本地环境中转换为绝对路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_dir)  # 假设此文件在项目根目录
                persist_directory = os.path.join(project_root, persist_directory)
        
        # 创建持久化目录
        os.makedirs(persist_directory, exist_ok=True)
        
        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(path=persist_directory)
        print(f"✅ ChromaDB客户端已初始化，持久化目录: {persist_directory}")

        # 配置嵌入函数
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-base-zh-v1.5",
            device="cpu"
        )

        # 定义HNSW索引参数
        self.hnsw_metadata_config = {
            "hnsw:space": "cosine",          # 使用余弦相似度
            "hnsw:M": 32,                    # 推荐M值
            "hnsw:construction_ef": 256,     # construction_ef值
            "hnsw:num_threads": 4            # 构建线程数
        }

        # 初始化集合
        self.collections = {
            "dialog_logs": self.client.get_or_create_collection(
                name="dialog_logs",
                embedding_function=self.embedding_function,
                metadata=self.hnsw_metadata_config
            ),
            "facts": self.client.get_or_create_collection(
                name="facts",
                embedding_function=self.embedding_function,
                metadata=self.hnsw_metadata_config
            ),
            "user_preferences": self.client.get_or_create_collection(
                name="user_preferences",
                embedding_function=self.embedding_function,
                metadata=self.hnsw_metadata_config
            ),
            "important_events": self.client.get_or_create_collection(
                name="important_events",
                embedding_function=self.embedding_function,
                metadata=self.hnsw_metadata_config
            )
        }

    # === 🧍‍♂️ 用户画像 ===
    # 更新策略，每次对话后都异步保存用户画像，避免阻塞主线程，且在智能体空闲时调用模型处理合并重复字段
    def save_user_profile(self, profile: Dict):
        """保存用户画像"""
        profile["last_updated"] = get_timestamp()
        with open(PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        print(f"✅ 用户画像已保存")

    def load_user_profile(self) -> Optional[Dict]:
        """加载用户画像"""
        if not os.path.exists(PROFILE_PATH):
            return None
        with open(PROFILE_PATH, encoding="utf-8") as f:
            return json.load(f)

    def update_user_profile(self, **updates):
        """更新用户画像（部分更新）"""
        profile = self.load_user_profile() or {}
        profile.update(updates)
        self.save_user_profile(profile)

    # === 💬 对话记录记忆 ===
    # 保存所有的对话记忆
    def save_dialog_log(self, user_input: str, ai_response: str, topic: str, 
                       sentiment: str, importance: float, tags: List[str], 
                       additional_metadata: Optional[Dict] = None):
        """保存对话记录到ChromaDB"""
        
        # 生成唯一ID
        dialog_id = f"dialog_{uuid4().hex}"
        timestamp = get_iso_timestamp()
        current_time = datetime.now()
        natural_time = format_natural_time(current_time)
        
        # 构建包含完整信息的文档内容
        tags_str = "、".join(tags) if tags else "无标签"
        
        dialog_content = f"""时间：{natural_time}
话题：{topic}
情感：{sentiment}
标签：{tags_str}
用户：{user_input}
AI：{ai_response}"""
        
        # 准备元数据（只保留必要的结构化数据）
        metadata = {
            "type": "dialog_log",
            "timestamp": timestamp,
            "topic": topic,
            "sentiment": sentiment,
            "importance": importance,
            "tags": json.dumps(tags, ensure_ascii=False),
            "user_input_length": len(user_input),
            "ai_response_length": len(ai_response)
        }
        
        # 添加额外元数据
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # 保存到ChromaDB
        try:
            self.collections["dialog_logs"].add(
                ids=[dialog_id],
                metadatas=[metadata],
                documents=[dialog_content]
            )
            print(f"✅ 对话记录已保存: {topic} (重要性: {importance})")
            
            # 更新活跃标签
            self.update_active_tags(tags)
            
            return dialog_id
        except Exception as e:
            print(f"❌ 保存对话记录失败: {e}")
            return None

    def search_dialog_logs(self, query: str, n_results: int = 5, 
                          where_filter: Optional[Dict] = None, 
                          similarity_threshold: float = 0.6) -> List[Dict]:
        """搜索对话记录"""
        try:
            search_params = {
                "query_texts": [query],
                "n_results": n_results
            }
            
            if where_filter:
                search_params["where"] = where_filter
            
            results = self.collections["dialog_logs"].query(**search_params)
            
            dialog_memories = []
            if (results and results["documents"] and results["documents"][0] and
                results["metadatas"] and results["metadatas"][0] and
                results["distances"] and results["distances"][0]):
                
                docs_list = results["documents"][0]
                metadatas_list = results["metadatas"][0]
                distances_list = results["distances"][0]
                ids_list = results["ids"][0]
                
                for i, doc_content in enumerate(docs_list):
                    if i < len(distances_list):
                        distance = distances_list[i]
                        if distance <= (1 - similarity_threshold):  # 转换为相似度阈值
                            metadata = metadatas_list[i]
                            # 解析标签
                            tags = json.loads(metadata.get("tags", "[]"))
                            
                            dialog_memory = {
                                "id": ids_list[i],
                                "content": doc_content,
                                "metadata": metadata,
                                "tags": tags,
                                "similarity": 1 - distance,
                                "topic": metadata.get("topic", ""),
                                "sentiment": metadata.get("sentiment", ""),
                                "importance": metadata.get("importance", 0.0),
                                "timestamp": metadata.get("timestamp", "")
                            }
                            dialog_memories.append(dialog_memory)
            
            return dialog_memories
        except Exception as e:
            print(f"❌ 搜索对话记录失败: {e}")
            return []

    def get_recent_dialogs(self, limit: int = 5) -> List[Dict]:
        """获取最近的对话记录"""
        try:
            # 获取所有对话记录
            all_dialogs = self.collections["dialog_logs"].get()
            
            if not all_dialogs or not all_dialogs["metadatas"]:
                return []
            
            # 按时间戳排序
            dialog_list = []
            for i, metadata in enumerate(all_dialogs["metadatas"]):
                dialog_info = {
                    "id": all_dialogs["ids"][i],
                    "content": all_dialogs["documents"][i],
                    "metadata": metadata,
                    "timestamp": metadata.get("timestamp", ""),
                    "tags": json.loads(metadata.get("tags", "[]"))
                }
                dialog_list.append(dialog_info)
            
            # 按时间戳倒序排序
            dialog_list.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return dialog_list[:limit]
        except Exception as e:
            print(f"❌ 获取最近对话失败: {e}")
            return []

    # === 🧠 概念知识（事实记忆）===
    # 先存到缓冲文件（持久化），然后在空闲时经过模型处理后保存到ChromaDB
    def save_fact_memory(self, content: str, tags: List[str], 
                        source: str = "dialog", confidence: float = 1.0,
                        additional_metadata: Optional[Dict] = None):
        """保存事实记忆到ChromaDB"""
        
        # 生成唯一ID
        fact_id = f"fact_{uuid4().hex}"
        timestamp = get_iso_timestamp()
        current_time = datetime.now()
        natural_time = format_natural_time(current_time)
        
        # 构建包含完整信息的文档内容
        tags_str = "、".join(tags) if tags else "无标签"
        
        fact_content = f"""时间：{natural_time}
来源：{source}
标签：{tags_str}
内容：{content}"""
        
        # 准备元数据
        metadata = {
            "type": "fact",
            "timestamp": timestamp,
            "source": source,
            "confidence": confidence,
            "tags": json.dumps(tags, ensure_ascii=False),
            "content_length": len(content)
        }
        
        # 添加额外元数据
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # 保存到ChromaDB
        try:
            self.collections["facts"].add(
                ids=[fact_id],
                metadatas=[metadata],
                documents=[fact_content]
            )
            print(f"✅ 事实记忆已保存: {content[:30]}... (置信度: {confidence})")
            
            # 更新活跃标签
            self.update_active_tags(tags)
            
            return fact_id
        except Exception as e:
            print(f"❌ 保存事实记忆失败: {e}")
            return None

    def search_fact_memory(self, query: str, n_results: int = 3,
                          where_filter: Optional[Dict] = None,
                          similarity_threshold: float = 0.7) -> List[Dict]:
        """搜索事实记忆"""
        try:
            search_params = {
                "query_texts": [query],
                "n_results": n_results
            }
            
            if where_filter:
                search_params["where"] = where_filter
            
            results = self.collections["facts"].query(**search_params)
            
            fact_memories = []
            if (results and results["documents"] and results["documents"][0] and
                results["metadatas"] and results["metadatas"][0] and
                results["distances"] and results["distances"][0]):
                
                docs_list = results["documents"][0]
                metadatas_list = results["metadatas"][0]
                distances_list = results["distances"][0]
                ids_list = results["ids"][0]
                
                for i, doc_content in enumerate(docs_list):
                    if i < len(distances_list):
                        distance = distances_list[i]
                        if distance <= (1 - similarity_threshold):
                            metadata = metadatas_list[i]
                            tags = json.loads(metadata.get("tags", "[]"))
                            
                            fact_memory = {
                                "id": ids_list[i],
                                "content": doc_content,
                                "metadata": metadata,
                                "tags": tags,
                                "similarity": 1 - distance,
                                "source": metadata.get("source", ""),
                                "confidence": metadata.get("confidence", 1.0),
                                "timestamp": metadata.get("timestamp", "")
                            }
                            fact_memories.append(fact_memory)
            
            return fact_memories
        except Exception as e:
            print(f"❌ 搜索事实记忆失败: {e}")
            return []

    def update_fact_confidence(self, fact_id: str, new_confidence: float):
        """更新事实记忆的置信度"""
        try:
            # 获取现有记忆
            result = self.collections["facts"].get(ids=[fact_id])
            if result and result["metadatas"]:
                metadata = result["metadatas"][0]
                metadata["confidence"] = new_confidence
                metadata["last_updated"] = get_timestamp()
                
                # 更新记忆
                self.collections["facts"].update(
                    ids=[fact_id],
                    metadatas=[metadata]
                )
                print(f"✅ 事实记忆置信度已更新: {new_confidence}")
                return True
        except Exception as e:
            print(f"❌ 更新事实记忆置信度失败: {e}")
        return False

    # === 📝 缓存管理方法 ===
    
    def cache_user_preference(self, content: str, preference_type: str, 
                             tags: List[str], confidence: float = 1.0,
                             additional_metadata: Optional[Dict] = None):
        """缓存用户偏好信息到本地JSON文件"""
        
        # 创建缓存条目
        cache_entry = {
            "id": f"preference_cache_{uuid4().hex}",
            "content": content,
            "preference_type": preference_type,
            "tags": tags,
            "confidence": confidence,
            "timestamp": get_iso_timestamp(),
            "natural_time": format_natural_time(datetime.now()),
            "content_length": len(content),
            "additional_metadata": additional_metadata or {}
        }
        
        # 读取现有缓存
        preferences_cache = self._load_cache_file(PREFERENCE_CACHE_PATH)
        
        # 添加新条目
        preferences_cache.append(cache_entry)
        
        # 保存缓存
        self._save_cache_file(PREFERENCE_CACHE_PATH, preferences_cache)
        
        print(f"✅ 用户偏好已缓存: {preference_type} - {content[:30]}...")
        return cache_entry["id"]
    
    def cache_fact_memory(self, content: str, tags: List[str], 
                         source: str = "dialog", confidence: float = 1.0,
                         additional_metadata: Optional[Dict] = None):
        """缓存事实记忆到本地JSON文件"""
        
        # 创建缓存条目
        cache_entry = {
            "id": f"fact_cache_{uuid4().hex}",
            "content": content,
            "tags": tags,
            "source": source,
            "confidence": confidence,
            "timestamp": get_iso_timestamp(),
            "natural_time": format_natural_time(datetime.now()),
            "content_length": len(content),
            "additional_metadata": additional_metadata or {}
        }
        
        # 读取现有缓存
        facts_cache = self._load_cache_file(FACT_CACHE_PATH)
        
        # 添加新条目
        facts_cache.append(cache_entry)
        
        # 保存缓存
        self._save_cache_file(FACT_CACHE_PATH, facts_cache)
        
        print(f"✅ 事实记忆已缓存: {content[:30]}... (置信度: {confidence})")
        return cache_entry["id"]
    
    def cache_profile_update(self, profile_data: Dict, source: str = "dialog"):
        """缓存用户画像更新信息到本地JSON文件"""
        
        # 创建缓存条目
        cache_entry = {
            "id": f"profile_cache_{uuid4().hex}",
            "profile_data": profile_data,
            "source": source,
            "timestamp": get_iso_timestamp(),
            "natural_time": format_natural_time(datetime.now())
        }
        
        # 读取现有缓存
        profile_cache = self._load_cache_file(PROFILE_CACHE_PATH)
        
        # 添加新条目
        profile_cache.append(cache_entry)
        
        # 保存缓存
        self._save_cache_file(PROFILE_CACHE_PATH, profile_cache)
        
        print(f"✅ 用户画像信息已缓存: {list(profile_data.keys())}")
        return cache_entry["id"]
    
    def _load_cache_file(self, file_path: str) -> List[Dict]:
        """加载缓存文件"""
        if not os.path.exists(file_path):
            return []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"⚠️ 缓存文件损坏或不存在，创建新文件: {file_path}")
            return []
    
    def _save_cache_file(self, file_path: str, cache_data: List[Dict]):
        """保存缓存文件"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ 保存缓存文件失败: {e}")
    
    def load_preference_cache(self) -> List[Dict]:
        """读取用户偏好缓存"""
        return self._load_cache_file(PREFERENCE_CACHE_PATH)
    
    def load_fact_cache(self) -> List[Dict]:
        """读取事实记忆缓存"""
        return self._load_cache_file(FACT_CACHE_PATH)
    
    def load_profile_cache(self) -> List[Dict]:
        """读取用户画像缓存"""
        return self._load_cache_file(PROFILE_CACHE_PATH)
    
    def clear_preference_cache(self):
        """清空用户偏好缓存"""
        self._save_cache_file(PREFERENCE_CACHE_PATH, [])
        print("🗑️ 用户偏好缓存已清空")
    
    def clear_fact_cache(self):
        """清空事实记忆缓存"""
        self._save_cache_file(FACT_CACHE_PATH, [])
        print("�️ 事实记忆缓存已清空")
    
    def clear_profile_cache(self):
        """清空用户画像缓存"""
        self._save_cache_file(PROFILE_CACHE_PATH, [])
        print("🗑️ 用户画像缓存已清空")
    
    def get_cache_status(self) -> Dict[str, int]:
        """获取各缓存文件的状态"""
        status = {
            "preferences_cache": len(self._load_cache_file(PREFERENCE_CACHE_PATH)),
            "facts_cache": len(self._load_cache_file(FACT_CACHE_PATH)),
            "profile_cache": len(self._load_cache_file(PROFILE_CACHE_PATH))
        }
        
        total = sum(status.values())
        print(f"📊 缓存状态: 偏好 {status['preferences_cache']} 条，事实 {status['facts_cache']} 条，画像 {status['profile_cache']} 条，总计 {total} 条")
        
        return status
    
    def clear_all_caches(self):
        """清空所有缓存"""
        self.clear_preference_cache()
        self.clear_fact_cache()
        self.clear_profile_cache()
        print("🗑️ 所有缓存已清空")

    # === 🎯 用户偏好信息 ===
    # 先存到缓冲文件（持久化），然后在空闲时经过模型处理后保存到ChromaDB
    def save_user_preference(self, content: str, preference_type: str, 
                            tags: List[str], additional_metadata: Optional[Dict] = None):
        """保存用户偏好信息到ChromaDB"""
        
        # 生成唯一ID
        preference_id = f"preference_{uuid4().hex}"
        timestamp = get_iso_timestamp()
        current_time = datetime.now()
        natural_time = format_natural_time(current_time)
        
        # 构建包含完整信息的文档内容
        tags_str = "、".join(tags) if tags else "无标签"
        
        preference_content = f"""时间：{natural_time}
类型：{preference_type}
标签：{tags_str}
内容：{content}"""
        
        # 准备元数据
        metadata = {
            "type": preference_type,
            "tags": json.dumps(tags, ensure_ascii=False),
            "timestamp": timestamp,
            "content_length": len(content)
        }
        
        # 添加额外元数据
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # 保存到ChromaDB
        try:
            self.collections["user_preferences"].add(
                ids=[preference_id],
                metadatas=[metadata],
                documents=[preference_content]
            )
            print(f"✅ 用户偏好已保存: {preference_type} - {content[:30]}...")
            
            # 更新活跃标签
            self.update_active_tags(tags)
            
            return preference_id
        except Exception as e:
            print(f"❌ 保存用户偏好失败: {e}")
            return None

    def search_user_preferences(self, query: str, n_results: int = 5,
                               where_filter: Optional[Dict] = None,
                               similarity_threshold: float = 0.7) -> List[Dict]:
        """搜索用户偏好信息"""
        try:
            search_params = {
                "query_texts": [query],
                "n_results": n_results
            }
            
            if where_filter:
                search_params["where"] = where_filter
            
            results = self.collections["user_preferences"].query(**search_params)
            
            preference_memories = []
            if (results and results["documents"] and results["documents"][0] and
                results["metadatas"] and results["metadatas"][0] and
                results["distances"] and results["distances"][0]):
                
                docs_list = results["documents"][0]
                metadatas_list = results["metadatas"][0]
                distances_list = results["distances"][0]
                ids_list = results["ids"][0]
                
                for i, doc_content in enumerate(docs_list):
                    if i < len(distances_list):
                        distance = distances_list[i]
                        if distance <= (1 - similarity_threshold):
                            metadata = metadatas_list[i]
                            tags = json.loads(metadata.get("tags", "[]"))
                            
                            preference_memory = {
                                "id": ids_list[i],
                                "content": doc_content,
                                "metadata": metadata,
                                "tags": tags,
                                "similarity": 1 - distance,
                                "type": metadata.get("type", ""),
                                "timestamp": metadata.get("timestamp", "")
                            }
                            preference_memories.append(preference_memory)
            
            return preference_memories
        except Exception as e:
            print(f"❌ 搜索用户偏好失败: {e}")
            return []

    def get_preferences_by_type(self, preference_type: str, limit: int = 10) -> List[Dict]:
        """根据类型获取用户偏好"""
        return self.search_user_preferences(
            query="", 
            n_results=limit,
            where_filter={"type": preference_type}
        )

    # === 🎯 重大事件管理 ===
    def save_important_event(self, content: str, event_type: str, summary: str,
                            tags: List[str], additional_metadata: Optional[Dict] = None):
        """保存重大事件到ChromaDB"""
        
        # 生成唯一ID
        event_id = f"event_{uuid4().hex}"
        timestamp = get_iso_timestamp()
        current_time = datetime.now()
        natural_time = format_natural_time(current_time)
        
        # 构建包含完整信息的文档内容
        tags_str = "、".join(tags) if tags else "无标签"
        
        event_content = f"""时间：{natural_time}
事件类型：{event_type}
概述：{summary}
标签：{tags_str}
详细内容：{content}"""
        
        # 准备元数据
        metadata = {
            "event_type": event_type,
            "summary": summary,
            "tags": json.dumps(tags, ensure_ascii=False),
            "timestamp": timestamp,
            "content_length": len(content)
        }
        
        # 添加额外元数据
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # 保存到ChromaDB
        try:
            self.collections["important_events"].add(
                ids=[event_id],
                metadatas=[metadata],
                documents=[event_content]
            )
            print(f"✅ 重大事件已保存: {event_type} - {summary}")
            
            # 更新活跃标签
            self.update_active_tags(tags)
            
            return event_id
        except Exception as e:
            print(f"❌ 保存重大事件失败: {e}")
            return None

    def search_important_events(self, query: str, n_results: int = 5,
                               where_filter: Optional[Dict] = None,
                               similarity_threshold: float = 0.7) -> List[Dict]:
        """搜索重大事件"""
        try:
            search_params = {
                "query_texts": [query],
                "n_results": n_results
            }
            
            if where_filter:
                search_params["where"] = where_filter
            
            results = self.collections["important_events"].query(**search_params)
            
            event_memories = []
            if (results and results["documents"] and results["documents"][0] and
                results["metadatas"] and results["metadatas"][0] and
                results["distances"] and results["distances"][0]):
                
                docs_list = results["documents"][0]
                metadatas_list = results["metadatas"][0]
                distances_list = results["distances"][0]
                ids_list = results["ids"][0]
                
                for i, doc_content in enumerate(docs_list):
                    if i < len(distances_list):
                        distance = distances_list[i]
                        if distance <= (1 - similarity_threshold):
                            metadata = metadatas_list[i]
                            tags = json.loads(metadata.get("tags", "[]"))
                            
                            event_memory = {
                                "id": ids_list[i],
                                "content": doc_content,
                                "metadata": metadata,
                                "tags": tags,
                                "similarity": 1 - distance,
                                "event_type": metadata.get("event_type", ""),
                                "summary": metadata.get("summary", ""),
                                "timestamp": metadata.get("timestamp", "")
                            }
                            event_memories.append(event_memory)
            
            return event_memories
        except Exception as e:
            print(f"❌ 搜索重大事件失败: {e}")
            return []

    def get_events_by_type(self, event_type: str, limit: int = 10) -> List[Dict]:
        """根据类型获取重大事件"""
        return self.search_important_events(
            query="", 
            n_results=limit,
            where_filter={"event_type": event_type}
        )

    # === ⏰ 近期关注事件管理 ===
    def save_temp_focus_event(self, content: str, event_time: str, 
                             expire_time: str, tags: List[str]):
        """保存近期关注事件"""
        temp_event = {
            "created_at": get_iso_timestamp(),
            "event_time": event_time,
            "expire_time": expire_time,
            "content": content,
            "tags": tags
        }
        
        # 读取现有事件
        temp_events = self.load_temp_focus_events()
        
        # 添加新事件
        temp_events.append(temp_event)
        
        # 保存更新后的事件列表
        try:
            with open(TEMP_FOCUS_EVENTS_PATH, "w", encoding="utf-8") as f:
                json.dump(temp_events, f, ensure_ascii=False, indent=2)
            print(f"✅ 近期关注事件已保存: {content[:30]}...")
            
            # 更新活跃标签
            self.update_active_tags(tags)
            
            return True
        except Exception as e:
            print(f"❌ 保存近期关注事件失败: {e}")
            return False

    def load_temp_focus_events(self) -> List[Dict]:
        """加载近期关注事件（自动清理过期事件）"""
        if not os.path.exists(TEMP_FOCUS_EVENTS_PATH):
            return []
        
        try:
            with open(TEMP_FOCUS_EVENTS_PATH, encoding="utf-8") as f:
                events = json.load(f)
            
            # 过滤掉过期事件
            current_time = datetime.now()
            valid_events = []
            
            for event in events:
                try:
                    expire_time = datetime.fromisoformat(event["expire_time"])
                    if current_time < expire_time:
                        valid_events.append(event)
                except:
                    # 如果时间解析失败，保留事件
                    valid_events.append(event)
            
            # 如果有事件被清理，更新文件
            if len(valid_events) != len(events):
                with open(TEMP_FOCUS_EVENTS_PATH, "w", encoding="utf-8") as f:
                    json.dump(valid_events, f, ensure_ascii=False, indent=2)
                print(f"🧹 已清理 {len(events) - len(valid_events)} 个过期的关注事件")
            
            return valid_events
        except Exception as e:
            print(f"❌ 加载近期关注事件失败: {e}")
            return []

    def update_temp_focus_event_expire_time(self, event_index: int, new_expire_time: str):
        """更新近期关注事件的过期时间"""
        events = self.load_temp_focus_events()
        
        if 0 <= event_index < len(events):
            events[event_index]["expire_time"] = new_expire_time
            
            try:
                with open(TEMP_FOCUS_EVENTS_PATH, "w", encoding="utf-8") as f:
                    json.dump(events, f, ensure_ascii=False, indent=2)
                print(f"✅ 事件过期时间已更新: {new_expire_time}")
                return True
            except Exception as e:
                print(f"❌ 更新事件过期时间失败: {e}")
        else:
            print(f"❌ 事件索引 {event_index} 超出范围")
        
        return False

    def get_active_focus_events(self) -> List[Dict]:
        """获取当前有效的关注事件"""
        return self.load_temp_focus_events()

    def clear_expired_focus_events(self):
        """手动清理过期的关注事件"""
        valid_events = self.load_temp_focus_events()  # 这会自动清理过期事件
        return len(valid_events)

    # === 🔖 活跃标签 ===
    def update_active_tags(self, new_tags: List[str]):
        """更新活跃标签统计"""
        tags = {}
        if os.path.exists(ACTIVE_TAGS_PATH):
            try:
                with open(ACTIVE_TAGS_PATH, encoding="utf-8") as f:
                    data = json.load(f)
                    tags = data.get("tags", {})
            except:
                tags = {}
        
        # 更新标签计数
        for tag in new_tags:
            tags[tag] = tags.get(tag, 0) + 1
        
        # 保存标签数据
        data = {
            "type": "active_tags", 
            "tags": tags, 
            "last_update": get_timestamp(),
            "total_tags": len(tags),
            "most_frequent": max(tags.items(), key=lambda x: x[1]) if tags else None
        }
        
        with open(ACTIVE_TAGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_active_tags(self, top_n: int = 10) -> Dict:
        """获取最活跃的标签"""
        if not os.path.exists(ACTIVE_TAGS_PATH):
            return {"tags": {}, "top_tags": []}
        
        with open(ACTIVE_TAGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
            tags = data.get("tags", {})
            
            # 按频率排序
            sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
            top_tags = sorted_tags[:top_n]
            
            return {
                "tags": tags,
                "top_tags": top_tags,
                "total_count": sum(tags.values()),
                "unique_count": len(tags),
                "last_update": data.get("last_update", "")
            }

    # === 🔍 综合搜索功能 ===
    def comprehensive_search(self, query: str, search_dialogs: bool = True, 
                           search_facts: bool = True, search_preferences: bool = True,
                           search_events: bool = True, n_results: int = 5) -> Dict:
        """综合搜索所有类型的记忆"""
        results = {
            "query": query,
            "timestamp": get_timestamp(),
            "dialog_memories": [],
            "fact_memories": [],
            "preference_memories": [],
            "event_memories": [],
            "focus_events": []
        }
        
        if search_dialogs:
            results["dialog_memories"] = self.search_dialog_logs(query, n_results)
        
        if search_facts:
            results["fact_memories"] = self.search_fact_memory(query, n_results)
            
        if search_preferences:
            results["preference_memories"] = self.search_user_preferences(query, n_results)
            
        if search_events:
            results["event_memories"] = self.search_important_events(query, n_results)
        
        # 检查是否有相关的关注事件
        focus_events = self.get_active_focus_events()
        relevant_focus_events = []
        for event in focus_events:
            if (query.lower() in event["content"].lower() or 
                any(tag.lower() in query.lower() for tag in event["tags"])):
                relevant_focus_events.append(event)
        results["focus_events"] = relevant_focus_events
        
        return results

    # === 📊 统计和管理功能 ===
    def get_memory_statistics(self) -> Dict:
        """获取记忆系统统计信息"""
        stats = {
            "timestamp": get_timestamp(),
            "dialog_count": 0,
            "fact_count": 0,
            "preference_count": 0,
            "event_count": 0,
            "focus_event_count": 0,
            "user_profile_exists": os.path.exists(PROFILE_PATH),
            "active_tags": self.get_active_tags(5)
        }
        
        try:
            # 统计对话记录数量
            dialogs = self.collections["dialog_logs"].count()
            stats["dialog_count"] = dialogs
        except:
            pass
        
        try:
            # 统计事实记忆数量
            facts = self.collections["facts"].count()
            stats["fact_count"] = facts
        except:
            pass
            
        try:
            # 统计用户偏好数量
            preferences = self.collections["user_preferences"].count()
            stats["preference_count"] = preferences
        except:
            pass
            
        try:
            # 统计重大事件数量
            events = self.collections["important_events"].count()
            stats["event_count"] = events
        except:
            pass
            
        # 统计关注事件数量
        focus_events = self.get_active_focus_events()
        stats["focus_event_count"] = len(focus_events)
        
        return stats

    def cleanup_old_memories(self, days_threshold: int = 30, 
                           importance_threshold: float = 0.3):
        """清理旧的低重要性记忆（可选功能）"""
        print(f"🧹 开始清理 {days_threshold} 天前重要性低于 {importance_threshold} 的记忆...")
        
        from datetime import timedelta
        cutoff_date = (datetime.now() - timedelta(days=days_threshold)).isoformat()
        
        # 这里可以实现具体的清理逻辑
        # 注意：ChromaDB的删除操作需要谨慎处理
        print("⚠️ 清理功能待实现，建议手动管理重要记忆")


# === 🧩 全局实例和便捷函数 ===

# 创建全局记忆系统实例
memory_system = MemorySystem()

# 便捷函数
def save_dialog_log(user_input: str, ai_response: str, topic: str, 
                   sentiment: str, importance: float, tags: List[str]):
    """便捷的对话记录保存函数"""
    return memory_system.save_dialog_log(user_input, ai_response, topic, 
                                       sentiment, importance, tags)

def save_fact_memory(content: str, tags: List[str], source: str = "dialog"):
    """便捷的事实记忆保存函数"""
    return memory_system.save_fact_memory(content, tags, source)

def save_user_preference(content: str, preference_type: str, tags: List[str]):
    """便捷的用户偏好保存函数"""
    return memory_system.save_user_preference(content, preference_type, tags)

def save_important_event(content: str, event_type: str, summary: str, tags: List[str]):
    """便捷的重大事件保存函数"""
    return memory_system.save_important_event(content, event_type, summary, tags)

def save_temp_focus_event(content: str, event_time: str, expire_time: str, tags: List[str]):
    """便捷的临时关注事件保存函数"""
    return memory_system.save_temp_focus_event(content, event_time, expire_time, tags)

def search_memories(query: str, n_results: int = 5):
    """便捷的记忆搜索函数"""
    return memory_system.comprehensive_search(query, n_results=n_results)

def parse_response_with_llm(response_text: str) -> Dict:
    """模拟LLM解析响应（待接入真实LLM）"""
    # 这里应该调用你的LLM来解析用户输入
    # 目前返回示例数据
    return {
        "topic": "流萤和星星",
        "sentiment": "温柔",
        "importance": 0.9,
        "tags": ["流萤", "浪漫", "崩坏星穹铁道"]
    }

# === 🎯 全局便捷函数（缓存版本）===

# 实例化全局记忆系统
_global_memory_system = None

def get_memory_system():
    """获取全局记忆系统实例"""
    global _global_memory_system
    if _global_memory_system is None:
        _global_memory_system = MemorySystem()
    return _global_memory_system

# 缓存相关便捷函数
def cache_user_preference(content: str, preference_type: str, tags: List[str], 
                         confidence: float = 1.0, additional_metadata: Optional[Dict] = None):
    """便捷函数：缓存用户偏好信息"""
    memory_system = get_memory_system()
    return memory_system.cache_user_preference(content, preference_type, tags, confidence, additional_metadata)

def cache_fact_memory(content: str, tags: List[str], source: str = "dialog", 
                     confidence: float = 1.0, additional_metadata: Optional[Dict] = None):
    """便捷函数：缓存事实记忆"""
    memory_system = get_memory_system()
    return memory_system.cache_fact_memory(content, tags, source, confidence, additional_metadata)

def cache_profile_update(profile_data: Dict, source: str = "dialog"):
    """便捷函数：缓存用户画像更新"""
    memory_system = get_memory_system()
    return memory_system.cache_profile_update(profile_data, source)

def load_preference_cache():
    """便捷函数：读取用户偏好缓存"""
    memory_system = get_memory_system()
    return memory_system.load_preference_cache()

def load_fact_cache():
    """便捷函数：读取事实记忆缓存"""
    memory_system = get_memory_system()
    return memory_system.load_fact_cache()

def load_profile_cache():
    """便捷函数：读取用户画像缓存"""
    memory_system = get_memory_system()
    return memory_system.load_profile_cache()

def clear_preference_cache():
    """便捷函数：清空用户偏好缓存"""
    memory_system = get_memory_system()
    return memory_system.clear_preference_cache()

def clear_fact_cache():
    """便捷函数：清空事实记忆缓存"""
    memory_system = get_memory_system()
    return memory_system.clear_fact_cache()

def clear_profile_cache():
    """便捷函数：清空用户画像缓存"""
    memory_system = get_memory_system()
    return memory_system.clear_profile_cache()

def get_cache_status():
    """便捷函数：获取缓存状态"""
    memory_system = get_memory_system()
    return memory_system.get_cache_status()

def clear_all_caches():
    """便捷函数：清空所有缓存"""
    memory_system = get_memory_system()
    return memory_system.clear_all_caches()

# 示例使用缓存功能
def test_cache_system():
    """测试缓存系统"""
    print("🧪 测试缓存系统...")
    
    # 测试缓存用户偏好
    cache_user_preference(
        content="用户喜欢听古典音乐，特别是贝多芬的作品",
        preference_type="音乐偏好",
        tags=["音乐", "古典", "贝多芬"],
        confidence=0.9
    )
    
    cache_user_preference(
        content="用户喜欢听古典音乐和轻音乐",
        preference_type="音乐偏好", 
        tags=["音乐", "古典", "轻音乐"],
        confidence=0.8
    )
    
    # 测试缓存事实记忆
    cache_fact_memory(
        content="用户的生日是3月15日",
        tags=["个人信息", "生日"],
        source="对话",
        confidence=1.0
    )
    
    cache_fact_memory(
        content="用户生日在春天，具体是3月中旬",
        tags=["个人信息", "生日", "春天"],
        source="对话",
        confidence=0.7
    )
    
    # 测试缓存用户画像
    cache_profile_update({
        "age_range": "25-30",
        "interests": ["音乐", "读书", "旅行"],
        "personality": "内向但友善"
    })
    
    cache_profile_update({
        "location": "北京",
        "occupation": "程序员"
    })
    
    # 显示缓存状态
    get_cache_status()
    
    # 测试读取缓存
    print("\n� 测试读取缓存:")
    prefs = load_preference_cache()
    facts = load_fact_cache()
    profile = load_profile_cache()
    
    print(f"用户偏好缓存: {len(prefs)} 条")
    print(f"事实记忆缓存: {len(facts)} 条")
    print(f"用户画像缓存: {len(profile)} 条")
    
    # 可以选择性清理某个缓存
    # clear_preference_cache()  # 注释掉，仅做演示
    
    print("✅ 缓存系统测试完成!")

# === 🧪 用例测试 ===
if __name__ == "__main__":
    print("🚀 初始化记忆系统测试...")
    
    # 测试用户画像（按照修改方案的结构）
    user_profile_example = {
        "basic": {
            "name": "小明",
            "gender": "男",
            "birthday": "2000-01-01"
        },
        "identity": {
            "roles": ["计算机专业学生", "吉他初学者"],
            "job": "学生",
            "dream": "制作自己的AI伴侣",
            "care_about_people": ["妈妈", "闺蜜"]
        }
    }
    memory_system.save_user_profile(user_profile_example)
    
    # 测试对话记录保存
    memory_data = parse_response_with_llm("我真的超级喜欢流萤，她就像一颗在夜里发光的星星……")
    
    dialog_id = memory_system.save_dialog_log(
        user_input="我真的超级喜欢流萤，她就像一颗在夜里发光的星星……",
        ai_response="哇，主人说得好温柔喵~ 流萤真的就是那样治愈人心的存在~",
        topic=memory_data["topic"],
        sentiment=memory_data["sentiment"],
        importance=memory_data["importance"],
        tags=memory_data["tags"]
    )
    
    # 测试事实记忆保存
    fact_id = memory_system.save_fact_memory(
        "主人喜欢流萤和星星的意象，认为她很治愈。", 
        memory_data["tags"]
    )
    
    # 测试用户偏好保存
    preference_id = memory_system.save_user_preference(
        "用户喜欢轻松搞笑的动画番剧，最喜欢的是《孤独摇滚》。",
        "娱乐",
        ["番剧", "搞笑", "轻松"]
    )
    
    # 测试重大事件保存
    event_id = memory_system.save_important_event(
        "用户希望考上理想的研究生院校，目前正在准备复习，每天都有学习计划。",
        "考试",
        "用户计划考研",
        ["目标", "长期", "紧张"]
    )
    
    # 测试临时关注事件保存
    temp_event_saved = memory_system.save_temp_focus_event(
        "用户将于7月8日参加英语四级考试",
        "2025-07-08",
        "2025-07-09T23:59:59",
        ["考试", "紧张", "短期焦点"]
    )
    
    # 测试搜索功能
    print("\n🔍 测试综合搜索功能:")
    search_results = memory_system.comprehensive_search("流萤")
    print(f"搜索到 {len(search_results['dialog_memories'])} 条对话记录")
    print(f"搜索到 {len(search_results['fact_memories'])} 条事实记忆")
    print(f"搜索到 {len(search_results['preference_memories'])} 条用户偏好")
    print(f"搜索到 {len(search_results['event_memories'])} 条重大事件")
    print(f"搜索到 {len(search_results['focus_events'])} 条关注事件")
    
    # 显示统计信息
    print("\n📊 记忆系统统计:")
    stats = memory_system.get_memory_statistics()
    print(f"对话记录: {stats['dialog_count']} 条")
    print(f"事实记忆: {stats['fact_count']} 条")
    print(f"用户偏好: {stats['preference_count']} 条")
    print(f"重大事件: {stats['event_count']} 条")
    print(f"关注事件: {stats['focus_event_count']} 条")
    print(f"活跃标签: {stats['active_tags']['unique_count']} 个")
    
    # 测试关注事件管理
    print("\n⏰ 测试关注事件管理:")
    focus_events = memory_system.get_active_focus_events()
    for i, event in enumerate(focus_events):
        print(f"事件 {i}: {event['content'][:50]}... (过期时间: {event['expire_time']})")
    
    # 测试缓存系统
    print("\n🗂️ 测试缓存系统:")
    test_cache_system()
    
    print("\n✅ 记忆系统测试完成！")