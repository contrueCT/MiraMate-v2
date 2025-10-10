# 已经接入了对话链，主要负责用户画像、对话记忆、事实记忆等的存储和检索，后续需要在对话完后的步骤中更新记忆内容：对话记忆、事实记忆、用户偏好、重大事件等。

import json
import os
# 在导入任何模型前设置环境变量
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import chromadb
from datetime import datetime, timezone
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
    
    return f"{dt.year}年{dt.month}月{dt.day}日{weekday}{time_period}{dt.hour}点{dt.minute}分"

# === 📦 二、存储目录设置 - Docker环境适配 ===
def get_project_root():
    """获取项目根目录，支持Docker环境"""
    if os.getenv('DOCKER_ENV'):
        return '/app'
    # 开发环境：从 modules/ 向上3级到项目根目录
    # 当前文件: src/MiraMate/modules/memory_system.py
    # 项目根目录: 向上3级
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(MODULE_DIR, '..', '..', '..'))

PROJECT_ROOT = get_project_root()

# Docker环境适配的存储路径
if os.getenv('DOCKER_ENV'):
    # Docker环境：使用环境变量指定的内存数据库目录
    MEMORY_DB_DIR = os.getenv('MEMORY_DB_DIR', '/app/memory_db')
    BASE_DIR = os.path.join(MEMORY_DB_DIR, "memory_storage")
else:
    # 开发环境：使用相对路径
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

    # --- 内部：安全查询 + 索引自修复 ---
    def _safe_query(self, collection_key: str, search_params: Dict):
        """
        对 Chroma 集合执行查询，若检测到 HNSW 段索引丢失/损坏，则自动重建并重试一次。
        不改变原有查询行为，只有在特定错误出现时才介入自修复。
        """
        try:
            return self.collections[collection_key].query(**search_params)
        except Exception as e:
            msg = str(e).lower()
            # 常见报错："error creating hnsw segment reader: Nothing found on disk"
            if ("hnsw" in msg) and ("segment" in msg or "nothing found on disk" in msg):
                print(f"\u26a0\ufe0f 检测到集合 '{collection_key}' 的 HNSW 索引缺失/损坏，正在尝试自动重建...")
                self._rebuild_collection_index(collection_key)
                # 重试一次
                return self.collections[collection_key].query(**search_params)
            # 其它异常按原样抛出
            raise

    def _rebuild_collection_index(self, collection_key: str, batch_size: int = 512):
        """
        重建指定集合的 HNSW 索引：
        1) 读取当前集合的数据（ids/documents/metadatas/embeddings-若可用）
        2) 删除集合并按原配置重建
        3) 分批写回数据，触发索引重建（若 embeddings 可用则直接写入以避免重新计算）
        """
        try:
            coll = self.collections.get(collection_key)
            if coll is None:
                print(f"❌ 重建失败：集合 '{collection_key}' 不存在")
                return

            # 读取数据，优先包含 embeddings；若不支持则降级
            data = None
            try:
                data = coll.get(include=["ids", "documents", "metadatas", "embeddings"])
            except Exception:
                data = coll.get(include=["ids", "documents", "metadatas"])  # 某些存储后端可能不支持 embeddings 导出

            ids = data.get("ids", []) or []
            docs = data.get("documents", []) or []
            metas = data.get("metadatas", []) or []
            embeds = data.get("embeddings") if isinstance(data, dict) else None

            total = len(ids)
            print(f"[MemorySystem] 备份集合 '{collection_key}' 中的 {total} 条记录用于重建")

            # 删除并重建集合
            self.client.delete_collection(name=collection_key)
            new_coll = self.client.get_or_create_collection(
                name=collection_key,
                embedding_function=self.embedding_function,
                metadata=self.hnsw_metadata_config
            )
            self.collections[collection_key] = new_coll

            # 若原集合为空，直接返回
            if total == 0:
                print(f"[MemorySystem] 集合 '{collection_key}' 为空，已完成空索引重建")
                return

            # 分批写回
            has_embeds = isinstance(embeds, list) and len(embeds) == total
            for i in range(0, total, batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_docs = docs[i:i+batch_size]
                batch_metas = metas[i:i+batch_size]
                if has_embeds:
                    batch_embeds = embeds[i:i+batch_size]
                    new_coll.add(
                        ids=batch_ids,
                        documents=batch_docs,
                        metadatas=batch_metas,
                        embeddings=batch_embeds
                    )
                else:
                    new_coll.add(
                        ids=batch_ids,
                        documents=batch_docs,
                        metadatas=batch_metas
                    )
            print(f"✅ 集合 '{collection_key}' 的 HNSW 索引重建完成，共写回 {total} 条记录")
        except Exception as e:
            print(f"❌ 重建集合 '{collection_key}' 索引失败: {e}")

    def _parse_iso_datetime(self, dt_str: str) -> Optional[datetime]:
        """尽可能稳健地解析 ISO 时间戳，返回 UTC 时区的 datetime。
        兼容示例：
        - 2025-08-24T00:00:00Z
        - 2025-08-24T00:00:00.123Z
        - 2025-08-24T00:00:00+08:00
        - 2025-08-24 00:00:00  （无时区，按 UTC 处理）
        """
        if not isinstance(dt_str, str):
            return None
        s = dt_str.strip().replace(' ', 'T')
        # 兼容以 Z 结尾（UTC）
        if s.endswith('Z'):
            s = s[:-1] + '+00:00'
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None

        # 统一转为 UTC 有时区时间
        if dt.tzinfo is None:
            # 无时区信息时，按 UTC 处理
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt

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
    def save_dialog_log(self, user_input: str, ai_response: str, topic: str, 
                       sentiment: str, importance: float, tags: List[str], 
                       additional_metadata: Optional[Dict] = None):
        """保存对话记录到ChromaDB"""
        dialog_id = f"dialog_{uuid4().hex}"
        timestamp = get_iso_timestamp()
        current_time = datetime.now()
        natural_time = format_natural_time(current_time)
        tags_str = "、".join(tags) if tags else "无"
        
        # 将关键元数据信息转化为自然语言
        document_header = f"[对话摘要] 这是一段记录于 {natural_time} 的对话。对话主题是“{topic}”，整体情感基调为“{sentiment}”，相关标签为“{tags_str}”。\n\n"
        dialog_body = f"[对话内容]\n用户：{user_input}\nAI：{ai_response}"
        dialog_content = document_header + dialog_body
        
        #  metadata 只保留用于精确过滤的结构化数据
        metadata = {
            "type": "dialog_log",
            "timestamp": timestamp,
            "topic": topic,
            "sentiment": sentiment,
            "importance": importance,
            "tags": json.dumps(tags, ensure_ascii=False)
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        try:
            self.collections["dialog_logs"].add(
                ids=[dialog_id],
                metadatas=[metadata],
                documents=[dialog_content]
            )
            print(f"✅ 对话记录已保存: {topic} (重要性: {importance})")
            self.update_active_tags(tags)
            return dialog_id
        except Exception as e:
            print(f"❌ 保存对话记录失败: {e}")
            return None

    def search_dialog_logs(self, query: str, n_results: int = 5, 
                          where_filter: Optional[Dict] = None, 
                          threshold: float = 0.5) -> List[Dict]:
        """搜索对话记录"""
        try:
            search_params = {
                "query_texts": [query],
                "n_results": n_results
            }
            
            if where_filter:
                search_params["where"] = where_filter
            
            # 使用安全查询，必要时自动重建索引
            results = self._safe_query("dialog_logs", search_params)
            
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
                        if distance <= threshold: 
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
            
            return dialog_list[:limit*2]
        except Exception as e:
            print(f"❌ 获取最近对话失败: {e}")
            return []

    # === 概念知识（事实记忆）===
    # 先存到缓冲文件（持久化），然后在空闲时经过模型处理后保存到ChromaDB
    def save_fact_memory(self, content: str, tags: List[str], 
                        source: str = "dialog", confidence: float = 1.0,
                        additional_metadata: Optional[Dict] = None):
        """保存事实记忆到ChromaDB"""
        fact_id = f"fact_{uuid4().hex}"
        timestamp = get_iso_timestamp()
        natural_time = format_natural_time(datetime.now())
        tags_str = "、".join(tags) if tags else "无"

        fact_content = f"[事实记忆] 这是一条记录于 {natural_time} 的事实，来源是“{source}”，相关标签为“{tags_str}”。事实内容：{content}"
        
        metadata = {
            "type": "fact",
            "timestamp": timestamp,
            "source": source,
            "confidence": confidence,
            "tags": json.dumps(tags, ensure_ascii=False)
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        try:
            self.collections["facts"].add(
                ids=[fact_id],
                metadatas=[metadata],
                documents=[fact_content]
            )
            print(f"✅ 事实记忆已保存: {content[:30]}... (置信度: {confidence})")
            self.update_active_tags(tags)
            return fact_id
        except Exception as e:
            print(f"❌ 保存事实记忆失败: {e}")
            return None

    def search_fact_memory(self, query: str, n_results: int = 3,
                          where_filter: Optional[Dict] = None,
                          threshold: float = 0.5) -> List[Dict]:
        """搜索事实记忆"""
        try:
            search_params = {
                "query_texts": [query],
                "n_results": n_results
            }
            
            if where_filter:
                search_params["where"] = where_filter
            
            # 使用安全查询，必要时自动重建索引
            results = self._safe_query("facts", search_params)
            
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
                        if distance <= threshold:
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

    # === 缓存管理方法 ===
    
    def cache_user_preference(self, content: str, preference_type: str, 
                             tags: List[str], confidence: float = 1.0):
        """缓存用户偏好信息到本地JSON文件"""
        
        # 创建缓存条目
        cache_entry = {
            "id": f"preference_cache_{uuid4().hex}",
            "content": content,
            "preference_type": preference_type,
            "tags": tags,
            "confidence": confidence,
            "timestamp": get_iso_timestamp(),
            "natural_time": format_natural_time(datetime.now())
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
                         source: str = "dialog", confidence: float = 1.0):
        """缓存事实记忆到本地JSON文件"""
        cache_entry = {
            "id": f"fact_cache_{uuid4().hex}",
            "content": content,
            "tags": tags,
            "source": source,
            "confidence": confidence,
            "timestamp": get_iso_timestamp(),
            "natural_time": format_natural_time(datetime.now()),
            "content_length": len(content)
        }
        
        facts_cache = self._load_cache_file(FACT_CACHE_PATH)
        facts_cache.append(cache_entry)
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

    # === 用户偏好信息 ===
    # 先存到缓冲文件（持久化），然后在空闲时经过模型处理后保存到ChromaDB
    def save_user_preference(self, content: str, preference_type: str, 
                            tags: List[str], additional_metadata: Optional[Dict] = None):
        """保存用户偏好信息到ChromaDB"""
        preference_id = f"preference_{uuid4().hex}"
        timestamp = get_iso_timestamp()
        natural_time = format_natural_time(datetime.now())
        tags_str = "、".join(tags) if tags else "无"

        preference_content = f"[用户偏好] 这是一条记录于 {natural_time} 的关于用户的偏好信息，类型为“{preference_type}”，相关标签为“{tags_str}”。偏好内容：{content}"
        
        metadata = {
            "type": "preference",
            "preference_type": preference_type,
            "tags": json.dumps(tags, ensure_ascii=False),
            "timestamp": timestamp
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        try:
            self.collections["user_preferences"].add(
                ids=[preference_id],
                metadatas=[metadata],
                documents=[preference_content]
            )
            print(f"✅ 用户偏好已保存: {preference_type} - {content[:30]}...")
            self.update_active_tags(tags)
            return preference_id
        except Exception as e:
            print(f"❌ 保存用户偏好失败: {e}")
            return None

    def search_user_preferences(self, query: str, n_results: int = 5,
                               where_filter: Optional[Dict] = None,
                               threshold: float = 0.5) -> List[Dict]:
        """搜索用户偏好信息"""
        try:
            search_params = {
                "query_texts": [query],
                "n_results": n_results
            }
            
            if where_filter:
                search_params["where"] = where_filter
            
            # 使用安全查询，必要时自动重建索引
            results = self._safe_query("user_preferences", search_params)
            
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
                        if distance <= threshold:
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

    # === 重大事件管理 ===
    def save_important_event(self, content: str, event_type: str, summary: str,
                            tags: List[str], additional_metadata: Optional[Dict] = None):
        """保存重大事件到ChromaDB"""
        event_id = f"event_{uuid4().hex}"
        timestamp = get_iso_timestamp()
        natural_time = format_natural_time(datetime.now())
        tags_str = "、".join(tags) if tags else "无"

        event_content = f"[重大事件] 这是一条记录于 {natural_time} 的重大事件。事件类型为“{event_type}”，概要是“{summary}”，相关标签为“{tags_str}”。\n\n[详细内容]\n{content}"
        
        metadata = {
            "type": "important_event",
            "event_type": event_type,
            "summary": summary,
            "tags": json.dumps(tags, ensure_ascii=False),
            "timestamp": timestamp
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        try:
            self.collections["important_events"].add(
                ids=[event_id],
                metadatas=[metadata],
                documents=[event_content]
            )
            print(f"✅ 重大事件已保存: {event_type} - {summary}")
            self.update_active_tags(tags)
            return event_id
        except Exception as e:
            print(f"❌ 保存重大事件失败: {e}")
            return None

    def search_important_events(self, query: str, n_results: int = 5,
                               where_filter: Optional[Dict] = None,
                               threshold: float = 0.5) -> List[Dict]:
        """搜索重大事件"""
        try:
            search_params = {
                "query_texts": [query],
                "n_results": n_results
            }
            
            if where_filter:
                search_params["where"] = where_filter
            
            # 使用安全查询，必要时自动重建索引
            results = self._safe_query("important_events", search_params)
            
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
                        if distance <= threshold:
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
            "id": f"temp_{uuid4().hex}",
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
            # 兼容历史数据：为缺少 id 的事件补齐并落盘
            updated_for_ids = False
            for e in events:
                if "id" not in e or not e.get("id"):
                    e["id"] = f"temp_{uuid4().hex}"
                    updated_for_ids = True
            if updated_for_ids:
                with open(TEMP_FOCUS_EVENTS_PATH, "w", encoding="utf-8") as f:
                    json.dump(events, f, ensure_ascii=False, indent=2)
            
            # 过滤掉过期事件（统一使用 UTC 比较）
            now_utc = datetime.now(timezone.utc)
            valid_events = []
            
            for event in events:
                expire_raw = event.get("expire_time")
                expire_dt = self._parse_iso_datetime(expire_raw)
                if expire_dt is None:
                    # 解析失败则保留，避免误删
                    valid_events.append(event)
                else:
                    if now_utc < expire_dt:
                        valid_events.append(event)
                    # 否则丢弃（已过期）

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
        # 根据过期机制自动清理
        valid_events = self.load_temp_focus_events()
        return len(valid_events)

    def delete_temp_focus_events_by_ids(self, ids: List[str]) -> int:
        """按 ID 删除临时关注事件，返回删除数量。"""
        if not ids:
            return 0
        try:
            events = self.load_temp_focus_events()
            before = len(events)
            id_set = set(ids)
            remaining = [e for e in events if e.get("id") not in id_set]
            if len(remaining) != before:
                with open(TEMP_FOCUS_EVENTS_PATH, "w", encoding="utf-8") as f:
                    json.dump(remaining, f, ensure_ascii=False, indent=2)
                removed = before - len(remaining)
                print(f"🧹 已按ID删除 {removed} 条临时关注事件")
                return removed
            return 0
        except Exception as e:
            print(f"❌ 按ID删除临时关注事件失败: {e}")
            return 0

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

    # TODO 未实现功能，未来考虑开发
    def cleanup_old_memories(self, days_threshold: int = 30, 
                           importance_threshold: float = 0.3):
        """清理旧的低重要性记忆（可选功能）"""
        print(f"🧹 开始清理 {days_threshold} 天前重要性低于 {importance_threshold} 的记忆...")
        
        from datetime import timedelta
        cutoff_date = (datetime.now() - timedelta(days=days_threshold)).isoformat()
        

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
                         confidence: float = 1.0):
    """便捷函数：缓存用户偏好信息"""
    memory_system = get_memory_system()
    # 现在的调用是正确的，参数数量和类型都匹配
    return memory_system.cache_user_preference(content, preference_type, tags, confidence)

def cache_fact_memory(content: str, tags: List[str], source: str = "dialog", 
                     confidence: float = 1.0): # <-- 移除 additional_metadata
    """便捷函数：缓存事实记忆"""
    memory_system = get_memory_system()
    # 现在的调用是正确的
    return memory_system.cache_fact_memory(content, tags, source, confidence)

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
