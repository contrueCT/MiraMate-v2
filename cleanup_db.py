import os
import chromadb
import json

# --- 1. 配置路径 (与 memory_system.py 保持一致) ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(PROJECT_ROOT, "memory", "memory_storage")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# 要操作的集合名称
COLLECTION_NAMES = [
    "facts",
    "user_preferences",
    "important_events",
    "dialog_logs" # 也加入对话记录，以备不时之需
]

def format_record_for_display(record, index):
    """格式化单条记录以便于清晰展示。"""
    doc = record.get('document', 'N/A')
    # 截断过长的文档内容
    display_doc = (doc[:500] + '...') if len(doc) > 100 else doc
    # 尝试解析metadata中的timestamp
    timestamp = record.get('metadata', {}).get('timestamp', 'N/A')
    return f"  [{index}] - {timestamp}\n      Content: {display_doc}"

def selective_cleanup_tool():
    """
    一个交互式的ChromaDB清理工具，支持预览和选择性删除。
    """
    print("=" * 60)
    print("✨ ChromaDB 数据库高级清理工具 ✨")
    print("=" * 60)
    print(f"\n数据库路径: {CHROMA_DB_DIR}")

    if not os.path.exists(CHROMA_DB_DIR):
        print(f"\n❌ 错误: 找不到数据库目录。请确保脚本位于项目根目录。")
        return

    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        print("✅ 成功连接到数据库。")
    except Exception as e:
        print(f"\n❌ 错误: 连接数据库失败: {e}")
        return

    print("\n此脚本将帮助您预览并删除集合中的记录。")
    print("-" * 60)

    for collection_name in COLLECTION_NAMES:
        print(f"\n{'='*20} 正在处理集合: '{collection_name}' {'='*20}")
        try:
            collection = client.get_collection(name=collection_name)
            count = collection.count()

            if count == 0:
                print(f"✅ 集合为空，无需清理。")
                continue
            
            print(f"发现 {count} 条记录。")

            # --- 预览最新记录 ---
            preview_limit = min(5, count)
            print(f"\n--- 预览最新的 {preview_limit} 条记录 (按插入顺序) ---")
            
            # ChromaDB 的 get() 默认按插入顺序返回
            latest_records_data = collection.get(limit=preview_limit, include=["documents", "metadatas"])
            records_to_display = []
            for i in range(len(latest_records_data['ids'])):
                record = {
                    "id": latest_records_data['ids'][i],
                    "document": latest_records_data['documents'][i],
                    "metadata": latest_records_data['metadatas'][i]
                }
                records_to_display.append(record)
            
            # 因为 get() 返回的是最早的记录，我们需要反转一下来显示最新的
            records_to_display.reverse()
            for i, record in enumerate(records_to_display):
                print(format_record_for_display(record, i + 1))
            print("--------------------------------------------------")

            # --- 提供操作选项 ---
            while True:
                choice = input(
                    "\n请选择操作:\n"
                    "  s - 跳过 (Skip)\n"
                    "  d - 删除最新的N条记录 (Delete latest N)\n"
                    "  a - 删除全部 (Delete All)\n"
                    "请输入您的选择 (s/d/a): "
                ).lower()

                if choice in ['s', 'd', 'a']:
                    break
                else:
                    print("❌ 无效输入，请重新选择。")

            if choice == 's':
                print(f"- 已跳过集合 '{collection_name}'。")
                continue

            elif choice == 'd':
                while True:
                    try:
                        num_to_delete = int(input(f"您想删除最新的多少条记录? (输入数字, 最多 {count}): "))
                        if 0 < num_to_delete <= count:
                            break
                        else:
                            print(f"❌ 请输入一个在 1 到 {count} 之间的数字。")
                    except ValueError:
                        print("❌ 无效输入，请输入一个数字。")

                # 获取要删除的记录ID
                all_ids = collection.get(include=[])['ids']
                ids_to_delete = all_ids[-num_to_delete:]

                # 最终确认
                print(f"\n‼️ 最终确认：您将从 '{collection_name}' 中永久删除最新的 {num_to_delete} 条记录。")
                confirm_word = input("请输入 'DELETE' 来确认: ")
                if confirm_word == 'DELETE':
                    collection.delete(ids=ids_to_delete)
                    print(f"🗑️ 成功删除 {len(ids_to_delete)} 条记录。")
                else:
                    print("- 操作已取消。")
            
            elif choice == 'a':
                print(f"\n‼️ 最终确认：您将从 '{collection_name}' 中永久删除全部 {count} 条记录。")
                confirm_word = input("请输入 'DELETE ALL' 来确认: ")
                if confirm_word == 'DELETE ALL':
                    client.delete_collection(name=collection_name)
                    print(f"🗑️ 成功删除并清空了整个集合 '{collection_name}'。")
                else:
                    print("- 操作已取消。")

        except ValueError:
            print(f"ℹ️ 未找到名为 '{collection_name}' 的集合，跳过。")
        except Exception as e:
            print(f"\n❌ 处理集合 '{collection_name}' 时发生错误: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("所有操作已完成。")
    print("=" * 60)

if __name__ == "__main__":
    selective_cleanup_tool()