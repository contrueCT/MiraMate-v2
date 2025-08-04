import traceback
import asyncio
import os
import pyfiglet
from uuid import uuid4
from dotenv import load_dotenv
from src.MiraMate.core.idle_processor import IdleProcessor

load_dotenv()
print("✅ .env 文件已加载 (如果存在)")

# 导入我们最终的链
# 使用绝对路径导入，因为我们采用了 src 布局
from MiraMate.core.pipeline import final_chain, get_memory_for_session
from MiraMate.core.post_sync_chain import post_sync_chain
from MiraMate.core.post_async_chain import post_async_chain

# 定义一些颜色，让界面更好看
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_welcome_message():
    """打印漂亮的欢迎横幅"""
    ascii_banner = pyfiglet.figlet_format("MiraMate CLI")
    print(f"{Colors.HEADER}{ascii_banner}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}欢迎来到 MiraMate 命令行测试界面！{Colors.ENDC}")
    print("--------------------------------------------------")
    print("这是一个用于测试核心对话链功能的交互式环境。")
    print("👉 输入 'multi' 开始多行输入模式，再次输入 'multi' 结束。")
    print("👉 输入 'exit' 或 'quit' 退出程序。")
    print("--------------------------------------------------\n")

async def run_interactive_session():
    """运行一个交互式的命令行会话"""
    session_id = str(uuid4())
    print(f"{Colors.WARNING}新会话已创建，Session ID: {session_id}{Colors.ENDC}\n")
    memory_instance = get_memory_for_session(session_id)

    idle_processor = IdleProcessor(idle_threshold_seconds=1200) # 20分钟
    idle_processor.start()

    in_multiline_mode = False
    multiline_input = []

    while True:
        if in_multiline_mode:
            prompt_text = f"{Colors.OKGREEN}...(多行输入中): {Colors.ENDC}"
        else:
            prompt_text = f"{Colors.OKGREEN}You: {Colors.ENDC}"
        
        try:
            user_input = input(prompt_text)

            # 更新最后互动时间
            idle_processor.update_last_interaction_time()

        except (KeyboardInterrupt, EOFError):
            # 停止 idle_processor 线程
            idle_processor.stop()

            print(f"\n{Colors.WARNING}再见！{Colors.ENDC}")
            break

        # 处理命令
        if user_input.lower() in ['exit', 'quit']:
            print(f"{Colors.WARNING}再见！{Colors.ENDC}")
            break
        
        if user_input.lower() == 'multi':
            in_multiline_mode = not in_multiline_mode
            if in_multiline_mode:
                print(f"{Colors.OKCYAN}已进入多行输入模式。再次输入 'multi' 并回车来发送消息。{Colors.ENDC}")
            else:
                print(f"{Colors.OKCYAN}已退出多行输入模式，正在发送消息...{Colors.ENDC}")
                user_input = "\n".join(multiline_input)
                multiline_input = [] # 清空
            
            if in_multiline_mode:
                continue

        if in_multiline_mode:
            multiline_input.append(user_input)
            continue

        if not user_input.strip():
            continue

        print(f"{Colors.OKBLUE}AI: ", end="", flush=True)

        full_response = ""

        history_before_this_turn = memory_instance.messages

        try:
            # --- 核心交互：流式调用主链 ---
            async for chunk in final_chain.astream(
                {"user_input": user_input},
                config={"configurable": {"session_id": session_id}}
            ):
                print(f"{Colors.OKBLUE}{chunk}{Colors.ENDC}", end="", flush=True)
                full_response += chunk
            
            print() # 换行


            print("[同步后处理] 正在分析状态变化(带历史)...")
            sync_result = await post_sync_chain.ainvoke({
                "conversation_history": history_before_this_turn, # <-- 传入本次对话之前的历史
                "user_input": user_input,
                "ai_response": full_response
            })
            print(f"[同步后处理] 分析完成")

            # --- 2b. 异步处理 (核心新增部分) ---
            print("[异步后处理] 正在将记忆分析任务提交到后台...")
            # 我们使用 asyncio.create_task() 来模拟一个“发射后不管”的后台任务
            # 这不会阻塞下一次的用户输入
            asyncio.create_task(
                post_async_chain.ainvoke({
                    "conversation_history": history_before_this_turn,
                    "user_input": user_input,
                    "ai_response": full_response
                })
            )

        except Exception as e:
            print(f"\n{Colors.FAIL}错误: 对话链执行失败!{Colors.ENDC}")
            print(f"{Colors.FAIL}{e}{Colors.ENDC}")
            # 在这里可以打印更详细的traceback用于调试
            traceback.print_exc()

        print("-" * 50)




if __name__ == "__main__":
    # 检查必要的配置文件是否存在
    if not os.path.exists("configs/llm_config.json"):
        print(f"{Colors.FAIL}错误: 找不到 'configs/llm_config.json' 文件。{Colors.ENDC}")
        print("请确保已在项目根目录下创建 configs 文件夹，并填入正确的 llm_config.json。")
    else:
        print_welcome_message()
        asyncio.run(run_interactive_session())