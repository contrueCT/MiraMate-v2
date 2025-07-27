from math import fabs
import os
import json
from typing import Dict, Any

# 导入需要的 LangChain LLM 类
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from sympy import false

# --- 配置 ---
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 从 modules/ 目录向上回溯两层，到达 src/MiraMate/ 的父目录，即项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, '..', '..', '..'))

# 3. 在项目根目录下构建指向 configs/llm_config.json 的路径
LLM_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "llm_config.json")

def create_llm_from_config(config: Dict[str, Any], is_main_llm: bool = False, **kwargs) -> BaseChatModel:
    """
    根据单条配置字典，动态创建并返回一个LangChain LLM实例。

    :param config: 包含 'api_type', 'model', 'api_key' 等的字典。
    :param kwargs: 额外的、要传递给LLM构造函数的参数 (例如 streaming=True)。
    :return: 一个 BaseChatModel 的实例。
    """
    api_type = config.get("api_type")
    model_name = config.get("model")
    api_key = config.get("api_key") or os.getenv("API_KEY") # 优先使用配置中的key，否则尝试环境变量
    base_url = config.get("base_url")

    if not all([api_type, model_name, api_key]):
        raise ValueError(f"配置不完整，缺少 'api_type', 'model', 或 'api_key': {config}")

    # 合并默认kwargs和传入的kwargs，传入的优先
    final_kwargs = {**config.get("model_kwargs", {}), **kwargs}

    print(f"正在创建模型实例: [Type: {api_type}, Model: {model_name}]")

    # --- 根据 api_type 动态选择要实例化的类 ---
    if api_type == "openai":
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            streaming=is_main_llm, # 只有主模型才默认开启流式
            **final_kwargs
        )
    elif api_type == "gemini":
        # 对于 Gemini，构造函数中没有 streaming 参数。
        # LangChain 会在调用 .stream() 时自动处理流式请求。
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            **final_kwargs
        )
    # 在这里可以继续添加对其他api类型的支持，比如 'anthropic'
    # elif api_type == "anthropic":
    #     from langchain_anthropic import ChatAnthropic
    #     return ChatAnthropic(model=model_name, api_key=api_key, **final_kwargs)
    
    else:
        raise ValueError(f"不支持的 API 类型: '{api_type}'。目前仅支持 'openai' 和 'gemini'。")

def load_llms_from_json(config_path: str) -> tuple[BaseChatModel, BaseChatModel]:
    """
    从JSON配置文件加载主模型和次模型。
    约定：列表中的第一个配置为 main_llm，第二个为 small_llm。
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            configs = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到 LLM 配置文件 '{config_path}'。")
        print("💡 已为您创建一个模板文件，请填写您的 API 信息后再重新运行。")
        # 创建一个模板文件引导用户
        template_config = [
            {
                "model": "gpt-4o",
                "api_key": "YOUR_OPENAI_API_KEY",
                "base_url": "https://api.openai.com/v1",
                "api_type": "openai",
                "model_kwargs": {"temperature": 0.8}
            },
            {
                "model": "gemini-1.5-flash-latest",
                "api_key": "YOUR_GEMINI_API_KEY",
                "api_type": "gemini",
                "model_kwargs": {"temperature": 0.0}
            }
        ]
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(template_config, f, ensure_ascii=False, indent=4)
        return None, None
    except json.JSONDecodeError:
        print(f"❌ 错误: LLM 配置文件 '{config_path}' 格式错误，请检查是否为有效的JSON。")
        return None, None

    if not isinstance(configs, list) or not configs:
        print(f"❌ 错误: '{config_path}' 的内容必须是一个非空的JSON列表。")
        return None, None

    # --- 根据约定创建 main 和 small 模型 ---
    # 第一个作为主模型
    main_config = configs[0]
    # 第二个作为次模型；如果只有一个，则次模型也使用主模型的配置
    small_config = configs[1] if len(configs) > 1 else configs[0]

    # 为主模型和次模型设置不同的默认参数
    # 主模型用于生成，需要流式和创造性
    main_llm_instance = create_llm_from_config(main_config,is_main_llm=True)
    
    # 次模型用于分析和JSON输出，需要确定性和特定格式
    small_llm_instance = create_llm_from_config(
        small_config,
        # 如果是 openai 兼容的 api，可以强制 json 输出
        model_kwargs={"response_format": {"type": "json_object"}} if small_config.get("api_type") == "openai" else {},
        is_main_llm=False
    )

    return main_llm_instance, small_llm_instance

# --- 模块主逻辑：加载模型并导出 ---
main_llm, small_llm = load_llms_from_json(LLM_CONFIG_PATH)

if main_llm is None or small_llm is None:
    print("⚠️ 由于模型加载失败，应用可能无法正常运行。请检查配置文件。")
else:
    print("✅ 主模型和次模型已成功加载！")