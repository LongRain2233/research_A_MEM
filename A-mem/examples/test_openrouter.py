"""
OpenRouter 连接测试脚本
用法：
  1. 在项目根目录 .env 文件中设置：OPENROUTER_API_KEY=sk-or-你的密钥
  2. 运行：python examples/test_openrouter.py
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# 按优先级排列的候选模型（付费模型更稳定）
CANDIDATE_MODELS = [
    "deepseek/deepseek-chat-v3-0324",          # DeepSeek V3, 便宜且国内友好
    "google/gemini-2.5-flash-preview-05-20",    # Gemini 2.5 Flash, 便宜快速
    "qwen/qwen3-4b:free",                       # 通义千问, 免费
    "mistralai/mistral-small-3.1-24b-instruct:free",  # Mistral, 免费
    "google/gemma-3-27b-it:free",               # Gemma 3, 免费
]


def try_model(client, model_id):
    """尝试用指定模型发送请求，成功返回 response，失败返回 None"""
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Say 'hello' in one word."}],
            max_tokens=10,
        )
        return response
    except Exception as e:
        print(f"   ⚠️ {model_id} 失败: {e}")
        return None


def test_openrouter_connection():
    """测试 OpenRouter API 是否能正常连接"""

    # 1. 检查 API Key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ 未找到 OPENROUTER_API_KEY 环境变量")
        print("   请在 .env 文件中设置：OPENROUTER_API_KEY=sk-or-你的密钥")
        return False

    print(f"✅ 找到 API Key: {api_key[:12]}...{api_key[-4:]}")

    # 2. 测试基础 API 连接（自动尝试多个模型）
    print("\n🔗 测试 OpenRouter API 连接...")
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    working_model = None
    for model_id in CANDIDATE_MODELS:
        print(f"   尝试模型: {model_id} ...")
        response = try_model(client, model_id)
        if response:
            reply = response.choices[0].message.content.strip()
            print(f"✅ API 连接成功！")
            print(f"   模型: {response.model}")
            print(f"   回复: {reply}")
            if response.usage:
                print(f"   Token: {response.usage.prompt_tokens} prompt + "
                      f"{response.usage.completion_tokens} completion")
            working_model = model_id
            break
        time.sleep(1)  # 等一秒再试下一个

    if not working_model:
        print("❌ 所有候选模型均连接失败，请检查网络或 API Key")
        return False

    # 3. 测试通过 AgenticMemorySystem 调用
    print(f"\n🧠 测试 AgenticMemorySystem + OpenRouter（模型: {working_model}）...")
    try:
        from agentic_memory.memory_system import AgenticMemorySystem

        memory_system = AgenticMemorySystem(
            model_name="all-MiniLM-L6-v2",
            llm_backend="openrouter",
            llm_model=working_model,
        )
        print("✅ AgenticMemorySystem 初始化成功")

        # 添加一条测试记忆
        memory_id = memory_system.add_note(
            content="这是一条 OpenRouter 连接测试记忆",
            tags=["test", "openrouter"],
            category="测试",
        )
        print(f"✅ 记忆添加成功，ID: {memory_id}")

        # 读取记忆
        memory = memory_system.read(memory_id)
        print(f"✅ 记忆读取成功:")
        print(f"   内容: {memory.content}")
        print(f"   标签: {memory.tags}")
        print(f"   上下文: {memory.context}")
        print(f"   关键词: {memory.keywords}")

        # 语义搜索
        results = memory_system.search_agentic("连接测试", k=1)
        if results:
            print(f"✅ 语义搜索成功，找到 {len(results)} 条结果")
        else:
            print("⚠️ 语义搜索未返回结果（可能正常，记忆数量较少）")

        # 清理
        memory_system.delete(memory_id)
        print("✅ 测试记忆已清理")

    except Exception as e:
        print(f"❌ AgenticMemorySystem 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 50)
    print(f"🎉 所有测试通过！OpenRouter 后端配置正常。")
    print(f"   推荐在项目中使用模型: {working_model}")
    print("=" * 50)
    return True


if __name__ == "__main__":
    test_openrouter_connection()
