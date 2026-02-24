"""轻量测试脚本：验证 OpenRouter 是否能正常调用（不依赖 memory_layer）"""
import os
import json
from dotenv import load_dotenv
from litellm import completion

# 加载 .env 文件
load_dotenv()

def test_openrouter():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[失败] 未找到 OPENROUTER_API_KEY，请检查 .env 文件")
        return
    print(f"[OK] 已读取 API Key: {api_key[:20]}...")

    # 使用一个便宜的模型做测试
    model = "openrouter/google/gemini-2.0-flash-001"
    print(f"\n正在测试模型: {model}")

    prompt = "What is 1 + 1? Answer in one word."
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string"
                    }
                },
                "required": ["answer"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

    print(f"发送测试请求: \"{prompt}\"")
    try:
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
            temperature=0.1,
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1"
        )
        result = response.choices[0].message.content
        print(f"[OK] 收到原始响应: {result}")

        parsed = json.loads(result)
        print(f"[OK] JSON 解析成功: {parsed}")
        print(f"\n===== 测试通过！OpenRouter 调用正常 =====")
    except Exception as e:
        print(f"[失败] 调用出错: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_openrouter()
