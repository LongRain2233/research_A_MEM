"""
快速冒烟测试脚本 - 验证 A-MEM 项目能否正常运行
测试内容:
  1. 依赖导入检查
  2. 数据集加载
  3. OpenRouter LLM API 连接
  4. 记忆添加与检索
"""
import sys
import os

# 确保从项目目录加载 .env
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("A-MEM 快速冒烟测试")
print("=" * 60)

# ---- 1. 依赖导入检查 ----
print("\n[1/4] 检查依赖导入...")
errors = []
try:
    import numpy as np; print(f"  [OK] numpy {np.__version__}")
except ImportError as e: errors.append(f"numpy: {e}")

try:
    import torch; print(f"  [OK] torch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
except ImportError as e: errors.append(f"torch: {e}")

try:
    from sentence_transformers import SentenceTransformer; print("  [OK] sentence-transformers")
except ImportError as e: errors.append(f"sentence-transformers: {e}")

try:
    from sklearn.metrics.pairwise import cosine_similarity; print("  [OK] scikit-learn")
except ImportError as e: errors.append(f"scikit-learn: {e}")

try:
    import nltk; print(f"  [OK] nltk {nltk.__version__}")
except ImportError as e: errors.append(f"nltk: {e}")

try:
    from rank_bm25 import BM25Okapi; print("  [OK] rank-bm25")
except ImportError as e: errors.append(f"rank-bm25: {e}")

try:
    from openai import OpenAI; print("  [OK] openai")
except ImportError as e: errors.append(f"openai: {e}")

try:
    from litellm import completion; print("  [OK] litellm")
except ImportError as e: errors.append(f"litellm: {e}")

try:
    from rouge_score import rouge_scorer; print("  [OK] rouge-score")
except ImportError as e: errors.append(f"rouge-score: {e}")

try:
    from bert_score import score as bert_score; print("  [OK] bert-score")
except ImportError as e: errors.append(f"bert-score: {e}")

try:
    from memory_layer import LLMController, AgenticMemorySystem; print("  [OK] memory_layer (项目核心模块)")
except ImportError as e: errors.append(f"memory_layer: {e}")

try:
    from load_dataset import load_locomo_dataset; print("  [OK] load_dataset (数据加载模块)")
except ImportError as e: errors.append(f"load_dataset: {e}")

try:
    from utils import calculate_metrics, aggregate_metrics; print("  [OK] utils (评估工具模块)")
except ImportError as e: errors.append(f"utils: {e}")

if errors:
    print(f"\n  [FAIL] 以下依赖导入失败:")
    for err in errors:
        print(f"    - {err}")
    print("\n请先修复依赖问题再继续测试。")
    sys.exit(1)
else:
    print("\n  [OK] 所有依赖导入成功!")

# ---- 2. 数据集加载 ----
print("\n[2/4] 测试数据集加载...")
try:
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "locomo10.json")
    samples = load_locomo_dataset(dataset_path)
    print(f"  [OK] 成功加载 {len(samples)} 个样本")
    # 显示第一个样本的基本信息
    if samples:
        s = samples[0]
        print(f"    - 样本0: {len(s.qa)} 个QA, {len(s.conversation.sessions)} 个会话")
except Exception as e:
    print(f"  [FAIL] 数据集加载失败: {e}")
    sys.exit(1)

# ---- 3. LLM API 连接测试 ----
print("\n[3/4] 测试 OpenRouter LLM API 连接...")
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("  [FAIL] 未找到 OPENROUTER_API_KEY 环境变量，请检查 .env 文件")
    sys.exit(1)
else:
    print(f"  [OK] 找到 API Key: {api_key[:20]}...")

try:
    llm_controller = LLMController(
        backend="openrouter",
        model="openai/gpt-4o-mini",  # 使用便宜的模型做测试
        api_key=api_key
    )
    # 做一个简单的 completion 测试
    response = llm_controller.llm.get_completion(
        "Say hello in one word.",
        response_format={"type": "json_schema", "json_schema": {
            "name": "response",
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"}
                },
                "required": ["answer"],
                "additionalProperties": False
            },
            "strict": True
        }}
    )
    print(f"  [OK] LLM API 连接成功! 响应: {response[:100]}")
except Exception as e:
    print(f"  [FAIL] LLM API 连接失败: {e}")
    print("  提示: 请检查 OPENROUTER_API_KEY 是否有效，以及网络连接是否正常")
    sys.exit(1)

# ---- 4. 记忆系统测试 ----
print("\n[4/4] 测试记忆添加与检索...")
try:
    memory_system = AgenticMemorySystem(
        model_name='all-MiniLM-L6-v2',
        llm_backend='openrouter',
        llm_model='openai/gpt-4o-mini',
        api_key=api_key
    )
    print("  [OK] AgenticMemorySystem 初始化成功")
    
    # 添加一条测试记忆
    print("  >> 正在添加测试记忆 (这需要 LLM 分析，可能需要几秒)...")
    mem_id = memory_system.add_note("Speaker A says: I just got a new job at Google as a software engineer.")
    print(f"  [OK] 记忆添加成功, ID: {mem_id}")
    
    # 再添加一条
    print("  >> 正在添加第二条测试记忆...")
    mem_id2 = memory_system.add_note("Speaker B says: That's amazing! I heard Google has great benefits and work culture.")
    print(f"  [OK] 第二条记忆添加成功, ID: {mem_id2}")
    
    # 检索测试
    print("  >> 正在测试记忆检索...")
    results = memory_system.find_related_memories_raw("What job did Speaker A get?", k=2)
    print(f"  [OK] 记忆检索成功!")
    print(f"    检索结果 (前200字符): {str(results)[:200]}")
    
except Exception as e:
    import traceback
    print(f"  [FAIL] 记忆系统测试失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# ---- 总结 ----
print("\n" + "=" * 60)
print("ALL TESTS PASSED! A-MEM can run normally.")
print("=" * 60)
print("\n要运行完整评估，请执行:")
print(f'  D:\\Anaconda3\\envs\\A-mem\\python.exe test_advanced.py --backend openrouter --model openai/gpt-4o-mini --dataset data/locomo10.json --ratio 0.1')
print()
