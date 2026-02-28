#!/usr/bin/env python3
"""
论文筛选脚本 - 多线程版本
基于 LLM 验证 Agent Memory 主题相关性

用法:
  1. 修改 config.yaml 填入 API Key
  2. python filter_papers.py
"""

import os
import sys
import json
import time
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

try:
    from openai import OpenAI
except ImportError:
    print("错误: pip install openai pyyaml")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("错误: pip install pyyaml")
    sys.exit(1)


# ============ 默认配置 ============
DEFAULT_CONFIG = {
    "api_key": "your-api-key-here",
    "base_url": "https://openrouter.ai/api/v1",
    "model": "anthropic/claude-3.5-sonnet",

    "input_dir": r"D:\research\research_A_MEM\paper2024_txt1_json",
    "backup_dir": r"D:\research\research_A_MEM\paper2024_txt1_json_backup",

    "dry_run": False,
    "backup": True,

    # 并发配置
    "max_workers": 5,  # 并发线程数，建议 3-10
    "delay_between_calls": 0.1,  # 每个请求间隔（秒）

    "max_retries": 3,
    "retry_delay": 2,
}


# ============ 全局变量 & 锁 ============
print_lock = Lock()
stats_lock = Lock()
results_lock = Lock()


SYSTEM_PROMPT = """<role>
你是一个严谨的学术主题分类专家，专门负责判断计算机科学论文是否与"Agent Memory（智能体记忆）"主题相关。你的判断标准严格且一致，只输出结构化的 JSON 结果。
</role>

<agent_memory_definition>
1. **显式记忆架构的提出与改进**：论文为智能体设计了外部的、可持久化的记忆存储结构，如：
   - 工作记忆（Working）、短期/长期记忆（Short/Long-term）
   - 语义记忆（Semantic）、情景记忆（Episodic）
   - 分层/图结构记忆系统（Hierarchical/Graph-based Memory）
   - 空间记忆/多模态记忆（Spatial/Multimodal Memory，常见于具身智能体）
   - 多智能体共享记忆/社会记忆（Shared/Collective Memory）

2. **核心记忆机制与管理策略的设计**：论文设计了智能体对记忆的动态操作和生命周期管理，如：
   - 记忆写入、读取、检索（Retrieval）、路由（Routing）
   - 记忆压缩（Compression）、总结提炼（Summarization）
   - 记忆遗忘机制（Forgetting）、衰减（Decay）、驱逐（Eviction）
   - 记忆反思（Reflection）、记忆的自我纠错与更新（Update/Consolidation）

3. **记忆驱动的 Agent 复杂能力应用**：论文强调外部记忆机制如何直接提升 Agent 的自主任务能力（而非纯对话）：
   - 多轮复杂任务/规划中的经验积累与复用（Experience/Skill reuse）
   - 跨会话的个性化 Agent 行为适应（基于记忆的用户认知演进，区别于传统推荐画像）
   - 复杂环境或持续学习（Continual Learning）中的历史状态跟踪

4. **Agent Memory 相关的资源与评估**：
   - 专门针对 Agent 记忆能力的评测基准（Benchmark）、数据集构建
   - Agent Memory 领域的综述文献（Survey）

【非 Agent Memory 的范畴 - 需严格排除】

以下主题即使涉及"Memory"或"History"概念，也不属于我们定义的 Agent Memory：
- 纯模型底层架构改进：如 Transformer 的 KV Cache 优化、长文本/无限上下文模型架构本身（没有显式的智能体记忆管理模块）。
- 传统对话与状态追踪：仅依靠拼接历史文本、滑动窗口的普通多轮聊天系统（缺少 Agent 的动作/规划/工具调用特性）。
- 纯信息检索与通用 RAG：仅通过向量数据库查询文档知识的通用 RAG 系统（没有智能体主动的写入、反思、遗忘等“记忆内化”过程）。
- 传统推荐系统或强化学习：如纯粹的电商用户画像推荐、传统 RL 的经验回放（Experience Replay池），且未结合 LLM Agent 框架。

【判断原则】
- 核心区分：必须是“智能体（Agent）”为了完成决策和动作而维护的“可读写外部系统”，而不仅仅是“模型（Model）”看到的长文本。
- 对于边界模糊或仅顺带提及记忆但非核心贡献的论文，优先返回 false。

输出 JSON: {"is_related": true/false, "reason": "判定理由"}"""


# ============ 评分阈值配置 ============
# 已移除：现在只根据 is_related 字段判断，is_related=false 才删除


def load_config() -> Dict[str, Any]:
    """加载配置"""
    config = DEFAULT_CONFIG.copy()

    config_file = Path(__file__).parent / "config.yaml"
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            file_config = yaml.safe_load(f)
            if file_config:
                config.update(file_config)
    else:
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(DEFAULT_CONFIG, f, allow_unicode=True, sort_keys=False)
        print(f"✓ 已创建配置文件: {config_file}")
        print("  请编辑填入 api_key 后重新运行")
        sys.exit(0)

    if config["api_key"] in ("", "your-api-key-here", None):
        print("错误: 请在 config.yaml 中设置 api_key")
        sys.exit(1)

    return config


def setup_logging():
    """设置日志"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"filter_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            # 控制台输出用自定义 handler 避免多线程混乱
        ]
    )

    return log_dir, timestamp, log_file


def console_print(msg: str):
    """线程安全的控制台输出"""
    with print_lock:
        print(msg)


def call_llm(client: OpenAI, model: str, content: str, max_retries: int, retry_delay: int, timeout: int = 60) -> Dict[str, Any]:
    """调用 LLM，带超时和重试"""
    user_prompt = f"请判断以下论文是否与 Agent Memory 相关：\n\n{content[:10000]}\n\n输出 JSON："

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"},
                timeout=timeout  # 超时设置
            )

            result = json.loads(response.choices[0].message.content)
            return {
                "is_related": result.get("is_related", True),
                "reason": result.get("reason", ""),
                "success": True
            }

        except Exception as e:
            wait = retry_delay * (2 ** attempt)
            if attempt < max_retries - 1:
                console_print(f"  → 请求失败，{wait}秒后重试... ({attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                return {"success": False, "error": str(e)}

    return {"success": False, "error": "Max retries"}


def extract_content(data: Dict) -> str:
    """提取论文内容"""
    parts = []
    if title := data.get("title"):
        parts.append(f"标题: {title}")
    if problem := data.get("problem_and_motivation"):
        parts.append(f"问题: {problem[:500]}")
    if method := data.get("core_method"):
        parts.append(f"方法: {method[:800]}")
    if not parts:
        parts.append(str(data)[:1500])
    return "\n\n".join(parts)


def process_file(task: Dict, config: Dict, client: OpenAI) -> Dict[str, Any]:
    """处理单个文件（在线程中运行）"""
    json_file = task["file"]
    idx = task["idx"]
    total = task["total"]

    try:
        # 加载
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查标记
        if not data.get("is_related_to_agent_memory", False):
            console_print(f"[{idx}/{total}] {json_file.name[:45]:45} → 跳过 (原标记不相关)")
            return {"file": json_file.name, "action": "skipped", "is_related": None, "reason": ""}

        # 延迟避免触发速率限制
        time.sleep(config["delay_between_calls"])

        # LLM 验证
        content = extract_content(data)
        result = call_llm(client, config["model"], content, config["max_retries"], config["retry_delay"])

        if not result["success"]:
            console_print(f"[{idx}/{total}] {json_file.name[:45]:45} → 错误: {result.get('error', 'Unknown')[:30]}")
            return {"file": json_file.name, "action": "error", "error": result.get("error")}

        is_related = result["is_related"]
        reason = result["reason"]

        # 决策 - 只看 is_related，false 就删除
        if not is_related:
            # 删除
            if config["dry_run"]:
                console_print(f"[{idx}/{total}] {json_file.name[:45]:45} → [将删除] is_related=false")
            else:
                if config["backup"]:
                    backup_path = Path(config["backup_dir"]) / json_file.name
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(json_file, backup_path)
                os.remove(json_file)
                console_print(f"[{idx}/{total}] {json_file.name[:45]:45} → 已删除 (is_related=false)")

            return {"file": json_file.name, "action": "deleted", "is_related": is_related, "reason": reason}
        else:
            # 保留
            console_print(f"[{idx}/{total}] {json_file.name[:45]:45} → 保留 (is_related=true)")
            return {"file": json_file.name, "action": "kept", "is_related": is_related, "reason": reason}

    except json.JSONDecodeError as e:
        console_print(f"[{idx}/{total}] {json_file.name[:45]:45} → JSON错误")
        return {"file": json_file.name, "action": "error", "error": f"JSON decode: {e}"}
    except Exception as e:
        console_print(f"[{idx}/{total}] {json_file.name[:45]:45} → 异常: {str(e)[:30]}")
        return {"file": json_file.name, "action": "error", "error": str(e)}


def main():
    config = load_config()

    log_dir, timestamp, log_file = setup_logging()
    input_dir = Path(config["input_dir"])

    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        sys.exit(1)

    # 获取文件列表
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        print("错误: 未找到 JSON 文件")
        sys.exit(1)

    # 预统计
    need_api_count = 0
    for f in json_files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                if data.get("is_related_to_agent_memory", False):
                    need_api_count += 1
        except:
            pass

    print(f"\n预统计: 共 {len(json_files)} 个文件")
    print(f"需验证: {need_api_count} 个 (is_related_to_agent_memory=true)")
    print(f"并发数: {config['max_workers']} 线程")
    print(f"预计时间: ~{need_api_count * 2 // config['max_workers']} 秒 (假设每请求2秒)\n")

    if config["dry_run"]:
        print("[Dry-run 模式] 只预览，不删除\n")
    else:
        confirm = input("确认继续? (y/n): ")
        if confirm.lower() != 'y':
            print("已取消")
            sys.exit(0)
    print()

    # 初始化 LLM 客户端
    try:
        client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
        print(f"模型: {config['model']}")
        print(f"日志: {log_file}\n")
    except Exception as e:
        print(f"LLM 初始化失败: {e}")
        sys.exit(1)

    # 准备任务
    tasks = [{"file": f, "idx": i, "total": len(json_files)} for i, f in enumerate(json_files, 1)]

    # 多线程处理
    start_time = time.time()
    results: List[Dict] = []

    print("开始处理...\n")
    with ThreadPoolExecutor(max_workers=config["max_workers"]) as executor:
        future_to_task = {executor.submit(process_file, task, config, client): task for task in tasks}

        for future in as_completed(future_to_task):
            result = future.result()
            results.append(result)

    duration = time.time() - start_time

    # 统计
    stats = {
        "total": len(json_files),
        "kept": sum(1 for r in results if r["action"] == "kept"),
        "deleted": sum(1 for r in results if r["action"] == "deleted"),
        "skipped": sum(1 for r in results if r["action"] == "skipped"),
        "error": sum(1 for r in results if r["action"] == "error"),
    }

    # 统计 is_related=false 的论文
    false_count = [
        r["file"] for r in results
        if r.get("is_related") == False
    ]
    stats["false_count"] = len(false_count)

    # 保存结果
    result_file = log_dir / f"results_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {k: v for k, v in config.items() if k != "api_key"},
            "stats": stats,
            "duration_seconds": duration,
            "results": results
        }, f, ensure_ascii=False, indent=2)

    # 输出摘要
    print(f"\n{'='*50}")
    print(f"完成! 耗时: {duration:.1f} 秒")
    print(f"总计: {stats['total']} | 保留: {stats['kept']} | 删除: {stats['deleted']}")
    if stats["false_count"] > 0:
        print(f"is_related=false: {stats['false_count']} 篇")
    print(f"跳过: {stats['skipped']} | 错误: {stats['error']}")
    print(f"结果: {result_file}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断")
        sys.exit(0)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
