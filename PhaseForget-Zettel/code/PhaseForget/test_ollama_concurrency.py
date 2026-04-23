import time
import asyncio
import litellm
import logging
import sys
import subprocess

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("ollama_stress_test")

# Ollama 的 API 地址
API_BASE = "http://localhost:11434"
# 替换为你实际使用的 3B 模型名字
MODEL_NAME = "ollama/qwen2.5:3b"


def stop_ollama_model():
    """
    压测完成后卸载模型，尽快释放显存。
    """
    ollama_model_name = MODEL_NAME
    if ollama_model_name.startswith("ollama/"):
        ollama_model_name = ollama_model_name.split("/", 1)[1]

    try:
        result = subprocess.run(
            ["ollama", "stop", ollama_model_name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info(f"已停止模型并尝试释放显存: {ollama_model_name}")
        else:
            logger.warning(
                f"停止模型失败(退出码={result.returncode}): {ollama_model_name}; "
                f"stderr={result.stderr.strip()}"
            )
    except Exception as e:
        logger.warning(f"停止模型时出现异常: {e}")

async def call_ollama(task_id, semaphore):
    """
    带信号量限制的请求函数，防止瞬间撑爆系统，但依然保持高并发
    """
    async with semaphore:
        logger.info(f"[任务 {task_id}] 正在发送请求...")
        start_time = time.time()
        
        prompt = f"请详细描述一下中国的高铁发展史，字数不少于500字。这是任务{task_id}。"
        
        try:
            # 调用 litellm 的异步接口
            response = await litellm.acompletion(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                api_base=API_BASE,
                temperature=0.7,
                stream=False,
                max_tokens=700,
                timeout=600 # 压力测试下响应会变慢，增加到10分钟
            )
            
            content = response.choices[0].message.content
            end_time = time.time()
            cost_time = end_time - start_time
            
            logger.info(f">>> [任务 {task_id}] 完成！生成字数: {len(content)}, 耗时: {cost_time:.2f} 秒")
            return cost_time
            
        except Exception as e:
            logger.error(f">>> [任务 {task_id}] 失败: {e}")
            return 0

async def run_level(n_requests, max_concurrency):
    """运行一轮测试，返回 (吞吐量, 单任务平均耗时, 总壁钟时间)"""
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [call_ollama(i, semaphore) for i in range(1, n_requests + 1)]
    start_total = time.time()
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_total
    valid = [r for r in results if r > 0]
    if not valid:
        return 0.0, 0.0, total_time
    throughput = len(valid) / total_time
    avg_time = sum(valid) / len(valid)
    return throughput, avg_time, total_time


async def main(n_requests=150, max_concurrency=150):
    print(f"开始并发压力测试 (模型: {MODEL_NAME}, 请求总数: {n_requests})...")
    start_total = time.time()

    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [call_ollama(i, semaphore) for i in range(1, n_requests + 1)]
    results = await asyncio.gather(*tasks)

    total_time = time.time() - start_total
    valid_results = [r for r in results if r > 0]

    print(f"\n{'='*60}")
    print(f"并发压力测试结果摘要")
    print(f"{'='*60}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"成功请求数: {len(valid_results)} / {n_requests}")
    print(f"并发度上限: {max_concurrency}")

    if valid_results:
        sum_tasks = sum(valid_results)
        avg_time = sum_tasks / len(valid_results)
        throughput = len(valid_results) / total_time
        print(f"单任务平均耗时: {avg_time:.2f} 秒")
        print(f"系统总吞吐量: {throughput:.2f} 请求/秒")
        print(f"理论串行总耗时: {sum_tasks:.2f} 秒")
        print(f"实际加速比: {sum_tasks / total_time:.2f}x")
    print(f"{'='*60}")


async def sweep(levels, requests_per_level_fn):
    """
    依次测试每个并发度，每轮前重置模型保证公平。
    每轮先发 1 个预热请求（不计入统计），再正式测量。
    requests_per_level_fn: callable(level) -> int
    """
    sweep_results = []

    for level in levels:
        requests_per_level = requests_per_level_fn(level)
        print(f"\n{'─'*60}")
        print(f"[SWEEP] 并发度 = {level}，正式请求数 = {requests_per_level}")
        print(f"{'─'*60}")

        # 重置显存，确保每轮起点一致
        stop_ollama_model()
        await asyncio.sleep(1)

        # 预热：让模型完整加载进 GPU，冷启动时间不算入正式测量
        print(f"[SWEEP] 预热中（发送 1 个请求，不计入统计）...")
        warmup_sem = asyncio.Semaphore(1)
        await call_ollama(0, warmup_sem)

        # 正式测量
        print(f"[SWEEP] 开始正式测量...")
        throughput, avg_time, total_time = await run_level(requests_per_level, level)
        sweep_results.append((level, throughput, avg_time, total_time))
        print(
            f"[SWEEP] 并发度={level}: 吞吐量={throughput:.4f} req/s, "
            f"单任务均耗时={avg_time:.2f}s, 总壁钟={total_time:.2f}s"
        )

    # 汇总报告
    print(f"\n{'='*60}")
    print(f"SWEEP 吞吐量扫描结果汇总")
    print(f"{'='*60}")
    print(f"{'并发度':>6}  {'吞吐量(req/s)':>14}  {'单任务均耗时':>12}  {'总壁钟时间':>10}  {'vs并发1':>8}")
    base_tp = sweep_results[0][1] if sweep_results else 1.0
    for level, tp, avg_t, wall_t in sweep_results:
        ratio = tp / base_tp if base_tp > 0 else 0
        print(f"{level:>6}  {tp:>14.4f}  {avg_t:>10.2f}s  {wall_t:>8.2f}s  {ratio:>7.2f}x")

    if sweep_results:
        best = max(sweep_results, key=lambda x: x[1])
        print(f"\n>>> 推荐并发度: {best[0]}（吞吐量最高: {best[1]:.4f} req/s）")
    print(f"{'='*60}")

if __name__ == "__main__":
    litellm.drop_params = True

    # ── 普通模式参数 ────────────────────────────────────────────
    # 参数1: 请求总数；参数2: 并发度上限；可选参数:
    #   --pre-stop : 测试前先 stop 一次模型（清理上次残留）
    #   --no-stop  : 测试结束后不自动 stop 模型
    #
    # ── Sweep 模式（自动找最优并发度）─────────────────────────
    #   --sweep [levels]  : 扫描多个并发度，levels 用逗号分隔
    #                       默认扫描 1,2,3,4,5,6,8,10
    #   --rpl N           : 每个并发度正式测量的请求数（默认 = 该并发度 × 3）
    # 示例:
    #   python test_ollama_concurrency.py --sweep
    #   python test_ollama_concurrency.py --sweep 1,2,4,8 --rpl 6
    # ───────────────────────────────────────────────────────────

    extra_args = sys.argv[1:]
    is_sweep = "--sweep" in extra_args

    if is_sweep:
        # 解析 --sweep 后面紧跟的 levels（可选）
        sweep_levels = [1, 2, 3, 4, 5, 6, 8, 10]
        rpl = None  # requests per level，None 表示自动 = level × 3

        idx = extra_args.index("--sweep")
        if idx + 1 < len(extra_args) and not extra_args[idx + 1].startswith("--"):
            try:
                sweep_levels = [int(x) for x in extra_args[idx + 1].split(",")]
            except ValueError:
                pass

        if "--rpl" in extra_args:
            rpl_idx = extra_args.index("--rpl")
            if rpl_idx + 1 < len(extra_args):
                try:
                    rpl = int(extra_args[rpl_idx + 1])
                except ValueError:
                    pass

        # 每轮请求数：用户指定 > 自动（level × 3，至少 3）
        requests_per_level_fn = (lambda lvl: rpl) if rpl else (lambda lvl: max(lvl * 3, 3))

        print(f"开始 SWEEP 扫描（模型: {MODEL_NAME}）")
        print(f"测试并发度序列: {sweep_levels}")
        print(f"每轮请求数: {'固定 ' + str(rpl) if rpl else '并发度 × 3（自动）'}")
        try:
            asyncio.run(sweep(sweep_levels, requests_per_level_fn))
        except KeyboardInterrupt:
            print("\nSWEEP 被用户中止")
        finally:
            stop_ollama_model()

    else:
        num = 120
        concurrency = 120
        auto_pre_stop_model = False
        auto_stop_model = True

        flags = set(extra_args)
        positional = [a for a in extra_args if not a.startswith("--")]

        if len(positional) > 0:
            try:
                num = int(positional[0])
            except ValueError:
                pass
        if len(positional) > 1:
            try:
                concurrency = int(positional[1])
            except ValueError:
                pass
        if "--pre-stop" in flags:
            auto_pre_stop_model = True
        if "--no-stop" in flags:
            auto_stop_model = False

        if auto_pre_stop_model:
            logger.info("测试前执行模型卸载，清理历史 KV cache / 显存占用...")
            stop_ollama_model()

        try:
            asyncio.run(main(num, concurrency))
        except KeyboardInterrupt:
            print("\n测试被用户中止")
        finally:
            if auto_stop_model:
                stop_ollama_model()
