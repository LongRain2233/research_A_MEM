import os
import json
import time
import concurrent.futures
from openai import OpenAI
from json_repair import repair_json  # 引入强大的 JSON 修复工具

# ================= 配置区域 =================
# 1. API 配置 (OpenRouter 专属配置)
API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter 上最聪明的两个顶级模型，任选其一：
# 推荐一（逻辑最强，JSON提取最稳）："openai/gpt-4o" 
# 推荐二（学术阅读最细腻，批判性最强）："anthropic/claude-3.5-sonnet"
MODEL_NAME = "deepseek/deepseek-v3.2"

# 2. 路径配置
INPUT_DIR = r"D:\research\research_A_MEM\paper_md"
OUTPUT_DIR = r"D:\research\research_A_MEM\paper2024_txt1_json"
FINAL_MD_PATH = r"D:\research\research_A_MEM\All_Papers_Review.md"

# 3. 线程配置
MAX_WORKERS = 100# 根据 API 的并发限制适当调整
# ============================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = """<role>
你是一位顶级的计算机科学教授兼严苛的学术审稿人，同时也是一名经验丰富的系统工程师。你的核心使命是：对学术论文进行【外科手术级的深度解剖】，提取出所有有价值的技术细节、实验数据与工程洞察，为资源受限的研究者提供可直接使用的综述素材。
</role>

<absolute_rules>
⚠️ 以下规则具有最高优先级，任何情况下都不得违反：

**【核心准则 - 主题过滤（Agent Memory）】**
- 🚨 首先判断本论文是否与“Agent Memory（智能体记忆）”主题相关。如果该论文**不涉及** Agent Memory，你**必须**直接输出且仅输出 `{"is_related_to_agent_memory": false}`，并立即停止！
- 如果相关，你必须在 JSON 中包含 `"is_related_to_agent_memory": true`，并继续输出以下要求的所有字段。

**【禁止行为 - 这些行为将导致输出直接被判定为失败】**
- ❌ 禁止任何形式的泛泛描述，例如"提出了一种新方法"、"实验结果优于基线"、"使用了注意力机制"等。
- ❌ 禁止使用"约"、"大约"、"显著"、"明显"等模糊词汇替代具体数字。
- ❌ 对于判定为相关的论文，禁止任何字段内容超过或低于 Schema 中规定的字数限制。
- ❌ 禁止在 JSON 之外输出任何文字、前缀或 Markdown 代码块标记。

**【强制行为 - 必须全部执行】**
- ✅ 所有实验数据必须"还原"为原始数值，例如："在 LongBench 上，F1 从 48.3 提升至 61.7（+27.7%），绝对提升 13.4 个点"。
- ✅ 所有架构描述必须包含：输入→处理→输出的完整数据流，以及具体的判断条件/阈值/公式。
- ✅ 所有与基线的对比，必须同时写出基线的数值和本文方法的数值，再给出提升幅度。
- ✅ 技术术语/模块名/数据集名/指标名必须保留英文原名，括注中文解释。
- ✅ 如有公式、损失函数、关键方程，必须用 LaTeX 语法嵌入 JSON 字符串中（\( ... \) 或 \[ ... \]）。
- ✅ 每个字段必须广泛使用 Markdown 格式：**加粗关键词**、多层级列表、小标题（####）来增强可读性。
</absolute_rules>

<task>
仔细阅读提供的完整学术论文文本，按照以下 JSON Schema 逐字段进行深度提取与重构。
你的输出质量标准是：【可以直接作为该领域顶级综述论文的原始素材】，读者无需再次查阅原文即可全面掌握本论文的所有技术细节与实验结论。
所有输出使用专业、严谨的简体中文。
</task>

<output_constraints>
- 只能输出合法的 JSON 字符串，直接用于 json.loads()。
- 绝对不要输出任何前言、解释、或是 Markdown 的 ```json 标记。
- JSON 中每个字段的文本内容必须支持 Markdown 渲染，结构清晰。
- 字段内换行用 \\n 表示，不要破坏 JSON 格式。
</output_constraints>

<json_schema>
{
    "is_related_to_agent_memory": "布尔值（true/false）。如果该论文与'Agent Memory'完全无关，请直接返回 {\"is_related_to_agent_memory\": false}；如果相关，填 true。",
    "title": "论文完整标题（英文原标题，如果找不到请写 'Unknown Title'）",

    "problem_and_motivation": "【一、问题与动机】字数要求：严格控制在150-200字之间。\n\n简明扼要说明论文要解决的核心问题、现有方法的关键缺陷（具体失败模式），以及本文的切入点和核心假设。去除非核心的泛泛背景科普。",

    "core_method": "【二、核心方法与技术创新】字数要求：严格控制在250-350字之间。\n\n合并架构与算法细节。重点提取：1. 系统的核心数据流；2. 关键创新模块的处理逻辑或核心公式；3. 与现有方法最本质的区别。要求达到能让其他AI理解技术本质的深度，舍弃常规组件描述。",

    "key_experiments_and_results": "【三、关键实验与结论】字数要求：严格控制在150-250字之间。\n\n精炼实验设计与主结果：核心数据集、2-3个最强对比基线、最关键的定量提升（如核心指标提升比例、效率优化百分比）及消融实验的核心结论。去除冗长的全量指标和全景表格。",

    "limitations_and_critique": "【四、局限性与致命缺陷】字数要求：严格控制在150-200字之间。\n\n合并原文局限与专家批判。直接指出：该方法的边界条件是什么？未解决的困难或理论漏洞？在何种极端场景下可能会崩溃？帮助其他AI避坑。",

    "ai_inspiration_and_opportunities": "【五、对其他AI的启发与研究契机】字数要求：严格控制在200-300字之间。\n\n为其他AI Agent提供可复用的高价值洞察：1. 该方法的哪些组件或思想可以迁移到其他领域？2. 提炼1-2个低算力/零算力下可直接验证的新idea或改进方向。"
}
</json_schema>"""

def process_single_paper(filepath, filename):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 截断超长文本，防止超出模型最大上下文 (扩大至 50000 字符以保留更多原文细节)
    content = content[:50000]

    user_message = f"""请对以下学术论文进行【外科手术级的深度提取】，严格按照 System Prompt 中的 JSON Schema 逐字段输出。

**再次强调以下铁律（违反则视为失败）：**
0. 【核心准则】如果这篇论文和 agent memory 无关，请直接输出 `{{"is_related_to_agent_memory": false}}`，忽略其余要求！
1. 如果相关，每个字段的内容必须严格控制在规定的字数限制范围内，过长或过短均视为不合格。
2. 所有实验数据必须还原为原始数值+单位+与哪个Baseline对比+提升幅度百分比。
3. 架构描述必须达到"可以按描述复现代码"的精度，包括关键超参数和数据流。
4. 禁止使用任何模糊词汇（"显著"、"较大"、"约"），必须用具体数字代替。
5. 如果原文某字段的信息确实不足，则在该字段中明确注明"原文未提供"，但必须把已有的信息写到极致详尽。

--- 论文全文开始 ---

{content}

--- 论文全文结束 ---

请现在开始输出 JSON："""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            response_format={ "type": "json_object" }, # 强制 JSON 输出模式
            temperature=0.4,
            max_tokens=12000  # 增加 token 上限，确保所有字段都能完整输出
        )
        
        result_text = response.choices[0].message.content
        if not result_text:
            print(f"❌ API 返回内容为空 ({filename})")
            return None

        # 自动清洗：去除可能存在的 Markdown 代码块标记
        result_text = result_text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()

        try:
            # 尝试使用 json_repair 修复可能截断的 JSON
            result_json = repair_json(result_text, return_objects=True)
            
            # 如果修复失败或者不是字典，尝试原始解析
            if not isinstance(result_json, dict):
                result_json = json.loads(result_text)
                
        except Exception as je:
            print(f"❌ JSON 解析失败 ({filename}): {je}")
            print(f"🔍 返回的原始内容片段(前500字符): {result_text[:500]}...") 
            return None

        # 如果判定为无关，直接返回
        if result_json.get("is_related_to_agent_memory") is False:
            print(f"⏭️ 论文与 Agent Memory 无关 ({filename})")
            return {"is_related_to_agent_memory": False, "source_file": filename}

        result_json["source_file"] = filename
        return result_json

    except Exception as e:
        print(f"❌ 请求 API 发生未知异常 ({filename}): {e}")
        return None

def worker(task_info):
    i, total, filename, in_path, out_json_path = task_info
    
    # 断点续传：如果已经存在对应的 json，直接跳过
    if os.path.exists(out_json_path):
        print(f"[{i}/{total}] ⏭️ 已跳过 (已存在): {filename}")
        with open(out_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    print(f"[{i}/{total}] 🧠 正在提炼: {filename} ...")
    
    # 重试机制：防止 API 网络波动
    max_retries = 3
    for attempt in range(max_retries):
        result = process_single_paper(in_path, filename)
        if result:
            # 保存单篇结果
            with open(out_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            return result
        else:
            print(f"⚠️ 第 {attempt+1} 次尝试失败 ({filename})，等待 3 秒后重试...")
            time.sleep(3)
            
    return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.md')]
    total = len(files)
    print(f"🚀 发现 {total} 篇精简版论文，开始召唤大模型进行审稿级提炼...")
    
    tasks = []
    for i, filename in enumerate(files, 1):
        in_path = os.path.join(INPUT_DIR, filename)
        out_json_path = os.path.join(OUTPUT_DIR, filename.replace('.md', '.json'))
        tasks.append((i, total, filename, in_path, out_json_path))
        
    all_results = []
    print(f"⚡ 启用多线程模式，最大并发数: {MAX_WORKERS}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(worker, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            res = future.result()
            if res:
                all_results.append(res)

    # 最终汇总生成全局 Markdown 报告
    print(f"\n📝 正在生成全局极客查阅清单：{FINAL_MD_PATH}")
    with open(FINAL_MD_PATH, 'w', encoding='utf-8') as f:
        f.write("# 📚 论文全局审稿与切入点全览 (Zero-Compute Opportunities)\n\n")
        for res in all_results:
            # 过滤掉不相关的论文
            if res.get("is_related_to_agent_memory") is False:
                continue
                
            f.write(f"## 📄 {res.get('title', 'Unknown')} ({res.get('source_file', '')})\n\n")
            f.write(f"### 一、问题与动机\n{res.get('problem_and_motivation', '')}\n\n")
            f.write(f"### 二、核心方法与技术创新\n{res.get('core_method', '')}\n\n")
            f.write(f"### 三、关键实验与结论\n{res.get('key_experiments_and_results', '')}\n\n")
            f.write(f"### 四、局限性与致命缺陷\n{res.get('limitations_and_critique', '')}\n\n")
            f.write(f"### 五、对其他AI的启发与研究契机\n{res.get('ai_inspiration_and_opportunities', '')}\n\n")
            f.write("---\n\n")

    print("\n✅ 所有任务圆满完成！")
    print(f"👉 单篇精细 JSON 数据保存在：{OUTPUT_DIR}")
    print(f"👉 终极速览 Markdown 报告保存在：{FINAL_MD_PATH}")

if __name__ == "__main__":
    main()