import json

with open('A-mem-ollma/A-mem/data/locomo10.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('output.txt', 'w', encoding='utf-8') as out:
    out.write('=== 数据集概览 ===\n')
    out.write(f'总条目数: {len(data)}\n\n')

    out.write('=== 字段结构 ===\n')
    for key in data[0].keys():
        val = data[0][key]
        type_name = type(val).__name__
        if isinstance(val, list):
            out.write(f'  - {key}: {type_name} (长度: {len(val)})\n')
        else:
            out.write(f'  - {key}: {type_name}\n')

    out.write('\n=== qa 字段详情 ===\n')
    qa_len = len(data[0]['qa'])
    out.write(f'每个条目包含 {qa_len} 个问答对\n')
    out.write('每个问答对包含以下字段:\n')
    for qkey in data[0]['qa'][0].keys():
        out.write(f'  - {qkey}\n')

    out.write('\n=== category 类别说明 ===\n')
    categories = {}
    for item in data:
        for qa in item['qa']:
            cat = qa['category']
            categories[cat] = categories.get(cat, 0) + 1
    out.write(f'类别分布: {categories}\n')
    out.write('含义: 1=显式事实, 2=隐式事实, 3=推理, 4=时间理解, 5=其他\n')

    out.write('\n=== 查看其他字段 ===\n')
    out.write(f'\n- sample_id: {data[0]["sample_id"]}\n')
    out.write(f'\n- conversation 字段的键: {list(data[0]["conversation"].keys())}\n')
    out.write(f'\n- event_summary 字段的键: {list(data[0]["event_summary"].keys())}\n')
    out.write(f'\n- observation 字段的键: {list(data[0]["observation"].keys())}\n')
    out.write(f'\n- session_summary 字段的键: {list(data[0]["session_summary"].keys())}\n')

    out.write('\n=== conversation 内容示例 ===\n')
    conv = data[0]['conversation']
    for k, v in list(conv.items())[:5]:
        out.write(f'  {k}: {str(v)[:150]}...\n')

print('输出完成')
