import json

filename = '/share_data/data1/fanshengda/DEvo/ckpts/models/qwen3-4b-difficulty_aware_solver_1224/rollouts.jsonl'
empty_count = 0
total_count = 0

try:
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            total_count += 1
            data = json.loads(line)

            # 根据示例结构：ground_truth 内部还有一个 ground_truth 键
            gt_obj = data.get('ground_truth', {})
            if isinstance(gt_obj, dict):
                # 检查嵌套的 ground_truth 是否为空字符串
                if gt_obj.get('ground_truth') == "":
                    empty_count += 1
            elif gt_obj == "":
                # 如果结构是 "ground_truth": ""
                empty_count += 1

    if total_count > 0:
        proportion = empty_count / total_count
        print(f"总行数: {total_count}")
        print(f"空 ground_truth 行数: {empty_count}")
        print(f"占比: {proportion:.2%} ({proportion})")
    else:
        print("文件为空或未找到有效数据。")

except FileNotFoundError:
    print(f"找不到文件: {filename}")
except Exception as e:
    print(f"处理出错: {e}")