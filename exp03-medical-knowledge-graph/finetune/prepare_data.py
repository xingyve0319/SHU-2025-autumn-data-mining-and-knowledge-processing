import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.setup import set_hf_mirrors
import json
from datasets import load_dataset

# 1. 设置镜像
set_hf_mirrors()

def convert_to_bio(data, label_map):
    """
    将 CMeEE 的 span 格式转换为 BERT 需要的 BIO 格式
    策略：处理嵌套实体时，优先保留最长的实体 (Longest Match)
    """
    formatted_data = []
    
    for item in data:
        text = item['text']
        entities = item.get('entities', [])
        
        # 初始化标签，全是 'O'
        labels = ['O'] * len(text)
        
        # 按实体长度降序排列，优先处理长实体，避免嵌套冲突
        entities.sort(key=lambda x: x['end_idx'] - x['start_idx'], reverse=True)
        
        # 记录已被标记的位置掩码
        mask = [False] * len(text)
        
        for entity in entities:
            start = entity['start_idx']
            end = entity['end_idx']
            e_type = entity['type']
            
            # 映射标签 (如 dis -> disease)
            # 如果不在我们需要的列表里，就跳过
            if e_type not in label_map:
                continue
                
            mapped_type = label_map[e_type]
            
            # 检查是否有重叠（简化处理：如果有重叠则跳过短的）
            if any(mask[start:end]):
                continue
                
            # 标记 BIO
            labels[start] = f"B-{mapped_type}"
            for i in range(start + 1, end):
                labels[i] = f"I-{mapped_type}"
                
            # 更新掩码
            for i in range(start, end):
                mask[i] = True
                
        formatted_data.append({
            "tokens": list(text), # 按字分词
            "ner_tags": labels
        })
    
    return formatted_data

def main():
    print("🚀 正在下载 CMeEE-V2 数据集...")
    # 加载数据集 (自动使用 cache)
    dataset = load_dataset("Aunderline/CMeEE-V2", trust_remote_code=True)
    
    # 定义我们要提取的标签映射
    # CMeEE原始标签: dis(疾病), sym(症状), dru(药物), pro(操作), equ(设备), ite(检查)
    # 映射到你的作业需求
    label_map = {
        "dis": "disease",
        "sym": "symptom",
        "dru": "drug",
        "pro": "check", # 医疗程序/手术 -> 检查
        "ite": "check", # 检查项目 -> 检查
        # "equ": "drug", # 设备可选，暂时忽略
        # "bod": "body"  # 身体部位，你的作业好像没要求
    }
    
    output_dir = "data/training_data"
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'test']: # CMeEE-V2 的 validation 其实是 dev
        print(f"正在处理 {split} 集...")
        # 注意：hf dataset 的 split 名字可能叫 'train', 'test', 'validation'
        ds_split = 'validation' if split == 'test' else 'train'
        if ds_split not in dataset:
            print(f"跳过 {ds_split} (不存在)")
            continue
            
        processed_data = convert_to_bio(dataset[ds_split], label_map)
        
        output_file = os.path.join(output_dir, f"{split}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in processed_data:
                json.dump(line, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✅ 已保存 {len(processed_data)} 条数据到 {output_file}")

if __name__ == "__main__":
    main()