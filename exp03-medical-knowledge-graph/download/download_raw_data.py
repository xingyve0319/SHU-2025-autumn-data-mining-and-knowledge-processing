import os
import json
import sys
from datasets import load_dataset

# 1. 设置镜像 (利用你现有的工具)
try:
    # 尝试导入项目里的 setup 工具
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from src.utils.setup import set_hf_mirrors
    set_hf_mirrors()
except ImportError:
    # 如果导入失败，手动设置
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def main():
    print("🚀 正在从 Hugging Face 下载 'ncbi/Open-Patients' 数据集...")
    
    try:
        # --- 核心代码：加载数据集 ---
        # trust_remote_code=True 是为了防止某些自定义数据集报错
        ds = load_dataset("ncbi/Open-Patients", trust_remote_code=True)
        
        # 2. 准备输出目录
        output_dir = "data/raw"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "Open-Patients.jsonl")
        
        # 3. 选择数据集的一个切分 (通常用 'train')
        # 如果没有 train，就用第一个可用的切分
        split_name = 'train'
        if split_name not in ds:
            split_name = list(ds.keys())[0]
            
        print(f"📦 正在处理 {split_name} 集 (共 {len(ds[split_name])} 条)...")
        
        # 4. 转换为 JSONL 格式并保存
        # JSONL 就是每一行都是一个独立的 JSON 对象
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in ds[split_name]:
                # 这里的 item 是一个字典，包含 patient_id, text 等字段
                json.dump(item, f, ensure_ascii=False)
                f.write('\n') # 换行
                
        print(f"✅ 数据下载并转换成功！")
        print(f"📂 文件已保存到: {output_file}")
        print("➡️ 现在你可以重新运行 ./run_extract_entity.sh 了")
        
    except Exception as e:
        print(f"❌ 下载或保存失败: {e}")
        print("请检查网络，或确认数据集名称是否正确。")

if __name__ == "__main__":
    main()