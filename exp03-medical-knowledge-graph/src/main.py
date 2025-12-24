import os
import torch
from src.models.entity_extractor import MedicalEntityExtractor
from src.utils.data_processor import DataProcessor
from src.config.config import config
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from src.models.bert_extractor import BertEntityExtractor 
from src.utils.data_processor import DataProcessor
from src.utils.translation import translate_to_chinese
from src.config.config import config
def init_models():
    """
    初始化双模型架构：
    1. Qwen (翻译)
    2. BERT (抽取)
    """
    print("正在初始化双模型架构...")

    # ----------------------------------------
    # 1. 初始化 Qwen (仅用于翻译)
    # ----------------------------------------
    print("1. 加载 Qwen2.5-3B (用于翻译)...")
    trans_model_name = "Qwen/Qwen2.5-3B"
    
    # 你的显卡很好，可以用 fp16 (不用量化也行，但 bitsandbytes 更省显存)
    # 如果想用量化，确保装了 bitsandbytes 并加 load_in_4bit=True
    trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name, trust_remote_code=True)
    trans_model = AutoModelForCausalLM.from_pretrained(
        trans_model_name,
        device_map="cuda:0", # 放在第一张卡
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    # ----------------------------------------
    # 2. 初始化 BERT (用于抽取)
    # ----------------------------------------
    print("2. 加载微调后的 BERT (用于抽取)...")
    # 你的模型路径
    bert_path = "./models/my_medical_bert" 
    bert_device = "cuda:1" 
    
    extractor = BertEntityExtractor(model_dir=bert_path, device=bert_device)
    
    return extractor, trans_model, trans_tokenizer

def merge_entities(entities1, entities2):
    """
    合并两个实体字典，去除重复项
    """
    if not entities1:
        return entities2
    if not entities2:
        return entities1
    
    keys = config.entity_keys
    result = {}
    
    for k in keys:
        l1 = entities1.get(k, [])
        l2 = entities2.get(k, [])
        # 确保是列表且去重
        result[k] = list(set((l1 if l1 else []) + (l2 if l2 else [])))
    return result

def main():
    # 初始化所有模型
    extractor, model, tokenizer = init_models()
    
    # 设置输入输出路径
    input_file = "data/raw/Open-Patients.jsonl"  # 原始文章数据
    output_dir = "data/processed"          # 处理后的数据目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载原始数据
    print("加载原始数据...")
    articles = DataProcessor.load_json_data(input_file)
    if not articles:
        print("没有找到原始数据，请确保数据文件存在")
        return
    
    # 处理每篇文章
    processed_articles = []
    try:
        for i, article in enumerate(articles):
            # 测试
            # if i >100:
            #     break
            print(f"\n{'='*50}")
            print(f"处理第 {i+1}/{len(articles)} 篇文章")
            print(f"{'='*50}")
            
            # 打印原文标题
            print("\n原文:")
            print(article["description"])
            
            # 翻译标题
            print("\n翻译...")
            translated = translate_to_chinese(
                article["description"],
                model,
                tokenizer,
            )
            print(f"翻译结果: {translated}")
            
            
            # 从中提取实体
            print("\n从翻译中提取医学实体...")
            entities = extractor.extract_entities(translated)
            
            if entities:
                for key, values in entities.items():
                    if values:
                        print(f"- {key}: {', '.join(values)}")
                
                # 合并结果
                processed_article = {
                    "id": article["_id"],
                    "translated": translated,
                    **entities
                }
                processed_articles.append(processed_article)
                print(f"\n成功处理文章: {translated}")
                print(f"提取的实体数量: 症状({len(entities['symptoms'])}), 疾病({len(entities['diseases'])}), 药物({len(entities['drugs'])}), 检查({len(entities['checks'])})")
            else:
                print(f"\n跳过文章: {translated} (未提取到实体)")
    except KeyboardInterrupt:
        print("\n检测到手动中断 (Ctrl+C)...")

    # 保存处理后的数据
    print("\n保存处理后的数据...")
    json_output = os.path.join(output_dir, "processed_articles.json")
    DataProcessor.save_json_data(processed_articles, json_output)
    
    # 保存为Neo4j格式
    print("保存为Neo4j格式...")
    neo4j_output = os.path.join(output_dir, "neo4j")
    DataProcessor.save_to_neo4j_format(processed_articles, neo4j_output)
    
    print("\n处理完成！")
    print(f"- 处理文章数: {len(processed_articles)}")
    print(f"- JSON输出: {json_output}")
    print(f"- Neo4j输出: {neo4j_output}")

if __name__ == "__main__":
    main() 