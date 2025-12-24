import sys
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)

# -------------------------
# 1. 路径设置 (关键步骤)
# -------------------------
# 将项目根目录添加到 python path，这样才能导入 src 包
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 导入你的自定义工具
from src.utils.setup import set_hf_mirrors, set_seed, setup_logging

# -------------------------
# 2. 全局配置
# -------------------------
# 指定只用一张卡训练（BERT-Base 在单张 2080Ti 上跑得很快，多卡配置复杂且收益不高）
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

MODEL_NAME = "google-bert/bert-base-chinese"
OUTPUT_DIR = os.path.join(project_root, "models", "my_medical_bert") # 绝对路径更安全

# 标签定义 (必须与 prepare_data.py 一致)
LABEL_LIST = [
    "O", 
    "B-disease", "I-disease", 
    "B-symptom", "I-symptom", 
    "B-drug", "I-drug", 
    "B-check", "I-check"
]
id2label = {i: label for i, label in enumerate(LABEL_LIST)}
label2id = {label: i for i, label in enumerate(LABEL_LIST)}

def main():
    # -------------------------
    # 3. 初始化环境
    # -------------------------
    # 设置镜像
    set_hf_mirrors()
    # 固定随机种子
    set_seed(42)
    # 设置日志
    log_dir = os.path.join(project_root, "logs")
    logger = setup_logging(save_dir=log_dir, model_name="bert_finetune")
    
    logger.info(f"🚀 开始微调任务")
    logger.info(f"使用模型: {MODEL_NAME}")
    logger.info(f"输出目录: {OUTPUT_DIR}")

    # -------------------------
    # 4. 加载分词器和模型
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id
    )

    # -------------------------
    # 5. 数据处理
    # -------------------------
    # 定义处理函数 (放在这里是为了能直接使用 tokenizer 变量)
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], 
            truncation=True, 
            is_split_into_words=True, 
            max_length=512
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100) # 忽略特殊 token (CLS, SEP)
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id.get(label[word_idx], 0)) # 默认为O
                else:
                    # 对于中文，同一个词的后续subword也标记为相同标签
                    label_ids.append(label2id.get(label[word_idx], 0))
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # 加载数据 (使用绝对路径避免找不到文件)
    data_dir = os.path.join(project_root, "data", "training_data")
    train_file = os.path.join(data_dir, "train.json")
    test_file = os.path.join(data_dir, "test.json")
    
    logger.info(f"加载训练数据: {train_file}")
    
    # 检查数据文件是否存在
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"找不到训练数据: {train_file}。请先运行 prepare_data.py")

    raw_train_dataset = load_dataset('json', data_files=train_file, split='train')
    raw_eval_dataset = load_dataset('json', data_files=test_file, split='train')
    
    logger.info("正在进行数据预处理...")
    tokenized_train = raw_train_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_eval = raw_eval_dataset.map(tokenize_and_align_labels, batched=True)

    # -------------------------
    # 6. 训练配置
    # -------------------------
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",   
        eval_steps=100,           # 每100步验证一次
        save_strategy="steps",    # 保持 steps
        save_steps=100,           # 每100步保存一次
        learning_rate=2e-5,       
        per_device_train_batch_size=32, # 2080Ti 11G 显存充裕，可以直接开32
        per_device_eval_batch_size=32,
        num_train_epochs=5,       
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        save_total_limit=2,       # 最多保留2个模型checkpoint，防止硬盘爆满
        fp16=True,                # 开启混合精度，速度快且省显存
        dataloader_num_workers=4, # 加速数据加载
        report_to="wandb",
        run_name="medical-bert"
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # -------------------------
    # 7. 开始训练
    # -------------------------
    logger.info("开始训练...")
    trainer.train()
    
    logger.info(f"✅ 训练完成！模型已保存到: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()