import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

class BertEntityExtractor:
    def __init__(self, model_dir="./models/my_medical_bert", device="cuda:0"):
        """
        初始化微调后的 BERT 模型
        """
        print(f"正在加载微调后的 BERT 模型: {model_dir} ...")
        
        self.device = device
        # 1. 加载模型和分词器
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        except OSError:
            print(f"❌ 找不到模型文件: {model_dir}")
            print("请确保你已经运行了 finetune/train_bert.py 并且训练成功。")
            raise

        # 2. 创建推理流水线
        # aggregation_strategy="simple" 非常关键！
        # 它会自动把 "B-drug", "I-drug" 合并成一个完整的词 "阿司匹林"
        self.pipe = pipeline(
            "token-classification", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            device=device,
            aggregation_strategy="simple" 
        )
        print("✅ BERT 模型加载完成")

    def extract_entities(self, text):
        """
        输入: 中文文本
        输出: 你的 main.py 需要的字典格式
        """
        if not text or not isinstance(text, str):
            return {"symptoms": [], "diseases": [], "checks": [], "drugs": []}

        # 1. 运行推理
        # BERT 有长度限制 (512)，如果文本特别长，需要在这里切分（简单起见先截断）
        results = self.pipe(text[:510])
        
        # 2. 格式转换
        # 我们的训练标签是: disease, symptom, drug, check
        # main.py 需要的键是: diseases, symptoms, drugs, checks (复数)
        entities = {
            "symptoms": [],
            "diseases": [],
            "checks": [],
            "drugs": []
        }
        
        # 映射表 (模型标签 -> 输出Key)
        # 注意：这里的 key 必须和你 prepare_data.py 里定义的 label_map 对应
        tag_map = {
            "disease": "diseases",
            "symptom": "symptoms",
            "drug": "drugs",
            "check": "checks"
        }

        for item in results:
            # item 格式: {'entity_group': 'drug', 'score': 0.99, 'word': '阿司匹林', ...}
            label = item['entity_group']
            raw_word = item['word']

            word = raw_word.replace(" ", "")

            if len(word) < 2:
                continue
            
            target_key = tag_map.get(label)
            if target_key:
                # 简单去重
                if word not in entities[target_key]:
                    entities[target_key].append(word)

        return entities