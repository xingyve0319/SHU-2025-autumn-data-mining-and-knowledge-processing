import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

class MedicalEntityExtractor:
    def __init__(self, model_name="wbb7/bert-base-chinese-medical-ner", device=None):
        """
        初始化基于 BERT 的提取器
        注意：这里的 model_name 必须是一个已经微调过的 NER 模型，不能是纯 bert-base-chinese
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"正在加载 BERT NER 模型: {model_name}...")
        # 加载分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        # 为了兼容你 main.py 里的翻译逻辑，我们需要保留 self.model 和 self.tokenizer 属性
        # 但 BERT 不能翻译，所以这里是个“假动作”，或者你需要保留 Qwen 单独做翻译
        # ⚠️ 重要：为了让 main.py 里的 translate_to_chinese 能跑，
        # 你需要在 init_models 里单独加载 Qwen，而不是复用 extractor.model
        
        # 初始化 NER 流水线
        # aggregation_strategy="simple" 会自动把 "B-DRUG", "I-DRUG" 合并成 "阿司匹林"
        device_id = 0 if self.device.type == "cuda" else -1
        self.ner_pipeline = pipeline(
            "ner", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            device=device_id, 
            aggregation_strategy="simple"
        )
        print("BERT 模型加载完成")

    def extract_entities(self, text):
        if not text:
            return {"symptoms": [], "diseases": [], "checks": [], "drugs": []}

        try:
            results = self.pipe(text[:510])
        except Exception as e:
            return {"symptoms": [], "diseases": [], "checks": [], "drugs": []}
        
        entities = {
            "symptoms": [], "diseases": [], "checks": [], "drugs": []
        }
        
        tag_map = {
            "disease": "diseases", "symptom": "symptoms", 
            "drug": "drugs", "check": "checks"
        }

        for item in results:
            label = item['entity_group']
            raw_word = item['word']
            
            # 1. 去除空格 (解决 "高 血 压" 问题)
            word = raw_word.replace(" ", "").replace("##", "")
            
            # 2. 过滤掉单字垃圾 (比如只提取了一个 "痛" 或 "的")
            if len(word) < 2:
                continue
                
            # 3. 存入字典
            target_key = tag_map.get(label)
            if target_key and word not in entities[target_key]:
                entities[target_key].append(word)

        return entities