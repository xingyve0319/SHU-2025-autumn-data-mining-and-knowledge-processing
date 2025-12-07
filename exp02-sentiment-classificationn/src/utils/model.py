import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        # 1. 使用 AutoModel 而不是 BertModel，这样可以自动识别 BERT 或 Qwen
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # 2. 自动获取 hidden_size (BERT是768, Qwen-0.5B可能是896或1024)
        self.hidden_size = self.encoder.config.hidden_size
        
        # 3. 分类头
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # 4. 激活函数 (可选)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        # 传入模型
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # --- 关键点：提取句子特征 ---
        # BERT 有 pooler_output (CLS token 经过处理后的向量)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            feature = outputs.pooler_output
        else:
            # Qwen 等生成式模型没有 pooler_output。
            # 这里的策略是：取最后一个非 padding 的 token，或者简单粗暴取最后一个 token。
            # 为了兼容性，且对于分类任务，常用的做法是取 last_hidden_state 的最后一个 token (EOS位置) 或 第一个 token (BOS位置)
            # 这里我们采用 "Mean Pooling" (取所有 token 的平均值)，这是一种稳健的做法
            
            # outputs.last_hidden_state 形状: [batch, seq_len, hidden]
            # attention_mask 形状: [batch, seq_len] -> 扩展为 [batch, seq_len, 1]
            mask = attention_mask.unsqueeze(-1).float()
            
            # 求和 (把 padding 的地方 mask 掉)
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, dim=1)
            # 算有效 token 数量
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            
            # 平均值
            feature = sum_embeddings / sum_mask

        # 分类
        logits = self.classifier(self.relu(feature))
        return logits
    
    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load_model(self, save_path):
        self.load_state_dict(torch.load(save_path))