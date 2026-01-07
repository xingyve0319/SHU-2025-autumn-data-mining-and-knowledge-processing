import torch
import torch.nn as nn
from transformers import AutoModel

class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        self.hidden_size = self.encoder.config.hidden_size
        
        self.classifier = nn.Linear(self.hidden_size, num_classes)
    
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        # 传入模型
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # BERT 有 pooler_output 
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            feature = outputs.pooler_output
        else:
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