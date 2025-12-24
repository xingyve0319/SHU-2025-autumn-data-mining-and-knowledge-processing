import yaml
import os

class Config:
    def __init__(self, config_path="config.yaml"):
        # 确保能找到配置文件
        if not os.path.exists(config_path):
            # 尝试相对于项目根目录查找
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            config_path = os.path.join(base_dir, "config.yaml")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self._cfg = yaml.safe_load(f)

    @property
    def model_name(self):
        return self._cfg['model']['name']

    @property
    def model_kwargs(self):
        """返回用于 AutoModel.from_pretrained 的参数字典"""
        params = self._cfg['model'].copy()
        params.pop('name', None) # name 单独传参，不放入 kwargs
        return params

    @property
    def generation_config(self):
        """返回生成参数字典"""
        return self._cfg['generation'].copy()

    @property
    def entity_types(self):
        """返回实体类型映射"""
        return self._cfg['graph_schema']['entity_types']

    @property
    def relations_config(self):
        """返回关系配置列表"""
        return self._cfg['graph_schema']['relations']

    @property
    def entity_keys(self):
        """返回所有需要提取的实体key列表"""
        return list(self._cfg['graph_schema']['entity_types'].keys())

# 单例模式实例化，方便其他模块直接导入
config = Config()