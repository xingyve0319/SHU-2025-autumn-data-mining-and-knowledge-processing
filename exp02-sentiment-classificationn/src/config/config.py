import yaml

class Config:
    """
    配置管理器:从YAML文件加载。
    """
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f) or {}
        for key, value in yaml_config.items():
            setattr(self, key, value)

cfg = Config()