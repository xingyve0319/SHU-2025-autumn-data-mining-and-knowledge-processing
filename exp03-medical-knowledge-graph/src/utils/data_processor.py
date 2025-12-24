import json
import os
import csv
from typing import Dict, List, Any
from src.config.config import config
class DataProcessor:
    @staticmethod
    def load_json_data(file_path: str) -> List[Dict[str, Any]]:
        """
        从JSONL文件加载数据（每行一个JSON对象）
        
        参数:
            file_path: JSONL文件路径
            
        返回:
            List[Dict]: 加载的数据列表
        """
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                        
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"第 {line_num} 行JSON解析失败: {str(e)}")
                        print(f"问题行内容: {line[:100]}...")
                        continue
                        
            if not data:
                print("没有找到有效的JSON数据")
                return []
            
            print(f"成功加载 {len(data)} 条数据")
            return data
            
        except Exception as e:
            print(f"加载JSONL文件失败: {str(e)}")
            return []
    
    @staticmethod
    def save_json_data(data: List[Dict[str, Any]], file_path: str) -> bool:
        """
        保存数据到JSON文件
        
        参数:
            data: 要保存的数据
            file_path: 保存路径
            
        返回:
            bool: 是否保存成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存JSON文件失败: {str(e)}")
            return False
    
    @staticmethod
    def save_to_neo4j_format(data: List[Dict[str, Any]], output_dir: str) -> bool:
        """
        将数据保存为Neo4j兼容的CSV格式
        
        参数:
            data: 要保存的数据
            output_dir: 输出目录
            
        返回:
            bool: 是否保存成功
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存节点
            nodes_file = os.path.join(output_dir, 'nodes.csv')
            with open(nodes_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'type', 'name'])
                
                # 用于去重的集合
                seen_nodes = set()
                
                entity_types = config.entity_types

                for article in data:
                    for field, label in entity_types.items():
                        for item in article.get(field, []):
                            clean_item = str(item).replace(',', ' ').strip()
                            if not clean_item: continue
                            
                            node_id = f"{label.lower()}_{clean_item}"
                            if node_id not in seen_nodes:
                                writer.writerow([node_id, label, clean_item])
                                seen_nodes.add(node_id)
            

                seen_relationships = set()
                # 【修改点】使用配置文件中的 relations_config
                relations_config = config.relations_config
                # 这里不需要再次定义 entity_types，上面已经定义了

                for article in data:
                    # 解包配置: src_field, tgt_field, rel_type
                    for src_field, tgt_field, rel_type in relations_config:
                        src_list = article.get(src_field, [])
                        tgt_list = article.get(tgt_field, [])
                        
                        # 获取对应的 Neo4j Label 用于生成 ID
                        src_label = entity_types[src_field].lower()
                        tgt_label = entity_types[tgt_field].lower()

                        for src in src_list:
                            for tgt in tgt_list:
                                # 清洗数据
                                s_clean = str(src).replace(',', ' ').strip()
                                t_clean = str(tgt).replace(',', ' ').strip()
                                if not s_clean or not t_clean: continue

                                # 【关键点】使用上面查到的 Label 拼接出唯一的 ID
                                # 只有这样，start_id 才能和 nodes.csv 里的 id 对上号！
                                start_id = f"{src_label}_{s_clean}"
                                end_id = f"{tgt_label}_{t_clean}"
                                
                                rel_id = f"{start_id}-{rel_type}-{end_id}"
                                
                                if rel_id not in seen_relationships:
                                    writer.writerow([start_id, end_id, rel_type])
                                    seen_relationships.add(rel_id)
            
            
        except Exception as e:
            print(f"保存Neo4j格式文件失败: {str(e)}")
            return False 