import json
import os
import ast
import re
from neo4j import GraphDatabase
from tqdm import tqdm

# --- 配置区域 ---
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "12345678")
DATA_FILE = "data/processed/processed_articles.json" 
# ----------------

class MedicalGraphBuilder:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        # 扩展补全词库，包含你发现的所有缺字词
        self.fix_map = {
            "烦躁不": "烦躁不安",
            "难以入": "难以入睡",
            "感觉过度焦": "感觉过度焦虑",
            "坐立不": "坐立不安",
            "高血": "高血压",
            "糖尿": "糖尿病",
            "心肌": "心肌炎",
            "结膜": "结膜炎",
            "呼吸困": "呼吸困难",
            "胸部疼": "胸部疼痛",
            "消化不": "消化不良"
        }

    def close(self):
        self.driver.close()

    def _smart_complete(self, entity, full_text):
        """
        全量探测逻辑：只要实体在原文中，就强行向后多吃一个字，除非那是标点符号
        """
        if not entity or not full_text:
            return entity
        
        # 1. 先过硬编码字典 (最稳)
        if entity in self.fix_map:
            return self.fix_map[entity]

        # 2. 原文动态补全
        try:
            # 在原文里找这个词
            match = re.search(re.escape(entity) + r"([\u4e00-\u9fa5])", full_text)
            if match:
                suffix = match.group(1)
                # 如果后面那个字不是常见的连接词，就接上来
                if suffix not in ["，", "。", "的", "了", "和", "、"]:
                    return entity + suffix
        except:
            pass
        return entity

    def build_graph(self, file_path, batch_size=1000):
        if not os.path.exists(file_path):
            print(f"❌ 找不到文件: {file_path}")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)

        print(f"🚀 正在执行终极修复并导入 {len(all_data)} 条数据...")

        # 1. 清空旧数据 (非常重要！否则你看到的还是旧的缺字节点)
        with self.driver.session() as session:
            print("🧹 正在清空数据库以替换旧节点...")
            session.run("MATCH (n) DETACH DELETE n")

        current_batch = []
        keys_to_fix = ['diseases', 'symptoms', 'drugs', 'checks']

        for item in tqdm(all_data, desc="修复中"):
            full_text = item.get('translated', "")
            
            for key in keys_to_fix:
                entities = item.get(key, [])
                if isinstance(entities, str):
                    try: entities = ast.literal_eval(entities)
                    except: entities = [entities]
                
                fixed_entities = []
                for ent in (entities or []):
                    # 运行补全逻辑
                    fixed_ent = self._smart_complete(ent, full_text)
                    # 二次校验补全
                    if fixed_ent in self.fix_map: 
                        fixed_ent = self.fix_map[fixed_ent]
                        
                    if len(fixed_ent) > 1:
                        fixed_entities.append(fixed_ent)
                
                item[key] = list(set(fixed_entities))

            current_batch.append(item)
            if len(current_batch) >= batch_size:
                self._submit_batch(current_batch)
                current_batch = []
        
        if current_batch:
            self._submit_batch(current_batch)
        print("✅ 修复与导入完成！")

    def _submit_batch(self, batch):
        with self.driver.session() as session:
            session.execute_write(self._create_subgraph, batch)

    @staticmethod
    def _create_subgraph(tx, batch_data):
        query = """
        UNWIND $batch AS row
        FOREACH (d_name IN row.diseases | MERGE (d:Disease {name: d_name}))
        FOREACH (s_name IN row.symptoms | MERGE (s:Symptom {name: s_name}))
        FOREACH (dr_name IN row.drugs   | MERGE (dr:Drug {name: dr_name}))
        FOREACH (c_name IN row.checks   | MERGE (c:Check {name: c_name}))
        WITH row
        UNWIND row.diseases AS d_name
        MATCH (d:Disease {name: d_name})
        FOREACH (s_name IN row.symptoms | MERGE (s:Symptom {name: s_name}) MERGE (d)-[:HAS_SYMPTOM]->(s))
        FOREACH (dr_name IN row.drugs | MERGE (dr:Drug {name: dr_name}) MERGE (d)-[:RECOMMEND_DRUG]->(dr))
        FOREACH (c_name IN row.checks | MERGE (c:Check {name: c_name}) MERGE (d)-[:NEED_CHECK]->(c))
        """
        tx.run(query, batch=batch_data)

if __name__ == "__main__":
    builder = MedicalGraphBuilder(URI, AUTH)
    builder.build_graph(DATA_FILE)
    builder.close()