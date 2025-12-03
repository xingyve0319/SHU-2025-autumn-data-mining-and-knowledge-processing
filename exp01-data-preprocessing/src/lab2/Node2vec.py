import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec, KeyedVectors
import os
from tqdm import tqdm
import logging
import re
import pickle

# 设置日志
logging.basicConfig(
    filename='Part2/node2vec.log',                    
    filemode='a',                        
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class LinkedInNode2Vec:
    def __init__(self):
        self.graph = None
        self.model = None
        self.node2vec = None
        
    def load_linkedin_data(self, file_path, sample_size=None):
        """读取LinkedIn数据集文件"""
        try:
            print(f"尝试读取数据集: {file_path}")
            
            # 直接使用逗号分隔符，跳过错误行
            read_params = {
                'sep': ',',
                'encoding': 'utf-8',
                'on_bad_lines': 'skip',
                'quotechar': '"'
            }
            
            if sample_size:
                read_params['nrows'] = sample_size
            
            df = pd.read_csv(file_path, **read_params)
            
            print(f"成功读取数据集: {file_path}")
            print(f"数据形状: {df.shape}")
            
            # 显示前几行数据以便调试
            print("\n数据前3行:")
            available_columns = []
            for col in ['title', 'company_name', 'skills_desc']:
                if col in df.columns:
                    available_columns.append(col)
            
            if available_columns:
                print(df[available_columns].head(3))
            else:
                print("未找到标准列名，显示所有列:")
                print(df.head(3))
            
            return df
            
        except Exception as e:
            print(f"读取文件失败: {e}")
            print("创建示例数据进行演示...")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """创建示例LinkedIn数据"""
        sample_data = {
            'title': [
                'Software Engineer', 'Data Scientist', 'Machine Learning Engineer',
                'Frontend Developer', 'Backend Engineer', 'Data Analyst'
            ],
            'company_name': [
                'Google', 'Facebook', 'Amazon', 'Microsoft', 'Apple', 'Netflix'
            ],
            'skills_desc': [
                'Python,Java,SQL,JavaScript',
                'Python,R,Statistics,Machine Learning',
                'Python,TensorFlow,PyTorch,Deep Learning',
                'JavaScript,React,HTML,CSS',
                'Java,Spring,SQL,Microservices',
                'SQL,Excel,Tableau,Statistics'
            ]
        }
        return pd.DataFrame(sample_data)

    def extract_skills_from_description(self, description):
        """从描述中提取技能"""
        if pd.isna(description):
            return []
        
        description = str(description).lower()
        
        # 常见技能关键词
        skill_keywords = [
            'python', 'java', 'javascript', 'sql', 'r', 'c++', 'c#', 'go', 'rust',
            'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring',
            'tensorflow', 'pytorch', 'machine learning', 'deep learning', 'ai',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'linux',
            'tableau', 'power bi', 'excel', 'statistics', 'data analysis',
            'html', 'css', 'typescript', 'mongodb', 'postgresql', 'mysql',
            'marketing', 'graphic design', 'adobe', 'photoshop', 'illustrator',
            'management', 'leadership', 'communication', 'project management'
        ]
        
        found_skills = []
        for skill in skill_keywords:
            if skill in description:
                found_skills.append(skill)
        
        return found_skills

    def create_graph_from_data(self, df):
        """从数据框创建图 - 优化版本"""
        G = nx.Graph()
        
        print("开始构建LinkedIn职业图...")
        
        # 检查必要的列是否存在
        has_title = 'title' in df.columns
        has_company = 'company_name' in df.columns
        has_skills = 'skills_desc' in df.columns
        
        print(f"检测到的列 - 职位: {has_title}, 公司: {has_company}, 技能: {has_skills}")
        
        jobs_added = set()
        companies_added = set()
        skills_added = set()
        
        # 统计信息
        stats = {
            'jobs': 0,
            'companies': 0,
            'skills': 0,
            'job_company_edges': 0,
            'job_skill_edges': 0,
            'company_skill_edges': 0
        }
        
        # 遍历每一行数据
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="构建图中"):
            current_job = None
            current_company = None
            
            # 处理职位标题
            if has_title and pd.notna(row['title']):
                title = str(row['title']).strip()
                if title and title != 'nan' and len(title) > 2:
                    job_id = f"JOB_{title}"
                    if job_id not in jobs_added:
                        G.add_node(job_id, type='job', name=title)
                        jobs_added.add(job_id)
                        stats['jobs'] += 1
                    current_job = job_id
            
            # 处理公司名称
            if has_company and pd.notna(row['company_name']):
                company = str(row['company_name']).strip()
                if company and company != 'nan' and len(company) > 2:
                    company_id = f"COMP_{company}"
                    if company_id not in companies_added:
                        G.add_node(company_id, type='company', name=company)
                        companies_added.add(company_id)
                        stats['companies'] += 1
                    current_company = company_id
            
            # 添加职位-公司关系
            if current_job and current_company:
                if not G.has_edge(current_job, current_company):
                    G.add_edge(current_job, current_company, relationship='works_at')
                    stats['job_company_edges'] += 1
            
            # 处理技能描述
            if has_skills and pd.notna(row['skills_desc']):
                skills_desc = str(row['skills_desc'])
                skills = self.extract_skills_from_description(skills_desc)
                
                for skill in skills:
                    skill_id = f"SKILL_{skill}"
                    if skill_id not in skills_added:
                        G.add_node(skill_id, type='skill', name=skill)
                        skills_added.add(skill_id)
                        stats['skills'] += 1
                    
                    # 添加职位-技能关系
                    if current_job:
                        if not G.has_edge(current_job, skill_id):
                            G.add_edge(current_job, skill_id, relationship='requires_skill')
                            stats['job_skill_edges'] += 1
                    
                    # 添加公司-技能关系
                    if current_company:
                        if not G.has_edge(current_company, skill_id):
                            G.add_edge(current_company, skill_id, relationship='uses_skill')
                            stats['company_skill_edges'] += 1
        
        print(f"\n图构建统计:")
        print(f"  职位节点: {stats['jobs']}")
        print(f"  公司节点: {stats['companies']}")
        print(f"  技能节点: {stats['skills']}")
        print(f"  职位-公司边: {stats['job_company_edges']}")
        print(f"  职位-技能边: {stats['job_skill_edges']}")
        print(f"  公司-技能边: {stats['company_skill_edges']}")
        print(f"  总节点数: {len(G.nodes())}")
        print(f"  总边数: {len(G.edges())}")
        
        # 显示一些示例节点
        if G.nodes():
            print("\n图中的节点示例:")
            # 按类型显示示例节点
            job_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'job']
            company_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'company']
            skill_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'skill']
            
            if job_nodes:
                print(f"  职位示例: {[G.nodes[n].get('name', n) for n in job_nodes[:3]]}")
            if company_nodes:
                print(f"  公司示例: {[G.nodes[n].get('name', n) for n in company_nodes[:3]]}")
            if skill_nodes:
                print(f"  技能示例: {[G.nodes[n].get('name', n) for n in skill_nodes[:5]]}")
        else:
            print("警告: 图为空!")
        
        return G
    
    def train_node2vec(self, dimensions=64, walk_length=30, num_walks=200, 
                      workers=4, p=1, q=1, window=10, min_count=1, epochs=5):
        """训练node2vec模型"""
        if self.graph is None:
            raise ValueError("请先构建图结构")
        
        # 检查图是否为空
        if len(self.graph.nodes()) == 0:
            raise ValueError("图为空，无法训练模型")
        
        print("开始训练node2vec模型...")
        
        # 创建Node2Vec实例
        self.node2vec = Node2Vec(
            self.graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
            p=p,
            q=q,
            quiet=False
        )
        
        # 生成游走
        print("生成随机游走...")
        walks = self.node2vec.walks
        
        # 训练Word2Vec模型
        print("训练Word2Vec模型...")
        self.model = Word2Vec(
            walks,
            vector_size=dimensions,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=1,  # skip-gram
            epochs=epochs
        )
        
        print("node2vec模型训练完成!")
        return self.model
    
    def get_node_vector(self, node):
        """获取节点向量"""
        if self.model and node in self.model.wv:
            return self.model.wv[node]
        else:
            print(f"节点 '{node}' 不在模型中")
            return None
    
    def find_similar_nodes(self, node, topn=10):
        """查找相似节点 - 修复版本，处理节点不存在的情况"""
        if self.model and node in self.model.wv:
            try:
                similar_nodes = self.model.wv.most_similar(node, topn=topn)
                print(f"与 '{node}' 最相似的节点:")
                valid_nodes = []
                for i, (node_name, similarity) in enumerate(similar_nodes, 1):
                    # 检查节点是否在图中
                    if self.graph and node_name in self.graph.nodes:
                        node_type = self.graph.nodes[node_name].get('type', 'unknown')
                        node_display = self.graph.nodes[node_name].get('name', node_name)
                        print(f"  {i:2d}. {node_display} ({node_type}): {similarity:.4f}")
                        valid_nodes.append((node_name, similarity))
                    else:
                        print(f"  {i:2d}. {node_name} (节点不在当前图中): {similarity:.4f}")
                return valid_nodes
            except KeyError as e:
                print(f"查找相似节点时出错: {e}")
                print("这可能是因为模型是在不同的图上训练的")
                return None
        else:
            print(f"节点 '{node}' 不在模型中")
            if self.model:
                print(f"可用节点数量: {len(self.model.wv.index_to_key)}")
                # 显示一些可用节点
                available_nodes = list(self.model.wv.index_to_key)[:5]
                print(f"示例可用节点: {available_nodes}")
            return None
    
    def save_model(self, file_path):
        """保存模型和图"""
        if self.model:
            # 保存模型
            model_path = file_path
            self.model.save(model_path)
            
            # 保存图信息
            graph_path = file_path.replace('.model', '_graph.pkl')
            with open(graph_path, 'wb') as f:
                pickle.dump({
                    'nodes': list(self.graph.nodes(data=True)),
                    'edges': list(self.graph.edges(data=True))
                }, f)
            
            print(f"Node2Vec模型已保存到 {model_path}")
            print(f"图结构已保存到 {graph_path}")
        else:
            print("没有模型可以保存")
    
    def load_model(self, file_path):
        """加载模型和图"""
        try:
            # 加载模型
            model_path = file_path
            self.model = Word2Vec.load(model_path)
            
            # 尝试加载图
            graph_path = file_path.replace('.model', '_graph.pkl')
            if os.path.exists(graph_path):
                with open(graph_path, 'rb') as f:
                    graph_data = pickle.load(f)
                
                # 重建图
                self.graph = nx.Graph()
                self.graph.add_nodes_from(graph_data['nodes'])
                self.graph.add_edges_from(graph_data['edges'])
                
                print(f"Node2Vec模型和图结构已从 {file_path} 加载")
                print(f"模型词汇表大小: {len(self.model.wv.index_to_key)}")
                print(f"图节点数: {len(self.graph.nodes())}")
                return True
            else:
                print(f"图文件 {graph_path} 不存在，只加载了模型")
                return True
                
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False

def main():
    print("实验2: Node2Vec - LinkedIn节点表示学习")
    print("=" * 50)
    
    # 创建Node2Vec实例
    node2vec_model = LinkedInNode2Vec()
    
    # 模型文件路径
    model_path = "Part2/node2vec_linkedin.model"
    graph_path = "Part2/node2vec_linkedin_graph.pkl"
    
    # 检查模型和图是否已存在
    if os.path.exists(model_path) and os.path.exists(graph_path):
        print("发现已训练模型和图，直接加载...")
        success = node2vec_model.load_model(model_path)
        
        if not success:
            print("模型加载失败，重新训练...")
            # 重新训练
            file_path = "dataset/postings.csv"
            df = node2vec_model.load_linkedin_data(file_path, sample_size=5000)
            if df is not None:
                graph = node2vec_model.create_graph_from_data(df)
                node2vec_model.graph = graph
                
                if len(graph.nodes()) > 0:
                    model = node2vec_model.train_node2vec(
                        dimensions=64,
                        walk_length=20,
                        num_walks=100,
                        workers=2,
                        epochs=5
                    )
                    os.makedirs("Part2", exist_ok=True)
                    node2vec_model.save_model(model_path)
                else:
                    print("错误: 无法构建有效的图结构")
    else:
        print("未发现完整模型文件，开始训练...")
        
        # 1. 读取数据集
        file_path = "dataset/postings.csv"
        df = node2vec_model.load_linkedin_data(file_path, sample_size=5000)
        
        if df is not None:
            # 2. 从数据构建图
            graph = node2vec_model.create_graph_from_data(df)
            node2vec_model.graph = graph
            
            # 检查图是否为空
            if len(graph.nodes()) == 0:
                print("警告: 构建的图为空，使用示例数据重新构建...")
                df_sample = node2vec_model.create_sample_data()
                graph = node2vec_model.create_graph_from_data(df_sample)
                node2vec_model.graph = graph
            
            # 3. 训练Node2Vec模型
            if len(node2vec_model.graph.nodes()) > 0:
                model = node2vec_model.train_node2vec(
                    dimensions=64,
                    walk_length=20,
                    num_walks=100,
                    workers=2,
                    epochs=5
                )
                
                # 4. 保存模型和图
                os.makedirs("Part2", exist_ok=True)
                node2vec_model.save_model(model_path)
            else:
                print("错误: 无法构建有效的图结构，跳过训练")
    
    # 5. 测试模型
    if node2vec_model.model and node2vec_model.graph:
        print("\n" + "="*50)
        print("Node2Vec模型测试")
        print("="*50)
        
        # 使用图中实际存在的节点进行测试，按类型选择
        job_nodes = [n for n in node2vec_model.graph.nodes() if node2vec_model.graph.nodes[n].get('type') == 'job']
        company_nodes = [n for n in node2vec_model.graph.nodes() if node2vec_model.graph.nodes[n].get('type') == 'company']
        skill_nodes = [n for n in node2vec_model.graph.nodes() if node2vec_model.graph.nodes[n].get('type') == 'skill']
        
        print(f"可用的职位节点: {len(job_nodes)}")
        print(f"可用的公司节点: {len(company_nodes)}")
        print(f"可用的技能节点: {len(skill_nodes)}")
        
        # 选择测试节点 - 每种类型选一个
        test_nodes = []
        if job_nodes:
            test_nodes.append(job_nodes[0])
        if company_nodes:
            test_nodes.append(company_nodes[0])
        if skill_nodes:
            test_nodes.append(skill_nodes[0])
        
        # 如果节点不够，从所有节点中补充
        if len(test_nodes) < 3:
            all_nodes = list(node2vec_model.graph.nodes())
            test_nodes.extend(all_nodes[:3-len(test_nodes)])
        
        print(f"\n使用图中的实际节点进行测试:")
        for node in test_nodes:
            node_type = node2vec_model.graph.nodes[node].get('type', 'unknown')
            node_name = node2vec_model.graph.nodes[node].get('name', node)
            print(f"  测试节点: {node_name} ({node_type}) - 节点ID: {node}")
        
        print()
        
        for node in test_nodes:
            display_name = node2vec_model.graph.nodes[node].get('name', node)
            print(f"--- 查找与 '{display_name}' 相似的节点 ---")
            result = node2vec_model.find_similar_nodes(node, topn=5)
            if not result:
                print("  未找到相似节点或出现错误")
            print()
    
    print("\n实验二完成: Node2Vec节点表示学习!")

if __name__ == "__main__":
    main()