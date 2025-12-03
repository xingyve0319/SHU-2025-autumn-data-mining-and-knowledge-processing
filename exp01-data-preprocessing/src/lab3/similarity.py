import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import cosine, euclidean
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import os

# 设置日志
logging.basicConfig(
    filename='Part3/similarity_analysis.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class VectorSimilarityAnalyzer:
    def __init__(self):
        self.word2vec_cbow = None
        self.word2vec_sg = None
        self.node2vec_model = None
        self.node2vec_graph = None
        
    def load_models(self):
        """加载Word2Vec和Node2Vec模型"""
        print("加载预训练模型...")
        
        # 加载Word2Vec模型
        try:
            cbow_path = "Part1/word2vec_cbow.model"
            sg_path = "Part1/word2vec_sg.model"
            
            if os.path.exists(cbow_path):
                self.word2vec_cbow = Word2Vec.load(cbow_path)
                print(f"CBOW模型加载成功，词汇表大小: {len(self.word2vec_cbow.wv.key_to_index)}")
            else:
                print("CBOW模型文件不存在")
                
            if os.path.exists(sg_path):
                self.word2vec_sg = Word2Vec.load(sg_path)
                print(f"Skip-gram模型加载成功，词汇表大小: {len(self.word2vec_sg.wv.key_to_index)}")
            else:
                print("Skip-gram模型文件不存在")
                
        except Exception as e:
            print(f"加载Word2Vec模型失败: {e}")
        
        # 加载Node2Vec模型
        try:
            node2vec_path = "Part2/node2vec_linkedin.model"
            graph_path = "Part2/node2vec_linkedin_graph.pkl"
            
            if os.path.exists(node2vec_path):
                self.node2vec_model = Word2Vec.load(node2vec_path)
                print(f"Node2Vec模型加载成功，节点数量: {len(self.node2vec_model.wv.key_to_index)}")
            else:
                print("Node2Vec模型文件不存在")
                
        except Exception as e:
            print(f"加载Node2Vec模型失败: {e}")
    
    def cosine_similarity_vectors(self, vec1, vec2):
        """计算两个向量的余弦相似度"""
        if vec1 is None or vec2 is None:
            return 0.0
        return 1 - cosine(vec1, vec2)
    
    def euclidean_similarity_vectors(self, vec1, vec2):
        """计算两个向量的欧几里得相似度（基于距离的相似度）"""
        if vec1 is None or vec2 is None:
            return 0.0
        distance = euclidean(vec1, vec2)
        # 将距离转换为相似度（距离越小，相似度越大）
        return 1 / (1 + distance)
    
    def dot_product_similarity(self, vec1, vec2):
        """计算两个向量的点积相似度"""
        if vec1 is None or vec2 is None:
            return 0.0
        return np.dot(vec1, vec2)
    
    def get_word_vector(self, word, model_type='cbow'):
        """获取词的向量表示"""
        if model_type == 'cbow' and self.word2vec_cbow and word in self.word2vec_cbow.wv:
            return self.word2vec_cbow.wv[word]
        elif model_type == 'sg' and self.word2vec_sg and word in self.word2vec_sg.wv:
            return self.word2vec_sg.wv[word]
        else:
            return None
    
    def get_node_vector(self, node):
        """获取节点的向量表示"""
        if self.node2vec_model and node in self.node2vec_model.wv:
            return self.node2vec_model.wv[node]
        else:
            return None
    
    def analyze_word_similarities(self, word_pairs):
        """分析词对相似度"""
        print("\n" + "="*60)
        print("Word2Vec 词对相似度分析")
        print("="*60)
        
        results = []
        
        for word1, word2 in word_pairs:
            row = {'word1': word1, 'word2': word2}
            
            # CBOW模型相似度
            vec1_cbow = self.get_word_vector(word1, 'cbow')
            vec2_cbow = self.get_word_vector(word2, 'cbow')
            
            if vec1_cbow is not None and vec2_cbow is not None:
                row['cbow_cosine'] = self.cosine_similarity_vectors(vec1_cbow, vec2_cbow)
                row['cbow_euclidean'] = self.euclidean_similarity_vectors(vec1_cbow, vec2_cbow)
                row['cbow_dot'] = self.dot_product_similarity(vec1_cbow, vec2_cbow)
            else:
                row['cbow_cosine'] = row['cbow_euclidean'] = row['cbow_dot'] = None
            
            # Skip-gram模型相似度
            vec1_sg = self.get_word_vector(word1, 'sg')
            vec2_sg = self.get_word_vector(word2, 'sg')
            
            if vec1_sg is not None and vec2_sg is not None:
                row['sg_cosine'] = self.cosine_similarity_vectors(vec1_sg, vec2_sg)
                row['sg_euclidean'] = self.euclidean_similarity_vectors(vec1_sg, vec2_sg)
                row['sg_dot'] = self.dot_product_similarity(vec1_sg, vec2_sg)
            else:
                row['sg_cosine'] = row['sg_euclidean'] = row['sg_dot'] = None
            
            results.append(row)
        
        # 创建结果DataFrame
        df_results = pd.DataFrame(results)
        
        # 打印结果
        print("\n词对相似度分析结果:")
        print("-" * 100)
        for idx, row in df_results.iterrows():
            print(f"词对: '{row['word1']}' vs '{row['word2']}'")
            if row['cbow_cosine'] is not None:
                print(f"  CBOW - 余弦: {row['cbow_cosine']:.4f}, 欧氏: {row['cbow_euclidean']:.4f}, 点积: {row['cbow_dot']:.4f}")
            else:
                print(f"  CBOW - 词不在词汇表中")
            
            if row['sg_cosine'] is not None:
                print(f"  Skip-gram - 余弦: {row['sg_cosine']:.4f}, 欧氏: {row['sg_euclidean']:.4f}, 点积: {row['sg_dot']:.4f}")
            else:
                print(f"  Skip-gram - 词不在词汇表中")
            print()
        
        return df_results
    
    def analyze_node_similarities(self, node_pairs):
        """分析节点对相似度"""
        print("\n" + "="*60)
        print("Node2Vec 节点对相似度分析")
        print("="*60)
        
        results = []
        
        for node1, node2 in node_pairs:
            row = {'node1': node1, 'node2': node2}
            
            vec1 = self.get_node_vector(node1)
            vec2 = self.get_node_vector(node2)
            
            if vec1 is not None and vec2 is not None:
                row['cosine_similarity'] = self.cosine_similarity_vectors(vec1, vec2)
                row['euclidean_similarity'] = self.euclidean_similarity_vectors(vec1, vec2)
                row['dot_product'] = self.dot_product_similarity(vec1, vec2)
            else:
                row['cosine_similarity'] = row['euclidean_similarity'] = row['dot_product'] = None
            
            results.append(row)
        
        # 创建结果DataFrame
        df_results = pd.DataFrame(results)
        
        # 打印结果
        print("\n节点对相似度分析结果:")
        print("-" * 80)
        for idx, row in df_results.iterrows():
            print(f"节点对: '{row['node1']}' vs '{row['node2']}'")
            if row['cosine_similarity'] is not None:
                print(f"  余弦相似度: {row['cosine_similarity']:.4f}")
                print(f"  欧氏相似度: {row['euclidean_similarity']:.4f}")
                print(f"  点积相似度: {row['dot_product']:.4f}")
            else:
                print(f"  节点不在模型中")
            print()
        
        return df_results
    
    def find_most_similar_words(self, target_words, topn=5):
        """查找最相似的词"""
        print("\n" + "="*60)
        print("最相似词查找")
        print("="*60)
        
        for target_word in target_words:
            print(f"\n目标词: '{target_word}'")
            
            # CBOW模型
            if self.word2vec_cbow and target_word in self.word2vec_cbow.wv:
                print("  CBOW模型最相似词:")
                similar_words = self.word2vec_cbow.wv.most_similar(target_word, topn=topn)
                for word, similarity in similar_words:
                    print(f"    {word}: {similarity:.4f}")
            else:
                print(f"  CBOW模型: 词 '{target_word}' 不在词汇表中")
            
            # Skip-gram模型
            if self.word2vec_sg and target_word in self.word2vec_sg.wv:
                print("  Skip-gram模型最相似词:")
                similar_words = self.word2vec_sg.wv.most_similar(target_word, topn=topn)
                for word, similarity in similar_words:
                    print(f"    {word}: {similarity:.4f}")
            else:
                print(f"  Skip-gram模型: 词 '{target_word}' 不在词汇表中")
    
    def find_most_similar_nodes(self, target_nodes, topn=5):
        """查找最相似的节点"""
        print("\n" + "="*60)
        print("最相似节点查找")
        print("="*60)
        
        for target_node in target_nodes:
            print(f"\n目标节点: '{target_node}'")
            
            if self.node2vec_model and target_node in self.node2vec_model.wv:
                print("  最相似节点:")
                similar_nodes = self.node2vec_model.wv.most_similar(target_node, topn=topn)
                for node, similarity in similar_nodes:
                    print(f"    {node}: {similarity:.4f}")
            else:
                print(f"  节点 '{target_node}' 不在模型中")
    
    def cross_model_similarity_analysis(self):
        """跨模型相似度分析"""
        print("\n" + "="*60)
        print("跨模型相似度分析")
        print("="*60)
        
        # 分析Word2Vec两种模型的向量差异
        if self.word2vec_cbow and self.word2vec_sg:
            common_words = set(self.word2vec_cbow.wv.key_to_index.keys()) & set(self.word2vec_sg.wv.key_to_index.keys())
            common_words = list(common_words)[:10]  # 取前10个共同词
            
            print(f"分析 {len(common_words)} 个共同词的跨模型相似度:")
            
            for word in common_words:
                cbow_vec = self.get_word_vector(word, 'cbow')
                sg_vec = self.get_word_vector(word, 'sg')
                
                if cbow_vec is not None and sg_vec is not None:
                    cross_similarity = self.cosine_similarity_vectors(cbow_vec, sg_vec)
                    print(f"  词 '{word}': CBOW与Skip-gram向量相似度 = {cross_similarity:.4f}")
    
    def visualize_similarity_matrix(self, words, model_type='cbow'):
        """可视化相似度矩阵"""
        if model_type == 'cbow' and not self.word2vec_cbow:
            return
        elif model_type == 'sg' and not self.word2vec_sg:
            return
        
        # 获取词的向量
        vectors = []
        valid_words = []
        
        for word in words:
            vec = self.get_word_vector(word, model_type)
            if vec is not None:
                vectors.append(vec)
                valid_words.append(word)
        
        if len(vectors) < 2:
            print("有效词数量不足，无法创建相似度矩阵")
            return
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(vectors)
        
        # 创建热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, 
                   xticklabels=valid_words, 
                   yticklabels=valid_words,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.3f')
        plt.title(f'Word Similarity Matrix ({model_type.upper()} Model)')
        plt.tight_layout()
        
        # 保存图像
        os.makedirs('Part3', exist_ok=True)
        plt.savefig(f'Part3/similarity_matrix_{model_type}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"相似度矩阵已保存为 'Part3/similarity_matrix_{model_type}.png'")

def main():
    print("实验三: 向量表示的相似度计算")
    print("="*50)
    
    # 创建分析器实例
    analyzer = VectorSimilarityAnalyzer()
    
    # 加载模型
    analyzer.load_models()
    
    # 确保输出目录存在
    os.makedirs('Part3', exist_ok=True)
    
    # 1. 词对相似度分析
    word_pairs = [
        ('good', 'great'),
        ('good', 'bad'),
        ('excellent', 'terrible'),
        ('product', 'quality'),
        ('fast', 'slow'),
        ('happy', 'sad'),
        ('computer', 'laptop'),
        ('car', 'vehicle'),
        ('music', 'song'),
        ('book', 'novel')
    ]
    
    word_similarity_results = analyzer.analyze_word_similarities(word_pairs)
    
    # 2. 节点对相似度分析（使用Node2Vec模型中的实际节点）
    node_pairs = []
    if analyzer.node2vec_model:
        # 获取模型中的一些节点
        all_nodes = list(analyzer.node2vec_model.wv.key_to_index.keys())
        
        # 按类型分组节点
        job_nodes = [n for n in all_nodes if n.startswith('JOB_')]
        company_nodes = [n for n in all_nodes if n.startswith('COMP_')]
        skill_nodes = [n for n in all_nodes if n.startswith('SKILL_')]
        
        # 创建节点对
        if len(job_nodes) >= 2:
            node_pairs.append((job_nodes[0], job_nodes[1]))
        if len(company_nodes) >= 2:
            node_pairs.append((company_nodes[0], company_nodes[1]))
        if len(skill_nodes) >= 2:
            node_pairs.append((skill_nodes[0], skill_nodes[1]))
        if job_nodes and company_nodes:
            node_pairs.append((job_nodes[0], company_nodes[0]))
        if job_nodes and skill_nodes:
            node_pairs.append((job_nodes[0], skill_nodes[0]))
    
    if node_pairs:
        node_similarity_results = analyzer.analyze_node_similarities(node_pairs)
    
    # 3. 查找最相似的词
    target_words = ['good', 'python', 'data', 'learning', 'software']
    analyzer.find_most_similar_words(target_words)
    
    # 4. 查找最相似的节点
    if analyzer.node2vec_model:
        target_nodes = all_nodes[:3] if len(all_nodes) >= 3 else all_nodes
        analyzer.find_most_similar_nodes(target_nodes)
    
    # 5. 跨模型相似度分析
    analyzer.cross_model_similarity_analysis()
    
    # 6. 可视化相似度矩阵
    visualization_words = ['good', 'great', 'excellent', 'bad', 'terrible', 'awful']
    analyzer.visualize_similarity_matrix(visualization_words, 'cbow')
    analyzer.visualize_similarity_matrix(visualization_words, 'sg')
    
    # 保存详细结果
    print("\n" + "="*60)
    print("保存分析结果")
    print("="*60)
    
    # 保存词相似度结果
    if not word_similarity_results.empty:
        word_similarity_results.to_csv('Part3/word_similarity_results.csv', index=False)
        print("词相似度结果已保存到 'Part3/word_similarity_results.csv'")
    
    # 保存节点相似度结果
    if 'node_similarity_results' in locals() and not node_similarity_results.empty:
        node_similarity_results.to_csv('Part3/node_similarity_results.csv', index=False)
        print("节点相似度结果已保存到 'Part3/node_similarity_results.csv'")
    
    # 生成分析报告
    with open('Part3/similarity_analysis_report.txt', 'w') as f:
        f.write("向量相似度分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. Word2Vec模型分析\n")
        f.write(f"   - CBOW模型词汇表大小: {len(analyzer.word2vec_cbow.wv.key_to_index) if analyzer.word2vec_cbow else 0}\n")
        f.write(f"   - Skip-gram模型词汇表大小: {len(analyzer.word2vec_sg.wv.key_to_index) if analyzer.word2vec_sg else 0}\n\n")
        
        f.write("2. Node2Vec模型分析\n")
        f.write(f"   - 节点数量: {len(analyzer.node2vec_model.wv.key_to_index) if analyzer.node2vec_model else 0}\n\n")
    
    
    print("分析报告已保存到 'Part3/similarity_analysis_report.txt'")
    print("\n实验三完成: 向量表示的相似度计算!")

if __name__ == "__main__":
    main()