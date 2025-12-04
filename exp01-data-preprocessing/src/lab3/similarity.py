import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine, euclidean
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# =============== 统一输出目录（相对路径） ===============
BASE_DIR = "result/lab3"
os.makedirs(BASE_DIR, exist_ok=True)

logging.basicConfig(
    filename=f"{BASE_DIR}/similarity_analysis.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

class VectorSimilarityAnalyzer:
    def __init__(self):
        self.word2vec_cbow = None
        self.word2vec_sg = None
        self.node2vec_model = None

    def load_models(self):
        print("加载模型...")

        cbow_path = "result/lab1/word2vec_cbow.model"
        sg_path   = "result/lab1/word2vec_sg.model"
        node2vec_path = "result/lab2/node2vec_linkedin.model"

        if os.path.exists(cbow_path):
            self.word2vec_cbow = Word2Vec.load(cbow_path)
            print("CBOW 模型加载成功")

        if os.path.exists(sg_path):
            self.word2vec_sg = Word2Vec.load(sg_path)
            print("Skip-gram 模型加载成功")

        if os.path.exists(node2vec_path):
            self.node2vec_model = Word2Vec.load(node2vec_path)
            print("Node2Vec 模型加载成功")

    def cosine_similarity_vectors(self, v1, v2):
        return 1 - cosine(v1, v2)

    def euclidean_similarity_vectors(self, v1, v2):
        return 1 / (1 + euclidean(v1, v2))

    def dot_product_similarity(self, v1, v2):
        return np.dot(v1, v2)

    def get_word_vector(self, word, model):
        if model == "cbow" and self.word2vec_cbow and word in self.word2vec_cbow.wv:
            return self.word2vec_cbow.wv[word]
        if model == "sg" and self.word2vec_sg and word in self.word2vec_sg.wv:
            return self.word2vec_sg.wv[word]
        return None

    def analyze_word_similarities(self, pairs):
        results = []
        for w1, w2 in pairs:
            row = {"word1": w1, "word2": w2}
            for model in ["cbow", "sg"]:
                v1 = self.get_word_vector(w1, model)
                v2 = self.get_word_vector(w2, model)
                if v1 is None or v2 is None:
                    row[f"{model}_cosine"] = None
                    row[f"{model}_euclidean"] = None
                    row[f"{model}_dot"] = None
                else:
                    row[f"{model}_cosine"] = self.cosine_similarity_vectors(v1, v2)
                    row[f"{model}_euclidean"] = self.euclidean_similarity_vectors(v1, v2)
                    row[f"{model}_dot"] = self.dot_product_similarity(v1, v2)
            results.append(row)
        return pd.DataFrame(results)

    def visualize_similarity_matrix(self, words, model):
        vectors, labels = [], []
        for w in words:
            vec = self.get_word_vector(w, model)
            if vec is not None:
                vectors.append(vec)
                labels.append(w)

        if len(vectors) < 2:
            print("有效词数量不足")
            return

        mat = cosine_similarity(vectors)
        plt.figure(figsize=(8, 6))
        sns.heatmap(mat, annot=True, xticklabels=labels, yticklabels=labels, cmap="coolwarm")
        plt.title(f"Similarity Matrix ({model.upper()})")
        plt.tight_layout()
        plt.savefig(f"{BASE_DIR}/similarity_matrix_{model}.png", dpi=300)
        plt.close()

def main():
    print("========== 实验三：相似度计算 ==========")

    analyzer = VectorSimilarityAnalyzer()
    analyzer.load_models()

    word_pairs = [
        ("good", "great"), ("good", "bad"),
        ("excellent", "terrible"), ("product", "quality")
    ]

    df = analyzer.analyze_word_similarities(word_pairs)
    df.to_csv(f"{BASE_DIR}/word_similarity_results.csv", index=False)

    analyzer.visualize_similarity_matrix(
        ["good", "great", "bad", "excellent"], "cbow"
    )
    analyzer.visualize_similarity_matrix(
        ["good", "great", "bad", "excellent"], "sg"
    )

    with open(f"{BASE_DIR}/similarity_analysis_report.txt", "w") as f:
        f.write("Similarity Analysis Completed.\n")

    print("所有输出均已保存到：result/lab3")

if __name__ == "__main__":
    main()
