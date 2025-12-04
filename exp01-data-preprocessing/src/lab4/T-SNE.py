import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import logging
import os
import argparse

sns.set_style("whitegrid")

# =============== 统一输出目录 ===============
BASE_DIR = "result/lab4"
os.makedirs(BASE_DIR, exist_ok=True)

logging.basicConfig(
    filename=f"{BASE_DIR}/visualization.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

class VectorVisualization:

    def __init__(self, args):
        self.args = args
        self.word2vec_cbow = None
        self.word2vec_sg = None
        self.node2vec_model = None

    def load_models(self):
        if os.path.exists("result/lab1/word2vec_cbow.model"):
            self.word2vec_cbow = Word2Vec.load("result/lab1/word2vec_cbow.model")

        if os.path.exists("result/lab1/word2vec_sg.model"):
            self.word2vec_sg = Word2Vec.load("result/lab1/word2vec_sg.model")

        if os.path.exists("result/lab2/node2vec_linkedin.model"):
            self.node2vec_model = Word2Vec.load("result/lab2/node2vec_linkedin.model")

    def tsne(self, vectors):
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            random_state=42,
            learning_rate="auto",
            init="random",
            max_iter=1000
        )
        return tsne.fit_transform(vectors)

    def visualize_words(self, model, name):
        words = list(model.wv.key_to_index.keys())[:self.args.word_size]
        vectors = np.array([model.wv[w] for w in words])
        emb = self.tsne(vectors)

        out = f"{BASE_DIR}/{name}_tsne.png"
        plt.figure(figsize=self.args.figsize)
        plt.scatter(emb[:, 0], emb[:, 1], alpha=0.7)

        for i, w in enumerate(words[:self.args.annotate]):
            plt.annotate(w, (emb[i, 0], emb[i, 1]))

        plt.title(f"{name} Word Embeddings (t-SNE)")
        plt.savefig(out, dpi=300)
        plt.close()

    def visualize_nodes(self):
        if self.node2vec_model is None:
            return
        
        nodes = list(self.node2vec_model.wv.key_to_index.keys())[:self.args.node_size]
        vectors = np.array([self.node2vec_model.wv[n] for n in nodes])
        emb = self.tsne(vectors)

        out = f"{BASE_DIR}/node2vec_tsne.png"
        plt.figure(figsize=self.args.figsize)
        plt.scatter(emb[:, 0], emb[:, 1], alpha=0.7)

        for i, n in enumerate(nodes[:self.args.annotate]):
            plt.annotate(n, (emb[i, 0], emb[i, 1]))

        plt.title("Node2Vec Embeddings (t-SNE)")
        plt.savefig(out, dpi=300)
        plt.close()
    def compare_word_models(self, method="tsne"):
        if self.word2vec_cbow is None or self.word2vec_sg is None:
            print("需要 CBOW 和 Skip-gram 模型才能比较")
            return

        # 取共同词
        common_vocab = list(
            set(self.word2vec_cbow.wv.key_to_index.keys()) &
            set(self.word2vec_sg.wv.key_to_index.keys())
        )
        words = common_vocab[:self.args.word_size]

        cbow_vectors = np.array([self.word2vec_cbow.wv[w] for w in words])
        sg_vectors   = np.array([self.word2vec_sg.wv[w]   for w in words])

        # 合并 TSNE 保持空间一致
        all_vectors = np.vstack([cbow_vectors, sg_vectors])
        embedded = self.tsne(all_vectors)

        n = len(words)
        cbow_emb = embedded[:n]
        sg_emb   = embedded[n:]

        # 绘图
        plt.figure(figsize=self.args.figsize)

        plt.scatter(cbow_emb[:, 0], cbow_emb[:, 1], alpha=0.7,
                    label="CBOW", s=50, edgecolors="white")
        plt.scatter(sg_emb[:, 0], sg_emb[:, 1], alpha=0.7,
                    label="Skip-gram", s=50, edgecolors="white")

        # 标注重要词
        important = ["good", "bad", "excellent", "terrible", "product", "quality"]
        for i, w in enumerate(words):
            if w in important:
                plt.annotate(w, (cbow_emb[i, 0], cbow_emb[i, 1]),
                             fontsize=8, color="blue", alpha=0.8)
                plt.annotate(w, (sg_emb[i, 0], sg_emb[i, 1]),
                             fontsize=8, color="red", alpha=0.8)

        plt.legend()
        plt.title("Word2Vec Model Comparison (CBOW vs Skip-gram)")
        plt.grid(alpha=0.3)

        # 保存到 result/lab4 根目录下
        out_path = f"{BASE_DIR}/word2vec_models_comparison_tsne.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"模型对比图已保存至: {out_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word-size", type=int, default=200)
    parser.add_argument("--node-size", type=int, default=300)
    parser.add_argument("--annotate", type=int, default=20)
    parser.add_argument("--figsize", type=int, nargs=2, default=[15, 12])
    return parser.parse_args()

def main():
    print("========== 实验四：T-SNE 可视化 ==========")
    args = parse_args()
    vis = VectorVisualization(args)
    vis.load_models()

    if vis.word2vec_cbow:
        vis.visualize_words(vis.word2vec_cbow, "cbow")
    if vis.word2vec_sg:
        vis.visualize_words(vis.word2vec_sg, "skipgram")
    if vis.node2vec_model:
        vis.visualize_nodes()
    print("\n--- 模型对比可视化 ---")
    vis.compare_word_models()
    print("所有输出已保存到：result/lab4")

if __name__ == "__main__":
    main()
