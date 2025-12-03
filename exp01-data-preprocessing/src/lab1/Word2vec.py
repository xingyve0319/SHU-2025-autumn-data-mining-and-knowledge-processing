import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import re
import os
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(
    filename='Part1/word2vec.log',                    
    filemode='a',                        
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# 下载必要的nltk数据
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def preprocess_text(text):
    """文本预处理函数"""
    if not isinstance(text, str):
        return []
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    return tokens

def load_and_preprocess_data(file_paths, sample_size=None):
    """加载并预处理多个数据文件"""
    all_dataframes = []

    # 加载文件
    for file_path in file_paths:
        if os.path.exists(file_path):
            if sample_size:
                df = pd.read_csv(file_path, nrows=sample_size)
            else:
                df = pd.read_csv(file_path)
            all_dataframes.append(df)
            print(f"成功加载文件: {file_path}, 数据量: {len(df)}")
        else:
            print(f"警告: 文件 {file_path} 不存在")

    if not all_dataframes:
        raise FileNotFoundError("没有找到任何数据文件")

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"合并后总数据量: {len(combined_df)}")

    # 合并标题和评论
    if len(combined_df.columns) >= 3:
        combined_df['text'] = combined_df.iloc[:, 1].astype(str) + " " + combined_df.iloc[:, 2].astype(str)
    else:
        combined_df['text'] = combined_df.iloc[:, 0].astype(str)

    # tqdm 
    corpus = []
    for text in tqdm(combined_df['text'], desc="预处理文本"):
        tokens = preprocess_text(text)
        if tokens:  
            corpus.append(tokens)

    return corpus, combined_df.iloc[:, 0].values, combined_df

def train_word2vec(corpus, vector_size=100, window=5, min_count=5, workers=4, sg=1):
    """训练Word2Vec模型
    
    参数:
    - sg: 1 for Skip-gram, 0 for CBOW (word2vec的两种模式)
    """
    mode_name = "Skip-gram" if sg == 1 else "CBOW"
    print(f"训练 Word2Vec 模型中... 模式: {mode_name}")
    
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg
    )
    print(f"Word2Vec {mode_name} 训练完成")
    return model

def get_document_vector(text, model):
    """获取文档的平均词向量"""
    tokens = preprocess_text(text)
    vectors = [model.wv[token] for token in tokens if token in model.wv]

    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)

def compare_models_on_same_tasks(cbow_model, sg_model, test_data):
    """在两个模型上执行相同的任务进行比较"""
    print("\n" + "="*60)
    print("CBOW vs Skip-gram 模型性能比较")
    print("="*60)
    
    # 测试词汇
    test_words = ['good', 'bad', 'product', 'quality', 'great', 'excellent', 'terrible']
    
    print("\n1. 词相似度比较:")
    for word in test_words:
        if word in cbow_model.wv and word in sg_model.wv:
            print(f"\n单词: '{word}'")
            
            print("  CBOW相似词:")
            cbow_similar = cbow_model.wv.most_similar(word, topn=3)
            for w, sim in cbow_similar:
                print(f"    {w}: {sim:.4f}")
            
            print("  Skip-gram相似词:")
            sg_similar = sg_model.wv.most_similar(word, topn=3)
            for w, sim in sg_similar:
                print(f"    {w}: {sim:.4f}")
    
    # 词类比任务比较
    print("\n2. 词类比任务比较:")
    analogies = [
        (['good'], ['bad'], 'positive-negative'),
        (['king', 'man'], ['woman'], 'gender'),
        (['fast'], ['slow'], 'speed')
    ]
    
    for pos, neg, task_type in analogies:
        print(f"\n任务: {task_type}")
        try:
            if all(word in cbow_model.wv for word in pos + neg):
                cbow_result = cbow_model.wv.most_similar(positive=pos, negative=neg, topn=1)
                print(f"  CBOW结果: {cbow_result[0][0]} ({cbow_result[0][1]:.4f})")
            
            if all(word in sg_model.wv for word in pos + neg):
                sg_result = sg_model.wv.most_similar(positive=pos, negative=neg, topn=1)
                print(f"  Skip-gram结果: {sg_result[0][0]} ({sg_result[0][1]:.4f})")
        except:
            print(f"  无法计算此词类比")
    
    # 词汇表覆盖比较
    print("\n3. 词汇表覆盖比较:")
    cbow_vocab = set(cbow_model.wv.key_to_index.keys())
    sg_vocab = set(sg_model.wv.key_to_index.keys())
    
    print(f"  CBOW词汇表大小: {len(cbow_vocab)}")
    print(f"  Skip-gram词汇表大小: {len(sg_vocab)}")
    print(f"  共同词汇: {len(cbow_vocab & sg_vocab)}")
    print(f"  CBOW特有词汇: {len(cbow_vocab - sg_vocab)}")
    print(f"  Skip-gram特有词汇: {len(sg_vocab - cbow_vocab)}")
    
    # 文档向量生成比较
    print("\n4. 文档向量生成比较:")
    sample_texts = [
        "this product is amazing and works great",
        "poor quality and terrible experience",
        "good product with fast delivery"
    ]
    
    for i, text in enumerate(sample_texts):
        cbow_doc_vec = get_document_vector(text, cbow_model)
        sg_doc_vec = get_document_vector(text, sg_model)
        
        print(f"\n文档 {i+1}: '{text}'")
        print(f"  CBOW文档向量范数: {np.linalg.norm(cbow_doc_vec):.4f}")
        print(f"  Skip-gram文档向量范数: {np.linalg.norm(sg_doc_vec):.4f}")
        
        # 计算两个模型生成向量的相似度
        if np.any(cbow_doc_vec) and np.any(sg_doc_vec):
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([cbow_doc_vec], [sg_doc_vec])[0][0]
            print(f"  两个模型向量相似度: {similarity:.4f}")

def main():
    data_files = ['dataset/train_part_1.csv', 'dataset/train_part_2.csv']
    test_files = ['dataset/test.csv']
    
    # 定义两个模型的路径
    cbow_model_path = "Part1/word2vec_cbow.model"
    sg_model_path = "Part1/word2vec_sg.model"
    
    # 检查模型是否已存在
    if os.path.exists(cbow_model_path) and os.path.exists(sg_model_path):
        print("发现已训练模型，直接加载...")
        cbow_model = Word2Vec.load(cbow_model_path)
        sg_model = Word2Vec.load(sg_model_path)
        
        print(f"CBOW模型词汇表大小: {len(cbow_model.wv.key_to_index)}")
        print(f"Skip-gram模型词汇表大小: {len(sg_model.wv.key_to_index)}")
        
    else:
        print("未发现模型或模型不完整，进行训练...")
        # 使用小样本进行训练（避免内存问题）
        corpus, labels, combined_df = load_and_preprocess_data(data_files, sample_size=100000)
        
        # 训练两个模型
        print("\n训练CBOW模型...")
        cbow_model = train_word2vec(corpus, sg=0)  # CBOW模式
        
        print("\n训练Skip-gram模型...")
        sg_model = train_word2vec(corpus, sg=1)  # Skip-gram模式
        
        # 保存两个模型
        cbow_model.save(cbow_model_path)
        sg_model.save(sg_model_path)
        print(f"\n两个模型已保存:")
        print(f"  CBOW模型: {cbow_model_path}")
        print(f"  Skip-gram模型: {sg_model_path}")
        
        # 为两个模型生成文档向量
        print("\n为两个模型生成文档向量...")
        for model, model_name in [(cbow_model, "CBOW"), (sg_model, "Skip-gram")]:
            doc_vectors = []
            sample_texts = combined_df['text'].head(1000)  # 使用部分数据
            
            for text in tqdm(sample_texts, desc=f"生成{model_name}文档向量"):
                doc_vector = get_document_vector(text, model)
                doc_vectors.append(doc_vector)
            
            X = np.array(doc_vectors)
            print(f"{model_name}文档向量形状: {X.shape}")
    
    # 在两个模型上执行相同的任务进行比较
    try:
        # 加载测试数据进行比较
        test_corpus, test_labels, test_df = load_and_preprocess_data(test_files, sample_size=5000)
        compare_models_on_same_tasks(cbow_model, sg_model, test_df)
    except Exception as e:
        print(f"比较任务失败: {e}")
        # 即使没有测试数据，也进行基本比较
        compare_models_on_same_tasks(cbow_model, sg_model, None)

    # 分别测试两个模型的词相似性
    print("\n" + "="*50)
    print("两个模型的词相似性测试")
    print("="*50)
    
    test_words = ['good', 'bad', 'great', 'product', 'quality']
    
    for model, model_name in [(cbow_model, "CBOW"), (sg_model, "Skip-gram")]:
        print(f"\n{model_name}模型词相似性:")
        for word in test_words:
            if word in model.wv:
                similar_words = model.wv.most_similar(word, topn=3)
                print(f"  与 '{word}' 最相似的词:")
                for w, score in similar_words:
                    print(f"    {w}: {score:.4f}")
            else:
                print(f"  单词 '{word}' 不在{model_name}词汇表中")

    print("\n实验一完成: Word2Vec两种模式比较!")
    print("训练了CBOW和Skip-gram两个模型")
    print("比较了两种模式的性能差异")

if __name__ == "__main__":
    main()