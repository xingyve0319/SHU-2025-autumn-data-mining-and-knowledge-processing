import os
import logging
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config.config import cfg
from src.utils.load_data import DataLoader
#  引入 visualization 里的绘图函数
from src.utils.visualization import plot_confusion_matrix, plot_model_comparison 
def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(save_dir, 'traditional_ml.log'), mode='w', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

def run_comparison():
    # 1. 准备目录
    result_dir = "results/traditional_comparison"
    logger = setup_logging(result_dir)
    logger.info("Starting Traditional ML Comparison Experiment...")

    # 2. 加载数据
    loader = DataLoader(cfg)
    logger.info(f"Loading Data...")
    train_texts, train_labels = loader.load_csv(cfg.train_path, nrows=None) 
    test_texts, test_labels = loader.load_csv(cfg.test_path, nrows=None)

    # 3. 特征提取
    logger.info("Extracting TF-IDF Features...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    models = {
        "Naive_Bayes": MultinomialNB(),
        "SVM": LinearSVC(random_state=42, max_iter=1000)
    }

    results = []
    
    # 用于画柱状图的数据
    model_names = []
    model_accs = []

    # 4. 训练与评估
    for name, model in models.items():
        logger.info(f"--- Training {name} ---")
        start_time = datetime.now()
        
        model.fit(X_train, train_labels)
        train_time = datetime.now() - start_time
        
        preds = model.predict(X_test)
        acc = accuracy_score(test_labels, preds)
        
        logger.info(f"{name} Test Accuracy: {acc:.4f}")
        
        # 混淆矩阵 (直接调用 visualization.py 里的函数)
        cm = confusion_matrix(test_labels, preds)
        cm_path = os.path.join(result_dir, f"{name}_confusion_matrix.png")
        plot_confusion_matrix(cm, ['Negative', 'Positive'], cm_path)
        
        results.append({"Model": name, "Accuracy": acc, "Train_Time": str(train_time)})
        
        # 收集数据画柱状图
        model_names.append(name)
        model_accs.append(acc)

    # 5. 保存结果
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(result_dir, "comparison_metrics.csv"), index=False)
    
    #  绘制对比柱状图 
    comp_plot_path = os.path.join(result_dir, "model_accuracy_comparison.png")
    plot_model_comparison(model_names, model_accs, comp_plot_path)
    
    logger.info("Experiment Done! Check 'results/traditional_comparison' folder.")

if __name__ == "__main__":
    run_comparison()
    models = ['Naive Bayes', 'SVM', 'BERT-Base', 'Qwen-0.5B']
    accuracies = [0.736, 0.782, 0.7350, 0.9300] 
    
    save_path = "results/traditional_comparison/final_report_comparison.png"
    
    print(f"Generating Final Report Chart to {save_path}...")
    plot_model_comparison(models, accuracies, save_path)
    print("Done!")