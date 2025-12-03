import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import os
import argparse
import logging

plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 设置日志
logging.basicConfig(
    filename='Part4/visualization.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class VectorVisualization:
    def __init__(self, args):
        self.args = args
        self.word2vec_cbow = None
        self.word2vec_sg = None
        self.node2vec_model = None
        self.node2vec_graph = None
        
    def load_models(self):
        """Load all pre-trained models"""
        print("Loading pre-trained models...")
        
        # Load Word2Vec models
        try:
            cbow_path = "Part1/word2vec_cbow.model"
            sg_path = "Part1/word2vec_sg.model"
            
            if os.path.exists(cbow_path):
                self.word2vec_cbow = Word2Vec.load(cbow_path)
                print(f"CBOW model loaded successfully, vocabulary size: {len(self.word2vec_cbow.wv.key_to_index)}")
                
            if os.path.exists(sg_path):
                self.word2vec_sg = Word2Vec.load(sg_path)
                print(f"Skip-gram model loaded successfully, vocabulary size: {len(self.word2vec_sg.wv.key_to_index)}")
                
        except Exception as e:
            print(f"Failed to load Word2Vec models: {e}")
        
        # Load Node2Vec model
        try:
            node2vec_path = "Part2/node2vec_linkedin.model"
            if os.path.exists(node2vec_path):
                self.node2vec_model = Word2Vec.load(node2vec_path)
                print(f"Node2Vec model loaded successfully, node count: {len(self.node2vec_model.wv.key_to_index)}")
        except Exception as e:
            print(f"Failed to load Node2Vec model: {e}")
    
    def prepare_word_vectors(self, model, words=None, n_words=None):
        """Prepare word vector data"""
        if n_words is None:
            n_words = self.args.word_size
            
        if words is None:
            # Select most common words
            words = list(model.wv.key_to_index.keys())[:n_words]
        
        vectors = []
        labels = []
        valid_words = []
        
        for word in words:
            if word in model.wv:
                vectors.append(model.wv[word])
                labels.append(word)
                valid_words.append(word)
        
        print(f"Prepared {len(vectors)} word vectors")
        return np.array(vectors), labels, valid_words
    
    def prepare_node_vectors(self, nodes=None, n_nodes=None):
        """Prepare node vector data"""
        if n_nodes is None:
            n_nodes = self.args.node_size
            
        if nodes is None:
            nodes = list(self.node2vec_model.wv.key_to_index.keys())
        
        vectors = []
        labels = []
        node_types = []
        valid_nodes = []
        
        for node in nodes[:n_nodes]:
            if node in self.node2vec_model.wv:
                vectors.append(self.node2vec_model.wv[node])
                labels.append(node)
                valid_nodes.append(node)
                
                # Determine node type
                if node.startswith('JOB_'):
                    node_types.append('job')
                elif node.startswith('COMP_'):
                    node_types.append('company')
                elif node.startswith('SKILL_'):
                    node_types.append('skill')
                else:
                    node_types.append('unknown')
        
        print(f"Prepared {len(vectors)} node vectors")
        return np.array(vectors), labels, node_types, valid_nodes
    
    def apply_tsne(self, vectors, n_components=2, perplexity=30, random_state=42):
        """Apply t-SNE dimensionality reduction"""
        print(f"Applying t-SNE, data shape: {vectors.shape}")
        
        try:
            # Newer scikit-learn uses max_iter instead of n_iter
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                max_iter=1000,
                learning_rate='auto',
                init='random'
            )
        except TypeError:
            # If above parameters don't work, try simpler ones
            try:
                tsne = TSNE(
                    n_components=n_components,
                    perplexity=perplexity,
                    random_state=random_state,
                    max_iter=1000
                )
            except:
                # Minimal version
                tsne = TSNE(
                    n_components=n_components,
                    random_state=random_state
                )
        
        embedded_vectors = tsne.fit_transform(vectors)
        print(f"After dimensionality reduction: {embedded_vectors.shape}")
        
        return embedded_vectors
     
    def smart_text_annotation(self, points, labels, max_annotations=None):
        """Smart text annotation to avoid overlap"""
        if max_annotations is None:
            max_annotations = self.args.annotate
            
        texts = []
        annotated_indices = set()
        
        # Prioritize important words
        important_words = ['good', 'bad', 'excellent', 'terrible', 'great', 'amazing', 
                          'poor', 'quality', 'product', 'fast', 'slow', 'python', 
                          'java', 'data', 'learning', 'software', 'computer']
        
        # First annotate important words
        for i, label in enumerate(labels):
            if label in important_words and len(annotated_indices) < max_annotations // 2:
                texts.append((points[i, 0], points[i, 1], label))
                annotated_indices.add(i)
        
        # Then evenly distribute other annotations
        remaining_slots = max_annotations - len(annotated_indices)
        if remaining_slots > 0:
            step = max(1, len(points) // remaining_slots)
            for i in range(0, len(points), step):
                if i not in annotated_indices and len(annotated_indices) < max_annotations:
                    texts.append((points[i, 0], points[i, 1], labels[i]))
                    annotated_indices.add(i)
        
        return texts
    
    def visualize_word_embeddings(self, model, model_name, words=None, method='tsne'):
        """Visualize word embeddings"""
        print(f"\nVisualizing {model_name} model word embeddings...")
        
        # Prepare data
        vectors, labels, valid_words = self.prepare_word_vectors(model, words)
        
        if len(vectors) == 0:
            print("No valid word vectors for visualization")
            return
        
        # Dimensionality reduction
        if True:
            embedded = self.apply_tsne(vectors)
            title_suffix = 't-SNE'
        embedded = self.apply_tsne(vectors)
        title_suffix = 't-SNE'
        
        # Create visualization - use non-interactive backend
        plt.switch_backend('Agg')
        plt.figure(figsize=self.args.figsize)
        
        # Create scatter plot
        scatter = plt.scatter(embedded[:, 0], embedded[:, 1], alpha=0.7, 
                            s=50, edgecolors='white', linewidth=0.5)
        
        # Smart text annotation
        texts_data = self.smart_text_annotation(embedded, labels)
        
        # Add text annotations
        for i, (x, y, text) in enumerate(texts_data):
            # Adjust font size based on text length
            fontsize = max(6, 10 - len(text) // 5)
            
            # Use different offsets to avoid overlap
            offset_x = 0.02 * (i % 3 - 1)
            offset_y = 0.02 * ((i // 3) % 3 - 1)
            
            plt.annotate(text, (x, y), 
                        xytext=(x + offset_x, y + offset_y),
                        fontsize=fontsize, 
                        alpha=0.8,
                        bbox=dict(boxstyle="round,pad=0.2", 
                                 fc='lightyellow', 
                                 alpha=0.7,
                                 edgecolor='gray',
                                 linewidth=0.5),
                        arrowprops=dict(arrowstyle="->", 
                                      color='gray', 
                                      lw=0.5,
                                      alpha=0.7))
        
        plt.title(f'{model_name} Word Embeddings Visualization ({title_suffix})', fontsize=16, fontweight='bold')
        plt.xlabel(f'{title_suffix} Dimension 1', fontsize=12)
        plt.ylabel(f'{title_suffix} Dimension 2', fontsize=12)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Save image
        os.makedirs('Part4/word_visualizations', exist_ok=True)
        filename = f'Part4/word_visualizations/{model_name.lower()}_{method}_word_embeddings.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Word embeddings visualization saved: {filename}")
        
        return embedded, labels
    
    def visualize_node_embeddings(self, method='tsne'):
        """Visualize node embeddings"""
        if self.node2vec_model is None:
            print("Node2Vec model not loaded")
            return
        
        print("\nVisualizing Node2Vec node embeddings...")
        
        # Prepare data
        vectors, labels, node_types, valid_nodes = self.prepare_node_vectors()
        
        if len(vectors) == 0:
            print("No valid node vectors for visualization")
            return
        
        # Dimensionality reduction
        if True:
            embedded = self.apply_tsne(vectors)
            title_suffix = 't-SNE'
        embedded = self.apply_tsne(vectors)
        title_suffix = 't-SNE'
        
        # Create visualization - use non-interactive backend
        plt.switch_backend('Agg')
        plt.figure(figsize=self.args.figsize)
        
        # Color by node type
        colors = {'job': '#FF6B6B', 'company': '#4ECDC4', 'skill': '#45B7D1', 'unknown': '#96CEB4'}
        sizes = {'job': 80, 'company': 100, 'skill': 60, 'unknown': 50}
        markers = {'job': 'o', 'company': 's', 'skill': '^', 'unknown': 'D'}
        
        for node_type in set(node_types):
            mask = [t == node_type for t in node_types]
            if any(mask):
                plt.scatter(embedded[mask, 0], embedded[mask, 1], 
                           c=colors[node_type], 
                           label=node_type, 
                           s=sizes[node_type], 
                           alpha=0.7, 
                           edgecolors='white', 
                           linewidth=0.5,
                           marker=markers[node_type])
        
        # Smart text annotation - only annotate important nodes
        display_labels = [label.replace('JOB_', '').replace('COMP_', '').replace('SKILL_', '') 
                         for label in labels]
        
        # Select nodes to annotate - improved strategy
        texts_data = []
        annotated_count = 0
        
        # Important skills and companies get priority
        important_skills = ['python', 'java', 'javascript', 'sql', 'machine learning', 'data analysis']
        important_companies = ['google', 'microsoft', 'amazon', 'facebook', 'apple', 'netflix']
        
        for i, (label, node_type) in enumerate(zip(display_labels, node_types)):
            if annotated_count >= self.args.annotate:
                break
                
            should_annotate = False
            
            # Check if important skill
            if node_type == 'skill' and any(skill in label.lower() for skill in important_skills):
                should_annotate = True
            # Check if important company
            elif node_type == 'company' and any(company in label.lower() for company in important_companies):
                should_annotate = True
            # Annotate some jobs
            elif node_type == 'job' and annotated_count < self.args.annotate // 3:
                should_annotate = True
            # Evenly distribute other annotations
            elif i % (len(display_labels) // (self.args.annotate // 2)) == 0:
                should_annotate = True
                
            if should_annotate:
                texts_data.append((embedded[i, 0], embedded[i, 1], label, node_type))
                annotated_count += 1
        
        # Add text annotations
        for i, (x, y, text, node_type) in enumerate(texts_data):
            # Adjust color based on node type
            bbox_color = colors.get(node_type, 'lightyellow')
            
            # Adjust font size based on text length
            fontsize = max(6, 9 - len(text) // 5)
            
            # Use different offsets to avoid overlap
            offset_x = 0.03 * (i % 3 - 1)
            offset_y = 0.03 * ((i // 3) % 3 - 1)
            
            plt.annotate(text, (x, y), 
                        xytext=(x + offset_x, y + offset_y),
                        fontsize=fontsize, 
                        alpha=0.9,
                        bbox=dict(boxstyle="round,pad=0.2", 
                                 fc=bbox_color, 
                                 alpha=0.8,
                                 edgecolor='gray',
                                 linewidth=0.5),
                        arrowprops=dict(arrowstyle="->", 
                                      color='gray', 
                                      lw=0.5,
                                      alpha=0.7))
        
        plt.title(f'Node2Vec Node Embeddings Visualization ({title_suffix})', fontsize=16, fontweight='bold')
        plt.xlabel(f'{title_suffix} Dimension 1', fontsize=12)
        plt.ylabel(f'{title_suffix} Dimension 2', fontsize=12)
        plt.legend(title='Node Type', loc='best')
        plt.grid(True, alpha=0.3)
        
        # Save image
        os.makedirs('Part4/node_visualizations', exist_ok=True)
        filename = f'Part4/node_visualizations/node2vec_{method}_embeddings.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Node embeddings visualization saved: {filename}")
        
        return embedded, labels, node_types
    
    def compare_word_models(self, method='tsne'):
        """Compare two Word2Vec models"""
        if self.word2vec_cbow is None or self.word2vec_sg is None:
            print("Two Word2Vec models are required for comparison")
            return
        
        print("\nComparing CBOW and Skip-gram model word embeddings...")
        
        # Get common vocabulary
        common_vocab = set(self.word2vec_cbow.wv.key_to_index.keys()) & \
                      set(self.word2vec_sg.wv.key_to_index.keys())
        common_words = list(common_vocab)[:self.args.word_size]
        
        # Prepare vectors for both models
        cbow_vectors, cbow_labels, _ = self.prepare_word_vectors(self.word2vec_cbow, common_words)
        sg_vectors, sg_labels, _ = self.prepare_word_vectors(self.word2vec_sg, common_words)
        
        # Combine vectors for unified dimensionality reduction
        all_vectors = np.vstack([cbow_vectors, sg_vectors])
        
        # Dimensionality reduction
        if True:
            embedded_all = self.apply_tsne(all_vectors)
        else:
            embedded_all = self.apply_pca(all_vectors)
        
        # Split back to two models
        n_words = len(cbow_vectors)
        cbow_embedded = embedded_all[:n_words]
        sg_embedded = embedded_all[n_words:]
        
        # Create comparison plot
        plt.switch_backend('Agg')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.args.figsize[0]*1.5, self.args.figsize[1]))
        
        # CBOW plot
        ax1.scatter(cbow_embedded[:, 0], cbow_embedded[:, 1], alpha=0.7, 
                   c='blue', s=50, edgecolors='white', linewidth=0.5)
        ax1.set_title('CBOW Model Word Embeddings', fontsize=14, fontweight='bold')
        ax1.set_xlabel(f'{method.upper()} Dimension 1')
        ax1.set_ylabel(f'{method.upper()} Dimension 2')
        ax1.grid(True, alpha=0.3)
        
        # Skip-gram plot
        ax2.scatter(sg_embedded[:, 0], sg_embedded[:, 1], alpha=0.7, 
                   c='red', s=50, edgecolors='white', linewidth=0.5)
        ax2.set_title('Skip-gram Model Word Embeddings', fontsize=14, fontweight='bold')
        ax2.set_xlabel(f'{method.upper()} Dimension 1')
        ax2.set_ylabel(f'{method.upper()} Dimension 2')
        ax2.grid(True, alpha=0.3)
        
        # Annotate some important words
        important_words = ['good', 'bad', 'excellent', 'terrible', 'product', 'quality', 
                          'great', 'amazing', 'poor', 'fast', 'slow']
        
        for i, word in enumerate(cbow_labels):
            if word in important_words and i < len(cbow_embedded):
                ax1.annotate(word, (cbow_embedded[i, 0], cbow_embedded[i, 1]), 
                           fontsize=8, alpha=0.8)
                ax2.annotate(word, (sg_embedded[i, 0], sg_embedded[i, 1]), 
                           fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        
        # Save image
        os.makedirs('Part4/comparisons', exist_ok=True)
        filename = f'Part4/comparisons/word2vec_models_comparison_{method}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model comparison plot saved: {filename}")
    
    def create_semantic_clusters_analysis(self, model, model_name, category_words, method='tsne'):
        """Create semantic clustering analysis"""
        print(f"\nCreating {model_name} semantic clustering analysis...")
        
        # Prepare data
        all_words = []
        categories = []
        
        for category, words in category_words.items():
            all_words.extend(words)
            categories.extend([category] * len(words))
        
        vectors, labels, valid_words = self.prepare_word_vectors(model, all_words)
        
        if len(vectors) == 0:
            print("No valid word vectors for clustering analysis")
            return
        
        # Dimensionality reduction
        if True:
            embedded = self.apply_tsne(vectors)
        else:
            embedded = self.apply_pca(vectors)
        
        # Create visualization
        plt.switch_backend('Agg')
        plt.figure(figsize=self.args.figsize)
        
        # Set colors for each category
        unique_categories = list(category_words.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))
        color_map = dict(zip(unique_categories, colors))
        
        # Plot points for each category
        for category in unique_categories:
            mask = [cat == category for cat in categories[:len(embedded)]]
            if any(mask):
                plt.scatter(embedded[mask, 0], embedded[mask, 1], 
                           c=[color_map[category]], label=category, 
                           s=80, alpha=0.8, edgecolors='white', linewidth=0.5)
        
        # Add labels
        for i, (x, y) in enumerate(embedded):
            # Only annotate some words to avoid overcrowding
            if i % max(1, len(embedded) // self.args.annotate) == 0:
                plt.annotate(labels[i], (x, y), fontsize=9, alpha=0.9,
                            bbox=dict(boxstyle="round,pad=0.2", fc='white', alpha=0.7))
        
        plt.title(f'{model_name} Semantic Clustering Analysis ({method.upper()})', fontsize=16, fontweight='bold')
        plt.xlabel(f'{method.upper()} Dimension 1')
        plt.ylabel(f'{method.upper()} Dimension 2')
        plt.legend(title='Semantic Categories', loc='best')
        plt.grid(True, alpha=0.3)
        
        # Save image
        os.makedirs('Part4/semantic_analysis', exist_ok=True)
        filename = f'Part4/semantic_analysis/{model_name.lower()}_semantic_clusters_{method}.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Semantic clustering analysis plot saved: {filename}")
        
        return embedded, labels, categories
    
    
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Experiment 4: Vector Representation Visualization Analysis')
    
    # Data size parameters
    parser.add_argument('--word-size', '-ws', type=int, default=200,
                       help='Number of words to visualize (default: 200)')
    parser.add_argument('--node-size', '-ns', type=int, default=300,
                       help='Number of nodes to visualize (default: 300)')
    
    # Visualization parameters
    parser.add_argument('--annotate', '-a', type=int, default=20,
                       help='Number of annotations (default: 20)')
    parser.add_argument('--figsize', '-f', type=int, nargs=2, default=[15, 12],
                       help='Figure size, format: width height (default: 15 12)')
    parser.add_argument('--method', '-m', choices=['tsne'], default='tsne',
                       help='Dimensionality reduction method: tsne, pca, or both (default: both)')
    
    # Control parameters
    parser.add_argument('--show-plots', '-s', action='store_true',
                       help='Show plots (default: do not show, only save)')
    parser.add_argument('--skip-word2vec', action='store_true',
                       help='Skip Word2Vec visualization')
    parser.add_argument('--skip-node2vec', action='store_true',
                       help='Skip Node2Vec visualization')
    parser.add_argument('--skip-comparison', action='store_true',
                       help='Skip model comparison')
    parser.add_argument('--skip-semantic', action='store_true',
                       help='Skip semantic clustering analysis')
    
    return parser.parse_args()

def main():
    print("Experiment 4: Data Visualization using T-SNE")
    print("=" * 50)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create visualization instance
    visualizer = VectorVisualization(args)
    
    # Load models
    visualizer.load_models()
    
    # Ensure output directory exists
    os.makedirs('Part4', exist_ok=True)
    
    # Determine methods to use
    methods = []
    methods = ['tsne']
    
    # 1. Word2Vec word embeddings visualization
    if not args.skip_word2vec:
        if visualizer.word2vec_cbow:
            print("\n--- CBOW Model Visualization ---")
            for method in methods:
                visualizer.visualize_word_embeddings(
                    visualizer.word2vec_cbow, 
                    'CBOW', 
                    method=method
                )
        
        if visualizer.word2vec_sg:
            print("\n--- Skip-gram Model Visualization ---")
            for method in methods:
                visualizer.visualize_word_embeddings(
                    visualizer.word2vec_sg, 
                    'Skip-gram', 
                    method=method
                )
    
    # 2. Node2Vec node embeddings visualization
    if not args.skip_node2vec and visualizer.node2vec_model:
        print("\n--- Node2Vec Node Visualization ---")
        for method in methods:
            visualizer.visualize_node_embeddings(method=method)
    
    # 3. Model comparison
    if not args.skip_comparison and visualizer.word2vec_cbow and visualizer.word2vec_sg:
        print("\n--- Model Comparison ---")
        for method in methods:
            visualizer.compare_word_models(method=method)
    
    # 4. Semantic clustering analysis
    if not args.skip_semantic and visualizer.word2vec_sg:
        print("\n--- Semantic Clustering Analysis ---")
        # Define semantic categories
        category_words = {
            'Positive Sentiment': ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'perfect'],
            'Negative Sentiment': ['bad', 'terrible', 'awful', 'horrible', 'poor', 'disappointing'],
            'Product Quality': ['quality', 'durable', 'reliable', 'sturdy', 'well_made'],
            'Delivery Service': ['fast', 'quick', 'slow', 'delivery', 'shipping', 'arrived'],
            'Price Related': ['expensive', 'cheap', 'affordable', 'price', 'cost', 'value'],
            'Electronic Products': ['computer', 'laptop', 'phone', 'tablet', 'device', 'electronic'],
            'Book Related': ['book', 'novel', 'story', 'author', 'reading', 'page']
        }
        
        for method in methods:
            visualizer.create_semantic_clusters_analysis(
                visualizer.word2vec_sg,
                'Skip-gram',
                category_words,
                method=method
            )
    
    
    print("\n" + "=" * 50)
    print("Experiment 4 Completed: Data Visualization Analysis!")
    print(f"All visualization results saved to Part4/ directory")
    print(f"Parameters: word_size={args.word_size}, node_size={args.node_size}, annotate={args.annotate}")

if __name__ == "__main__":
    main()