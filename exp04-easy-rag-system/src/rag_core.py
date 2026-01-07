import streamlit as st
import torch
from config import MAX_NEW_TOKENS_GEN, TEMPERATURE, TOP_P, REPETITION_PENALTY
from sentence_transformers import CrossEncoder
import streamlit as st

# 加载 Re-ranker 模型 (建议缓存)
@st.cache_resource
def load_reranker_model(model_name="BAAI/bge-reranker-base"):
    try:
        # BGE-Reranker 是目前效果较好的开源模型
        reranker = CrossEncoder(model_name, max_length=512)
        return reranker
    except Exception as e:
        st.error(f"Failed to load reranker: {e}")
        return None

def rerank_documents(query, retrieved_docs, reranker_model, top_k=3):
    """
    对检索到的文档进行重排序
    """
    if not retrieved_docs or not reranker_model:
        return retrieved_docs[:top_k] # 降级处理：直接返回前k个

    # 准备 pairs: [[query, doc1], [query, doc2], ...]
    pairs = [[query, doc['content']] for doc in retrieved_docs]
    
    # 计算得分
    scores = reranker_model.predict(pairs)
    
    # 将得分附加到文档中并排序
    for doc, score in zip(retrieved_docs, scores):
        doc['rerank_score'] = score
    
    # 按分数降序排列
    sorted_docs = sorted(retrieved_docs, key=lambda x: x['rerank_score'], reverse=True)
    
    return sorted_docs[:top_k]

def generate_answer(query, context_docs, gen_model, tokenizer):
    """Generates an answer using the LLM based on query and context."""
    if not context_docs:
        return "I couldn't find relevant documents to answer your question."
    if not gen_model or not tokenizer:
         st.error("Generation model or tokenizer not available.")
         return "Error: Generation components not loaded."

    context = "\n\n---\n\n".join([doc['content'] for doc in context_docs]) # Combine retrieved docs

    prompt = f"""Based ONLY on the following context documents, answer the user's question.
If the answer is not found in the context, state that clearly. Do not make up information.

Context Documents:
{context}

User Question: {query}

Answer:
"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(gen_model.device)
        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_GEN,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.eos_token_id # Important for open-end generation
            )
        # Decode only the newly generated tokens, excluding the prompt
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        st.error(f"Error during text generation: {e}")
        return "Sorry, I encountered an error while generating the answer." 