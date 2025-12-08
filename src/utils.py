import re
import nltk

# 如第一次使用 nltk，要取消注释下面这行：
nltk.download('punkt')

def clean_text(text):
    if text is None:
        return ""
    s = str(text)
    s = re.sub(r'http\S+', ' ', s)
    s = re.sub(r'<.*?>', ' ', s)
    s = re.sub(r'[^0-9A-Za-z\s\.,!?;:()\'"-]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()

def tokenize(text):
    return nltk.word_tokenize(text)
