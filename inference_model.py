"""
Script Inference untuk Model Analisis Sentimen Gojek
Menggunakan DistilBERT yang sudah di-training

Usage:
    python inference_model.py
"""

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi preprocessing tools
stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

def clean_text(text):
    """Cleaning text"""
    if not text:
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text):
    """Preprocessing lengkap"""
    text = clean_text(text)
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text

class SentimentPredictor:
    """Class untuk prediksi sentimen"""
    
    def __init__(self, model_path='./sentiment_model_distilbert'):
        """
        Initialize predictor
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥  Loading model dengan device: {self.device}")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device).eval()
        
        self.label_map = {0: 'negatif', 1: 'netral', 2: 'positif'}
        
        print("✅ Model berhasil dimuat!")
    
    def predict(self, text, show_probabilities=True):
        """Prediksi sentimen"""        
        text_clean = preprocess_text(text)

        encoding = self.tokenizer.encode_plus(
            text_clean,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
        
        result = {
            'text': text,
            'sentiment': self.label_map[pred_class],
            'confidence': probs[0][pred_class].item()
        }
        
        if show_probabilities:
            result['probabilities'] = {
                'negatif': probs[0][0].item(),
                'netral': probs[0][1].item(),
                'positif': probs[0][2].item(),
            }
        
        return result

def print_result(result):
    """Print hasil dalam format rapi"""
    print("="*60)
    print(f"📝 Review: {result['text']}")
    print(f"🎯 Sentimen: {result['sentiment'].upper()}")
    print(f"💯 Confidence: {result['confidence']*100:.2f}%")
    
    if 'probabilities' in result:
        print("\n📊 Probabilitas:")
        for sentiment, prob in result['probabilities'].items():
            bar = '█' * int(prob * 50)
            print(f"   {sentiment.capitalize():8} | {bar} {prob*100:.1f}%")
    print("="*60)

def main():
    print("="*60)
    print("🚀 SENTIMENT ANALYSIS - GOJEK REVIEWS")
    print("="*60)

    predictor = SentimentPredictor()

    test_reviews = [
        "Aplikasi gojek sangat membantu, driver ramah dan cepat sampai tujuan",
        "Aplikasi jelek, tidak recommended",
        "Biasa aja sih, tidak ada yang spesial",
        "MANTAP BANGET! Gojek emang terbaik, pelayanan cepat dan harga terjangkau",
        "Aplikasi lemot banget, mau order susah, driver juga lama datangnya",
    ]
    
    print("\n🧪 Testing dengan 5 contoh review:\n")
    
    for i, review in enumerate(test_reviews, 1):
        print(f"\n--- REVIEW {i} ---")
        print_result(predictor.predict(review))
    
    print("\n" + "="*60)
    print("💬 MODE INTERAKTIF")
    print("="*60)
    print("Ketik review Gojek untuk dianalisis.")
    print("Ketik 'exit' untuk keluar.\n")
    
    while True:
        user_input = input("📝 Review: ").strip()
        
        if user_input.lower() == 'exit':
            print("\n👋 Terima kasih! Sampai jumpa!")
            break
        
        if not user_input:
            print("⚠  Review tidak boleh kosong!\n")
            continue
        
        print()
        print_result(predictor.predict(user_input))
        print()

if __name__ == "__main__":
    main()
