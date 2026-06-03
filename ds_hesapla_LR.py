import torch
import pandas as pd
import numpy as np
import pickle
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.linear_model import Ridge

# 1. GPU Hazırlığı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 İşlem birimi: {device}")

# 2. Hafızayı (Train Vectors) Yükle ve Lojistik Regresyonu Eğit
with open("Datasets/train_embeddings.pkl", "rb") as f:
    train_data = pickle.load(f)

# Veri yükleme adımları aynı...
X_train = np.array([item['vector'] for item in train_data])
y_train = np.array([item['label'] for item in train_data])

# Farklı Alpha değerlerini test ediyoruz
alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
best_alpha = 1.0
best_model = None

# Not: Eğer validation setin tamamen ayrı bir CSV ise döngüyü fast_process skoruna göre kurabilirsin.
# Burada eğitim seti üzerindeki kararlılığa bakarak hızlı bir seçim yapıyoruz:
print("⏳ En iyi Alpha değeri aranıyor...")
for a in alphas:
    reg_model = Ridge(alpha=a)
    reg_model.fit(X_train, y_train)
    # Modelin R^2 skoruna veya validation başarısına göre seçebilirsiniz
    print(f"Alpha: {a:<5} | Eğitim Seti Skoru: {reg_model.score(X_train, y_train):.4f}")

# Nihai karar verdiğin alpha ile modeli sabitle:
final_alpha = 1.0  # Döngü sonucuna göre burayı güncelleyebilirsin
reg_model = Ridge(alpha=final_alpha)
reg_model.fit(X_train, y_train)

# Katsayıları tekrar GPU'ya taşıyoruz
lr_weights = torch.tensor(reg_model.coef_, dtype=torch.float32).to(device).unsqueeze(0)
lr_bias = torch.tensor(reg_model.intercept_, dtype=torch.float32).to(device)

# 3. Model Hazırlığı (BERT)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertModel.from_pretrained("bert-base-multilingual-cased").to(device)
model.eval()

def gpu_logistic_regression_score(query_vec):
    with torch.no_grad():
        # Ridge regresyon doğrusal çıktı üretir: Xw + b
        prediction = torch.matmul(query_vec, lr_weights.t()) + lr_bias
        
        # Çıktıyı [0, 1] arasına sıkıştır (clip/clamp)
        # çünkü regresyon modelleri bazen 0'ın altında veya 1'in üstünde tahmin üretebilir.
        prediction = torch.clamp(prediction, min=0.0, max=1.0)
        
    return prediction.item()

def fast_process(csv_path):
    df = pd.read_csv(csv_path)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row['text'])
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
        word_ids = inputs.word_ids()
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[0]
        
        word_scores = []
        temp_vecs = {}
        
        for idx, w_id in enumerate(word_ids):
            if w_id is None: continue
            if w_id not in temp_vecs: temp_vecs[w_id] = [embeddings[idx]]
            else: temp_vecs[w_id].append(embeddings[idx])
            
        for w_id in temp_vecs:
            avg_vec = torch.stack(temp_vecs[w_id]).mean(dim=0).unsqueeze(0) # (1, 768)
            
            # Lojistik Regresyon skoru çağrılıyor
            ds_w = gpu_logistic_regression_score(avg_vec)
            word_scores.append(ds_w)
            
        if word_scores:
            results.append({
                'id': row['id'],
                'actual_disagreement': row['is_disagreement'],
                'sds_mean': np.mean(word_scores),
                'sds_max': np.max(word_scores),
                'sds_sum': np.sum(word_scores)
            })
            
    return pd.DataFrame(results)

# Çalıştır
df_val_results = fast_process("Datasets/harmonized_dev.csv")
df_val_results.to_csv("Datasets/validation_scores.csv", index=False)
print("✅ Tüm süreç tamamlandı ve Lojistik Regresyon skorları kaydedildi.")