import torch
import pandas as pd
import numpy as np
import pickle
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# 1. GPU Hazırlığı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 İşlem birimi: {device}")

# 2. Hafızayı (Train Vectors) GPU'ya Yükle
with open("Datasets/train_embeddings.pkl", "rb") as f:
    train_data = pickle.load(f)

# Vektörleri PyTorch Tensor'a çevirip GPU'ya atıyoruz
train_vectors = torch.tensor(np.array([item['vector'] for item in train_data])).to(device)
train_labels = torch.tensor(np.array([item['label'] for item in train_data])).to(device)

# 3. Model Hazırlığı (BERT)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertModel.from_pretrained("bert-base-multilingual-cased").to(device)
model.eval()

# 4. Mesafe (Benzerlik) Ağırlıklı K-NN Skor Hesaplama Fonksiyonu (GPU'da)
def gpu_weighted_knn_ds_score(query_vec, k=15):
    """
    query_vec: (1, 768) -> GPU üzerinde
    train_vectors: (N, 768) -> GPU üzerinde
    train_labels: (N,) -> GPU üzerinde
    
    Ağırlıklandırma: Kosinüs benzerliği doğrudan ağırlık olarak kullanılır.
    Sıfıra bölme hatasını engellemek için küçük bir epsilon eklenmiştir.
    """
    # Kosinüs Benzerliği Hesapla
    cos = torch.nn.CosineSimilarity(dim=1)
    similarities = cos(train_vectors, query_vec) # Ölçü: [-1, 1] arası
    
    # Negatif benzerlikleri sıfırlamak veya ölçeklemek mantıklı olabilir. 
    # Genellikle BERT embedding'lerinde pozitif bölgede çalışılır ancak güvenli kılmak için:
    similarities = torch.clamp(similarities, min=0.0)
    
    # En yakın K komşuyu ve onların benzerlik skorlarını (ağırlıklarını) bul
    topk_vals, topk_indices = torch.topk(similarities, k=k)
    
    # Sıfıra bölme riskine karşı epsilon koruması
    eps = 1e-8
    weights = topk_vals + eps 
    
    # İlgili komşuların etiketlerini al
    labels = train_labels[topk_indices]
    
    # Ağırlıklı Ortalama Hesaplama: Formül -> sum(w_i * y_i) / sum(w_i)
    weighted_sum = torch.sum(weights * labels)
    total_weight = torch.sum(weights)
    
    ds_score = weighted_sum / total_weight
    return ds_score.item()

def fast_process(csv_path, k_val=15):
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
            
            # Güncellenmiş ağırlıklı fonksiyon çağrısı ve K optimizasyon parametresi
            ds_w = gpu_weighted_knn_ds_score(avg_vec, k=k_val)
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

# Çalıştır ve K değerini optimize et (Doğrulama setine göre k=15 veya k=25 denenebilir)
df_val_results = fast_process("Datasets/harmonized_dev.csv", k_val=5)
df_val_results.to_csv("Datasets/validation_scores.csv", index=False)
print("✅ İşlem tamamlandı ve skorlar kaydedildi.")