import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import pickle

# 1. Donanım ve Model Hazırlığı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_hidden_states=True).to(device)
model.eval()

def get_embeddings(csv_path):
    df = pd.read_csv(csv_path)
    word_data_list = [] # Her kelime için: {vektör, soft_label, word_text}

    print(f"🔄 {csv_path} işleniyor...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row['text'])
        soft_label = row['soft_label_1'] # Anlaşmazlık/Nefret olasılığı
        
        # Tokenizasyon ve kelime hizalama
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
        word_ids = inputs.word_ids() # Tokenların hangi kelimeye ait olduğu bilgisi
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Son gizli katmanı (last hidden state) alıyoruz 
            embeddings = outputs.last_hidden_state[0] 
        
        # Kelime parçalarını (subwords) birleştirme mantığı
        temp_word_map = {}
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        for idx, word_id in enumerate(word_ids):
            if word_id is None: continue # [CLS] ve [SEP] gibi özel işaretleri atla
            
            if word_id not in temp_word_map:
                temp_word_map[word_id] = [embeddings[idx]]
            else:
                temp_word_map[word_id].append(embeddings[idx])
        
        # Her kelime için ortalama vektörü hesapla ve listeye ekle
        for word_id, vecs in temp_word_map.items():
            avg_vec = torch.stack(vecs).mean(dim=0).cpu().numpy()
            word_data_list.append({
                'vector': avg_vec,
                'label': soft_label,
                'dataset': row['dataset']
            })
            
    return word_data_list

# İşlemi başlat ve kaydet
train_embeddings = get_embeddings("Datasets/harmonized_train.csv")

# Büyük veriyi hızlı yüklemek için pickle kullanıyoruz
with open("Datasets/train_embeddings.pkl", "wb") as f:
    pickle.dump(train_embeddings, f)

print(f"✅ Toplam {len(train_embeddings)} kelime vektörü başarıyla kaydedildi.")