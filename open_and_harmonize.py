import json
import pandas as pd
import os

def clean_and_harmonize_recursive(base_path="Datasets"):
    all_data = []
    
    # os.walk ile tüm alt klasörleri dolaşıyoruz
    for root, dirs, files in os.walk(base_path):
        for file_name in files:
            if not file_name.endswith('.json'):
                continue
                
            # Tam dosya yolunu oluştur
            file_path = os.path.join(root, file_name)
            
            # Klasör veya dosya isminden hangi veri seti olduğunu anla
            # Örn: ArMIS_Dataset/train.json veya ArMIS_train.json
            full_path_lower = file_path.lower()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for key, item in data.items():
                # 1. Metin İşleme (ConvAbuse için özel durum)
                if "convabuse" in full_path_lower:
                    try:
                        text_dict = json.loads(item['text']) if isinstance(item['text'], str) else item['text']
                        text = text_dict.get('user', '')
                    except:
                        text = item['text']
                else:
                    text = item['text']
                
                # 2. Etiket Normalizasyonu
                raw_annotations = [int(x) for x in str(item['annotations']).split(',')]
                
                if "convabuse" in full_path_lower:
                    # -3, -2, -1 abusivedir (1); 0, 1 değildir (0)
                    clean_annotations = [1 if x < 0 else 0 for x in raw_annotations]
                else:
                    clean_annotations = raw_annotations
                
                # 3. Anlaşmazlık (Disagreement) Hesaplama
                is_disagreement = 1 if len(set(clean_annotations)) > 1 else 0
                
                all_data.append({
                    'id': f"{key}",
                    'text': text,
                    'is_disagreement': is_disagreement,
                    'soft_label_1': float(item.get('soft_label', {}).get('1', 0)),
                    'language': item.get('lang', 'en'),
                    'split': item.get('split', 'train'), # Split bilgisini koruyoruz
                    'dataset': os.path.basename(root) # Veri setini klasör adından alıyoruz
                })
                
    return pd.DataFrame(all_data)

# Veriyi topla
df_final = clean_and_harmonize_recursive()

# Dosyaları split (train/dev/test) bazında ayırarak kaydet
for split_name in ['train', 'dev', 'test']:
    split_df = df_final[df_final['split'] == split_name]
    if not split_df.empty:
        split_df.to_csv(f"Datasets/harmonized_{split_name}.csv", index=False, encoding='utf-8')
        print(f"✅ {split_name} dosyası hazırlandı ({len(split_df)} satır)")
print(len(df_final), "toplam satır harmonize edildi.")