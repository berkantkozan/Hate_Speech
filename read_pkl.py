import pandas as pd

# Dosya yolunu belirterek veriyi yükle
file_path = 'Datasets/train_embeddings.pkl'
data = pd.read_pickle(file_path)

# Veri bir liste olduğu için ilk 5 elemanı dilimleyerek (slicing) alıyoruz
sample_data = data[:5]

# Sonucu ekrana yazdır
for i, example in enumerate(sample_data):
    print(f"--- Örnek {i+1} ---")
    print(example)