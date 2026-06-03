"""# 🔍 Hate Speech & Annotator Disagreement Detection

Bu depo, nefret söylemi ve ofansif metinlerde veri etiketleyicileri arasındaki görüş ayrılıklarını (**annotator disagreement**) kelime (token) düzeyinde tespit etmek ve modellemek amacıyla geliştirilmiş, uçtan uca (end-to-end) bir makine öğrenmesi ve derin öğrenme (Deep Learning) veri hattını (pipeline) içermektedir.

Geleneksel sınıflandırma sistemlerinin "çoğunluk oyu" (majority vote) yaklaşımını geride bırakan bu çalışma, **perspektifçilik (perspectivism)** ilkesini benimseyerek metinlerdeki insan öznelliğini ve dilbilimsel belirsizlikleri korumayı amaçlar. Milyarlarca parametreli Büyük Dil Modellerinin (LLM) devasa hesaplama maliyetleri ve donanım bağımlılıklarını aşmak adına, çok daha hafif fakat **State-of-the-Art (SOTA)** performansa sahip bağlamsal kelime gömüsü tabanlı derin sinir ağları tasarlanmıştır.

---

## 📂 Proje Yapısı ve Dosya Açıklamaları

Depo içerisindeki kaynak kodlar ve üstlendikleri görevler aşağıda kronolojik sırasıyla açıklanmıştır:

1.  **`gputest.py`**: Sistemdeki grafik kartının (CUDA) PyTorch tarafından başarıyla tanınıp tanınmadığını ve donanım adını (`NVIDIA GeForce RTX 4050` vb.) doğrulayan ön kontrol scriptidir.
2.  **`open_and_harmonize.py`**: `Datasets/` altındaki farklı formatlara sahip ham JSON veri setlerini (örn. ConvAbuse, ArMIS) hiyerarşik olarak tarar. Metinleri temizler, etiketleri normalize eder, annotator'lar arasında fikir ayrılığı olup olmadığını hesaplar (`is_disagreement`) ve veriyi `train`, `dev`, `test` splitlerine ayırarak uyumlu CSV dosyaları üretir.
3.  **`embed.py`**: Harmonize edilmiş metinleri `bert-base-multilingual-cased` (mBERT) tokenizer'ı ile işler. Token'ların hangi kelimelere ait olduğunu (`word_ids()`) takip ederek alt kelimeleri (subwords) birleştirir ve her kelime için bağlamı koruyan **768 boyutlu yoğun vektörler** çıkararak `train_embeddings.pkl` olarak kaydeder.
4.  **`ds_hesapla.py`**: Standart **K-En Yakın Komşu (K-NN)** tabanlı baseline modelidir. Gömü uzayındaki kelimelerin kosinüs benzerliklerini GPU üzerinde matris çarpımıyla hesaplar ve en yakın $K$ komşunun etiket ortalamasını alarak disagreement skoru üretir.
5.  **`ds_hesapla_wknn.py`**: Gelişmiş **Uzaklık Ağırlıklı K-NN (Distance-Weighted K-NN)** modelidir. Komşuların kosinüs benzerlik skorlarını doğrudan ağırlık faktörü olarak kullanır ve $\frac{\sum (w_i \cdot y_i)}{\sum w_i}$ formülüyle ağırlıklı kararlar üretir. Sıfıra bölme hatalarına karşı epsilon korumalıdır.
6.  **`ds_hesapla_LR.py`**: 768 boyutlu vektörler üzerinde `SVD` çözücü kullanarak doğrusal bir **Ridge Regresyon** modeli eğitir. Öğrenilen katsayıları ve intercept değerini GPU tensorlerine taşıyarak doğrulamayı son derece hızlı matris çarpımlarıyla tamamlar ve çıktıları $[0.0, 1.0]$ arasına sınırlar.
7.  **`evaluation.py`**: Baseline modeller tarafından üretilen `sds_mean`, `sds_max` ve `sds_sum` toplulaştırma stratejilerini girdi kabul ederek bir Lojistik Regresyon eğitir. Strateji ağırlıklarını raporlar, doğruluk ve ROC-AUC skorlarını hesaplayarak `final_report.txt` ve yüksek çözünürlüklü `confusion_matrix.png` grafiğini kaydeder.
8.  **`mlp.py`**: Projenin nihai derin öğrenme mimarisidir. Sınıf dengesizliğini çözmek için eğitim setine **SMOTE** uygular. PyTorch kullanarak Batch Normalization ve Dropout korumalı 3 katmanlı derin bir Yapay Sinir Ağı (MLP) eğitir ve test seti üzerinde SOTA sonuçları üretir.
9.  **`loss_eğrisi.py`**: MLP modelinin 15 epokluk eğitim süreci boyunca sergilediği kayıp (BCE Loss) azalışını `matplotlib` ve `seaborn` kullanarak akademik yayın standartlarında çizgi grafiğine (`mlp_loss_egrisi.png`) dönüştürür.

---

## 🧬 Veri Seti ve Sınıf Dengeleme (SMOTE)

Projede işlenen kelime havuzunun doğal dağılımı yüksek derecede dengesiz bir yapıya sahiptir:
* **Toplam Kelime Sayısı:** 242.040
* **Orijinal Eğitim Dağılımı:** Anlaşmazlık Yok (0): 134.065 | Anlaşmazlık Var (1): 59.567

**Doğruluk Paradoksu (Accuracy Paradox):** Ağırlıklandırılmamış modeller bu dengesizlik sebebiyle çoğunluk sınıfını ezberleme eğilimi göstermiş ve "Anlaşmazlık Var" sınıfını yakalama oranı (Recall) %17'de kalmıştır. 

**Çözüm:** Veri sızıntısını (**data leakage**) kesin olarak önlemek adına, **SMOTE (Synthetic Minority Over-sampling Technique) algoritması yalnızca eğitim setine uygulanmıştır.** Test setinin orijinal dokusu korunurken, eğitim havuzu her iki sınıf için eşit olacak şekilde **134.065** sentetik vektörle kusursuz bir dengeye ulaştırılmıştır.

---

## 🧠 Derin Öğrenme Mimarisi (MLP)

768 boyutlu BERT gömülerinin doğrusal olmayan karmaşık ilişkilerini çözümlemek için PyTorch ile inşa edilen Çok Katmanlı Algılayıcı (MLP) ağı şu katman dizilimine sahiptir: