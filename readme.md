Markdown
## 🚀 Kurulum ve Kullanım Rehberi

Projeyi kendi bilgisayarınızda çalıştırmak ve elde edilen sonuçları (Baseline ve SOTA) yeniden üretmek için aşağıdaki adımları sırasıyla takip ediniz. Proje, yoğun GPU kullanımı gerektiren vektör işlemleri içerdiğinden CUDA destekli bir donanım önerilmektedir.

### Adım 1: Repoyu Klonlama ve Ortam Hazırlığı

Terminalinizi açın ve projeyi bilgisayarınıza indirin:
```bash
git clone [https://github.com/berkantkozan/Hate_Speech.git](https://github.com/berkantkozan/Hate_Speech.git)
cd Hate_Speech
Projeye özel izole bir sanal ortam (virtual environment) oluşturun ve aktif edin:

Bash
python -m venv venv

# Windows kullanıcıları için:
venv\Scripts\activate

# Linux / Mac kullanıcıları için:
source venv/bin/activate
Gerekli tüm kütüphaneleri (PyTorch, Transformers, Imbalanced-learn, XGBoost, Pandas vb.) kurun:

Bash
pip install -r requirements.txt
Sistemin GPU'yu (CUDA) başarıyla tanıyıp tanımadığını test etmek için ön kontrol scriptini çalıştırın:

Bash
python gputest.py
Adım 2: Model Eğitim Akışı (Pipeline)
Projedeki Python scriptleri, ham verinin işlenmesinden derin öğrenme modelinin eğitilmesine kadar hiyerarşik bir sırayla çalıştırılacak şekilde tasarlanmıştır.

1. Veri Harmonizasyonu (Data Preprocessing):
Farklı formatlardaki ham JSON veri setlerini (örn. ConvAbuse) okuyup etiketleri normalize eden ve veriyi train/dev/test olarak ayıran scripti çalıştırın. Bu adım Datasets klasörü içine harmonized_*.csv dosyalarını üretecektir.

Bash
python open_and_harmonize.py
2. Bağlamsal Gömü Çıkarımı (Embedding Extraction):
mBERT modelini kullanarak harmonize edilmiş eğitim verisindeki kelimelerden 768 boyutlu vektörleri çıkarın. Bu işlem sonucunda train_embeddings.pkl dosyası oluşacaktır.

Bash
python embed.py
3. Doğrusal Sınıflandırma ve Baseline Modeller (Opsiyonel):
Kelime toplulaştırma stratejilerini (max, mean, sum) test etmek için temel makine öğrenmesi modellerini çalıştırabilirsiniz. Çalıştırmak istediğiniz yaklaşıma göre aşağıdaki scriptlerden birini seçin (Skorlar validation_scores.csv olarak kaydedilir):

Ridge Regresyon için: python ds_hesapla_LR.py

Uzaklık Ağırlıklı (Distance-Weighted) K-NN için: python ds_hesapla_wknn.py

Standart K-NN için: python ds_hesapla.py

4. Baseline Model Değerlendirmesi:
3. adımda üretilen validation_scores.csv dosyasındaki skorları analiz etmek, strateji ağırlıklarını belirlemek ve Hata Matrisi (Confusion Matrix) üretmek için değerlendirme scriptini çalıştırın:

Bash
python evaluation.py
5. Nihai Derin Öğrenme Modeli (State-of-the-Art):
Projenin ana odak noktası olan, eğitim verisini SMOTE ile fiziksel olarak dengeleyen ve 3 katmanlı MLP (Çok Katmanlı Algılayıcı) sinir ağını GPU üzerinde eğiten ana dosyayı çalıştırın. En yüksek doğruluk ve Makro F1 sonuçlarını bu script üzerinden alacaksınız.

Bash
python mlp.py
6. Sonuçların Görselleştirilmesi:
MLP modelinin eğitim sürecindeki pürüzsüz kayıp (loss) azalışını akademik formatta bir grafiğe (mlp_loss_egrisi.png) dönüştürmek için:

Bash
python loss_eğrisi.py