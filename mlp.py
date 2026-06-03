import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# 1. BERT Kelime Vektörlerini Yükleme
print("🔄 Vektörler yükleniyor...")
with open("Datasets/train_embeddings.pkl", "rb") as f:
    word_data_list = pickle.load(f)

# 2. Veriyi Hazırlama ve Hipoteze Uygun Etiketleme (Binarization)
X = np.array([item['vector'] for item in word_data_list])
soft_labels = np.array([item['label'] for item in word_data_list])

# Eşik değer: 0.5 ve üzeri anlaşmazlık (1), altı anlaşmazlık yok (0)
# Bu eşiği verinizin yapısına göre 0.4 veya 0.6 olarak da deneyebilirsiniz.
y = np.where(soft_labels >= 0.5, 1, 0)

print(f"📊 Toplam Kelime Sayısı: {len(X)}")
print(f"⚖️ Sınıf Dağılımı -> Anlaşmazlık Yok (0): {np.sum(y==0)}, Anlaşmazlık Var (1): {np.sum(y==1)}")

# 3. Eğitim ve Test Ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Hafif ve Hızlı Modelin (XGBoost) Eğitilmesi
print("🚀 Karmaşık olmayan model (XGBoost) eğitiliyor...")
model = XGBClassifier(
    n_estimators=100, 
    max_depth=6, 
    learning_rate=0.1, 
    random_state=42, 
    tree_method="hist" # Hızlı eğitim için
)
model.fit(X_train, y_train)

# 5. Hipotez Kontrolü (Doğruluk Ölçümü)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*40)
print(f"🎯 MODEL DOĞRULUK ORANI (ACCURACY): %{accuracy * 100:.2f}")
print("="*40)

if accuracy >= 0.80:
    print("🎉 TEBRİKLER! Hipoteziniz doğrulandı. Karmaşık modellere ihtiyaç duymadan %80 barajını geçtiniz.")
else:
    print("📉 Doğruluk %80'in altında kaldı. Eşik değerini (threshold) değiştirmeyi veya hiperparametreleri ayarlamayı deneyebilirsiniz.")

print("\n📝 Detaylı Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=["Anlaşmazlık Yok", "Anlaşmazlık Var"]))