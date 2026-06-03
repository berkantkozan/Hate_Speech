import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Veriyi Yükle
df = pd.read_csv("Datasets/validation_scores.csv")
X = df[['sds_mean', 'sds_max', 'sds_sum']]
y = df['actual_disagreement']

# 2. Model Eğitimi
clf = LogisticRegression()
clf.fit(X, y)
y_pred = clf.predict(X)
y_probs = clf.predict_proba(X)[:, 1]

# 3. Raporu Metin Olarak Hazırla
accuracy = accuracy_score(y, y_pred)
report = classification_report(y, y_pred)
importance = clf.coef_[0]

report_text = f"""--- PERFORMANS RAPORU ---
Genel Doğruluk (Accuracy): %{accuracy*100:.2f}
ROC-AUC Skoru: {roc_auc_score(y, y_probs):.4f}

Sınıflandırma Detayları:
{report}

Strateji Ağırlıkları:
Mean: {importance[0]:.4f}
Max: {importance[1]:.4f}
Sum: {importance[2]:.4f}
"""

# --- KAYIT İŞLEMLERİ ---

# 1. Raporu .txt dosyasına kaydet
with open("Datasets/final_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)
print("✅ Performans raporu 'final_report.txt' olarak kaydedildi.")

# 2. Confusion Matrix grafiğini .png olarak kaydet
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Anlaşma', 'Anlaşmazlık'], 
            yticklabels=['Anlaşma', 'Anlaşmazlık'])
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değer')
plt.title(f'Hata Matrisi (Accuracy: %{accuracy*100:.2f})')

plt.savefig("Datasets/confusion_matrix.png", dpi=300) # Yüksek kalite kayıt
print("✅ Hata matrisi grafiği 'confusion_matrix.png' olarak kaydedildi.")

plt.show() # Yine de ekranda gör