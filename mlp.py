import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np
import pickle

# 1. GPU Hazırlığı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 İşlem birimi: {device}")

# 2. Veri Yükleme ve Etiketleme
print("🔄 Vektörler yükleniyor...")
with open("Datasets/train_embeddings.pkl", "rb") as f:
    word_data_list = pickle.load(f)

X = np.array([item['vector'] for item in word_data_list], dtype=np.float32)
soft_labels = np.array([item['label'] for item in word_data_list])
y = np.where(soft_labels >= 0.5, 1, 0)

# 3. Eğitim ve Test Ayrımı (SMOTE'dan ÖNCE yapılmalı!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"⚖️ Orjinal Sınıf Dağılımı -> 0: {np.sum(y_train==0)}, 1: {np.sum(y_train==1)}")

# 4. SMOTE ile Veri Dengeleme (Sadece Eğitim Setine)
print("🧬 SMOTE ile azınlık sınıfı için sentetik vektörler üretiliyor...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"⚖️ SMOTE Sonrası Dağılım -> 0: {np.sum(y_train_smote==0)}, 1: {np.sum(y_train_smote==1)}")

# 5. PyTorch Tensorlerine Çevirme ve DataLoader Hazırlığı
X_train_t = torch.tensor(X_train_smote).to(device)
y_train_t = torch.tensor(y_train_smote, dtype=torch.float32).unsqueeze(1).to(device)

X_test_t = torch.tensor(X_test).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# 6. Çok Katmanlı Sinir Ağı (MLP) Mimarisi
class DisagreementMLP(nn.Module):
    def __init__(self):
        super(DisagreementMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256), # Vektörleri standardize eder (Eğitimi hızlandırır)
            nn.ReLU(),
            nn.Dropout(0.3),     # %30'unu unutarak ezberi engeller
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1)     # Çıktı katmanı (1 Nöron)
        )

    def forward(self, x):
        return self.network(x)

model = DisagreementMLP().to(device)

# Optimizasyon ve Kayıp Fonksiyonu (BCEWithLogitsLoss sigmoid'i kendi içinde yapar)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# 7. Model Eğitimi (Epoch Döngüsü)
epochs = 15
print("🧠 MLP Sinir Ağı Eğitimi Başlıyor...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs} | Kayıp (Loss): {total_loss/len(train_loader):.4f}")

# 8. Test Seti Üzerinde Değerlendirme
print("📊 Doğrulama (Validation) Yapılıyor...")
model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
    # Logitleri olasılığa çevir ve 0.5 eşiğine göre tahmin üret
    test_preds = (torch.sigmoid(test_logits) >= 0.5).int().cpu().numpy()

y_test_cpu = y_test_t.cpu().numpy()

accuracy = accuracy_score(y_test_cpu, test_preds)
print("\n" + "="*40)
print(f"🎯 MLP MODEL DOĞRULUK ORANI (ACCURACY): %{accuracy * 100:.2f}")
print("="*40)

print("\n📝 Detaylı Sınıflandırma Raporu:")
print(classification_report(y_test_cpu, test_preds, target_names=["Anlaşmazlık Yok", "Anlaşmazlık Var"]))
torch.save(model.state_dict(), 'Datasets/mlp_model.pth')
print("✅ Eğitilen model 'Datasets/mlp_model.pth' olarak kaydedildi.")