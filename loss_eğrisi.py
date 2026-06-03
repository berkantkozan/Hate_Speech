import matplotlib.pyplot as plt
import seaborn as sns

# Akademik ve şık bir görünüm için seaborn temasını aktif ediyoruz
sns.set_theme(style="whitegrid")

# Eğitim loglarından aldığımız Epoch ve Loss değerleri
epochs = list(range(1, 16))
loss_values = [
    0.5718, 0.4991, 0.4477, 0.4076, 0.3733, 
    0.3466, 0.3239, 0.3070, 0.2898, 0.2739, 
    0.2635, 0.2512, 0.2409, 0.2328, 0.2245
]

# Grafiğin boyutunu ayarlıyoruz (Akademik raporlar için ideal boyut)
plt.figure(figsize=(10, 6))

# Çizgi grafiğini oluşturuyoruz
plt.plot(epochs, loss_values, marker='o', linestyle='-', color='#b30000', linewidth=2.5, markersize=8, label='Eğitim Kaybı (Training Loss)')

# Eksen isimlendirmeleri ve Başlık
plt.title('MLP Modeli Eğitim Kaybı (Loss) Eğrisi', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Epok (Epoch)', fontsize=14, fontweight='bold')
plt.ylabel('Kayıp (BCE Loss)', fontsize=14, fontweight='bold')

# Eksen ölçeklerini ayarlama
plt.xticks(epochs, fontsize=12)
plt.yticks(fontsize=12)

# Lejant (Bilgi Kutusu) ekleme
plt.legend(fontsize=12, loc='upper right')

# Arka plan ızgaralarını daha yumuşak yapma
plt.grid(True, linestyle='--', alpha=0.7)

# Grafiği bilgisayara yüksek çözünürlükte (300 dpi) kaydetme
plt.savefig('mlp_loss_egrisi.png', dpi=300, bbox_inches='tight')

# Grafiği ekranda göster
plt.show()

print("✅ Akademik kalitede grafik 'mlp_loss_egrisi.png' adıyla kaydedildi!")