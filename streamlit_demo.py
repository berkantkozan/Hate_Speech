import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# 1. MLP Mimarisini Tekrar Tanımlıyoruz (Ağırlıkları yüklemek için)
class DisagreementMLP(nn.Module):
    def __init__(self):
        super(DisagreementMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

# 2. Modelleri Hafızaya Yükleme (Önbelleğe alıyoruz ki her seferinde baştan yüklenmesin)
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # mBERT Yükle
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    bert = BertModel.from_pretrained("bert-base-multilingual-cased").to(device)
    bert.eval()
    
    # Kendi MLP Modelimizi Yükle
    mlp = DisagreementMLP().to(device)
    mlp.load_state_dict(torch.load('Datasets/mlp_model.pth', map_location=device))
    mlp.eval()
    
    return tokenizer, bert, mlp, device

tokenizer, bert, mlp, device = load_models()

# 3. Streamlit Arayüz Tasarımı
st.set_page_config(page_title="Nefret Söylemi & Anlaşmazlık Tespiti", page_icon="🔍")

st.title("🔍 Etiketleyici Anlaşmazlığı Tespiti (Demo)")
st.markdown("""
Bu demo, girilen metindeki kelimelerin veri etiketleyicileri arasında (annotator disagreement) 
bir kararsızlık veya görüş ayrılığı yaratıp yaratmayacağını **mBERT + MLP** mimarisiyle tahmin eder.
""")

# Kullanıcıdan metin alma
user_input = st.text_area("Analiz edilecek metni girin:", height=150, placeholder="Örneğin: Bu konu hakkında çok farklı görüşler var...")

if st.button("🚀 Metni Analiz Et"):
    if user_input.strip() == "":
        st.warning("Lütfen analiz etmek için bir metin girin.")
    else:
        with st.spinner('Bağlamsal vektörler çıkarılıyor ve MLP ağı üzerinden geçiriliyor...'):
            # Metni Tokenize et ve BERT'e sok
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
            word_ids = inputs.word_ids()
            
            with torch.no_grad():
                outputs = bert(**inputs)
                embeddings = outputs.last_hidden_state[0]
            
            # Kelime bazlı vektörleri toparla
            temp_word_map = {}
            for idx, w_id in enumerate(word_ids):
                if w_id is None: continue
                if w_id not in temp_word_map:
                    temp_word_map[w_id] = [embeddings[idx]]
                else:
                    temp_word_map[w_id].append(embeddings[idx])
            
            # MLP Tahmini
            disagreement_found = False
            
            for w_id, vecs in temp_word_map.items():
                avg_vec = torch.stack(vecs).mean(dim=0).unsqueeze(0) # (1, 768)
                
                with torch.no_grad():
                    logit = mlp(avg_vec)
                    prob = torch.sigmoid(logit).item()
                
                # Eğer herhangi bir kelimede anlaşmazlık olasılığı 0.5'ten büyükse
                if prob >= 0.5:
                    disagreement_found = True
                    break
            
            # Sonucu Ekrana Bas
            st.markdown("---")
            st.subheader("📊 Analiz Sonucu")
            
            if disagreement_found:
                st.error("⚠️ **Sonuç: ANLAŞMAZLIK VAR (Disagreement Detected)**")
                st.markdown("Model, bu metnin farklı demografik yapılara veya inançlara sahip insanlar (etiketleyiciler) arasında **görüş ayrılığı ve öznellik** yaratacağını tahmin etti.")
            else:
                st.success("✅ **Sonuç: ANLAŞMAZLIK YOK (Consensus)**")
                st.markdown("Model, bu metnin etiketleyiciler arasında genel bir fikir birliğiyle değerlendirileceğini tahmin etti.")