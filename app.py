import streamlit as st
import torch
from PytorchWildlife.models import detection as pw_detection
from PIL import Image
import numpy as np

st.set_page_config(page_title="Fotokapan Analiz", page_icon="🐾")

st.title("🐾 Fotokapan Akıllı Analiz Sistemi")
st.write("Sistem şu an ayağa kalkıyor. Yapay zeka modeli sunucuya yüklenirken lütfen bekleyin...")

@st.cache_resource
def load_model():
    return pw_detection.MegaDetectorV6(device="cpu", pretrained=True, version="MDV6-yolov9-c")

try:
    detector = load_model()
    st.success("✅ MegaDetectorV6 başarıyla kuruldu ve sistem analize hazır!")
    
    # Fotoğraf yükleme alanı
    uploaded_file = st.file_uploader("Çalıştığını görmek için bir fotoğraf seçin", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Yüklenen fotoğrafı aç ve ekranda göster
        image = Image.open(uploaded_file)
        st.image(image, caption="Yüklenen Fotoğraf", use_column_width=True)
        
        # Analizi başlatacak buton
        if st.button("Fotoğrafı Analiz Et"):
            with st.spinner("Yapay zeka fotoğrafı inceliyor, lütfen bekleyin..."):
                # Görüntüyü modelin anlayacağı formata (Numpy RGB) çevir
                img_array = np.array(image.convert("RGB"))
                
                # PytorchWildlife modeli ile analizi yap
                sonuclar = detector.single_image_detection(img_array)
                
                st.success("Analiz başarıyla tamamlandı!")
                
                # Sonuç çıktılarını ekrana yazdır
                st.write(sonuclar)
                
except Exception as e:
    st.error(f"Kurulum veya analiz sırasında bir hata oluştu: {e}")
