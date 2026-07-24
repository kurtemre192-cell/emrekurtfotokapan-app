import streamlit as st
import torch
from PytorchWildlife.models import detection as pw_detection

st.set_page_config(page_title="Fotokapan Analiz", page_icon="🐾")

st.title("🐾 Fotokapan Akıllı Analiz Sistemi")
st.write("Sistem şu an ayağa kalkıyor. Yapay zeka modeli sunucuya yüklenirken lütfen bekleyin...")

@st.cache_resource
def load_model():
    return pw_detection.MegaDetectorV6(device="cpu", pretrained=True, version="MDV6-yolov9-c")

try:
    detector = load_model()
    st.success("✅ MegaDetectorV6 başarıyla kuruldu ve sistem analize hazır!")
    st.file_uploader("Çalıştığını görmek için bir fotoğraf seçin", type=['png', 'jpg', 'jpeg'])
except Exception as e:
    st.error(f"Kurulum sırasında bir hata oluştu: {e}")
