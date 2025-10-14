import streamlit as st
import pandas as pd 
import joblib

model = joblib.load("model_klasifikasi_jeruk.joblib")

st.set_page_config(
	page_title = "Klasifikasi Jeruk",
	page_icon = ":tangerine:"
)

st.title(":tangerine: Klasifikasi Jeruk")
st.markdown("Aplikasi machine learning untuk klasifikasi jeruk bagus, sedang, jelek")

diameter= st.slider("Diameter", 0.5, 8.0,6.0)
berat= st.slider("Berat", 80.0, 225.0, 100.0)
tebal_kulit= st.slider("Tebal Kulit", 0.1, 1.2, 0.4)
kadar_gula= st.slider("Kadar Gula", 5.0, 14.0, 9.0)
asal_daerah= st.pills("Asal Daerah", ["Jawa tengah", "Kalimantan", "Jawa Barat"], default="Kalimantan")
warna= st.pills("Warna", ["Hijau", "Kuning", "oranye"], default="Hijau")
musim_panen= st.pills("Musim Panen", ["Hujan", "Kemarau"], default="Hujan ")

if st.button("Prediksi", type="primary"):
	data_baru=pd.DataFrame([[7.89, 100, 0.35, 10, "Jawa Tengah", "hijau", "hujan"]], columns=["diameter", "berat", "tebal_kulit", "kadar_gula", "asal_daerah", 	"warna", "musim_panen"])
	prediksi = model.predict(data_baru)[0]
	presentase=max(model.predict_proba(data_baru)[0])
	st.success(f" Memprediksi dengan **{prediksi}** dengan keyakinan **{presentase*100:.2f}%**")
	st.ballons()
st.divider()
st.caption("Dibuat dengan :tangerine: oleh **Nabil Albara**")
