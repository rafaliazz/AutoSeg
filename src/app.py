import streamlit as st
import pandas as pd
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AutoSeg", layout="centered")
st.title("AutoSeg")

# -------------------------------------------------
# Health check (/root)
# -------------------------------------------------
st.subheader("API Status")

try:
    r = requests.get(f"{API_URL}/", timeout=2)
    if r.status_code == 200:
        st.success("API is running")
    else:
        st.error("API responded but not healthy")
except Exception as e:
    st.error(f"API not reachable: {e}")
    st.stop()

# -------------------------------------------------
# Transcribe audio and do everything  (/transcribe)
# -------------------------------------------------

st.subheader("Transcribe")

try:
    file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])

    confi_thres = st.number_input("confi_thres", value=-0.2)
    resegment_minim = st.number_input("resegment_minim", value=2.0)

    model_size = st.selectbox("model_size", ["tiny", "base", "small", "medium", "large"], index=2)
    device = st.selectbox("device", ["cpu", "cuda"], index=0)
    compute_type = st.selectbox("compute_type", ["int8", "float16", "float32"], index=0)

    if st.button("Start Transcription") and file:
        with st.spinner("Transcription Running"): 
            files = {
                "file": (file.name, file, file.type)
            }            
            data = {
                "confi_thres": confi_thres,
                "resegment_minim": resegment_minim,
                "model_size": model_size,
                "device": device,
                "compute_type": compute_type
            }
            r = requests.post(f"{API_URL}/transcribe", files = files, data=data)
            if r.status_code == 200:
                st.success("Transcription ran succesfully")
                result = r.json()
                st.success("Transcription completed")

                raw_text = result.get("raw_transcriptions", [])
                silent_segments = result.get("silent_segments", [])
                non_silent_segments = result.get("non_silent_segments", [])
                speech = result.get("speech_transcriptions", [])
                final = result.get("confidence_resegmented_transcriptions", [])

                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Raw Transcriptions",
                "Silent Segments", 
                "Non Silent Segments",
                "Speech",
                "Final Transcriptions"
            ])
                with tab1:
                    st.subheader("Raw Transcription Details")
                    df = pd.DataFrame(raw_text)

                    st.dataframe(
                        df,
                        width="stretch"
                    )
                with tab2:
                    st.subheader("Silent Segments Details")
                    df = pd.DataFrame(silent_segments)

                    st.dataframe(
                        df,
                        width="stretch"
                    )
                with tab3:
                    st.subheader("Non Silent Segments Details")
                    df = pd.DataFrame(non_silent_segments)

                    st.dataframe(
                        df,
                        width="stretch"
                    )
                with tab4:
                    st.subheader("Speech Transcription Details")
                    df = pd.DataFrame(speech)

                    st.dataframe(
                        df,
                        width="stretch"
                    )
                with tab5: 
                    st.subheader("Final Transcription Details (Non-Silent Transcription + Confidence based retranscription)")
                    df = pd.DataFrame(final)

                    st.dataframe(
                        df,
                        width="stretch"
                    )


            else:
                st.error("API responded but not succesful")

except Exception as e:
    st.error(f"Transcription failed: {e}")
    st.stop()