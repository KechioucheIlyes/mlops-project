import io
from typing import Any

import requests
import streamlit as st
from PIL import Image


PREDICT_API_URL = "http://predict-api:8001/predict"
HEALTH_API_URL = "http://predict-api:8001/health"


st.set_page_config(
    page_title="Shifaa Chest X-ray Classifier",
    page_icon="🫁",
    layout="wide",
)


def check_api_health() -> tuple[bool, str]:
    try:
        response = requests.get(HEALTH_API_URL, timeout=10)
        response.raise_for_status()
        return True, "predict-api disponible"
    except Exception as e:
        return False, f"predict-api indisponible: {e}"


def call_predict_api(uploaded_file) -> dict[str, Any]:
    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
    }

    response = requests.post(PREDICT_API_URL, files=files, timeout=120)
    response.raise_for_status()
    return response.json()


def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


st.title("🫁 Shifaa Chest X-ray Classifier")
st.caption("Interface de prédiction connectée à la predict-api")

ok, health_message = check_api_health()
if ok:
    st.success(health_message)
else:
    st.error(health_message)

left, right = st.columns([1.1, 1])

with left:
    st.subheader("Upload d'une image")
    uploaded_file = st.file_uploader(
        "Dépose une image radio (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
    )

    camera_file = st.camera_input("Ou prends une photo directement")

    image_source = uploaded_file if uploaded_file is not None else camera_file

    if image_source is not None:
        image = Image.open(image_source)
        st.image(image, caption="Image envoyée", use_container_width=True)

with right:
    st.subheader("Résultat")

    image_source = uploaded_file if uploaded_file is not None else camera_file

    if image_source is None:
        st.info("Ajoute une image pour lancer une prédiction.")
    else:
        if st.button("Lancer la prédiction", use_container_width=True):
            try:
                with st.spinner("Analyse en cours..."):
                    result = call_predict_api(image_source)

                predicted_class = result.get("predicted_class", "N/A")
                confidence = float(result.get("confidence", 0.0))
                probabilities = result.get("probabilities", {})
                model_info = result.get("model", {})

                st.success("Prédiction terminée")

                st.metric("Classe prédite", predicted_class)
                st.metric("Confiance", format_percentage(confidence))

                st.markdown("### Probabilités par classe")
                for label, proba in probabilities.items():
                    try:
                        value = float(proba)
                    except Exception:
                        value = 0.0
                    st.progress(value, text=f"{label} — {format_percentage(value)}")

                st.markdown("### Modèle utilisé")
                st.write(
                    {
                        "candidate_id": model_info.get("candidate_id"),
                        "run_name": model_info.get("run_name"),
                        "metric_name": model_info.get("metric_name"),
                        "metric_value": model_info.get("metric_value"),
                    }
                )

                st.markdown("### Réponse brute")
                st.json(result)

            except requests.HTTPError as e:
                st.error(f"Erreur HTTP predict-api: {e}")
                if e.response is not None:
                    try:
                        st.json(e.response.json())
                    except Exception:
                        st.code(e.response.text)
            except Exception as e:
                st.error(f"Erreur pendant la prédiction: {e}")

st.markdown("---")
st.caption("Front Streamlit léger, logique d'inférence déléguée à predict-api.")