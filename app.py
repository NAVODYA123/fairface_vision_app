import streamlit as st
import torch
from PIL import Image

from model import FairVisionCNN
from utils import preprocess_image, get_top_predictions
from class_names import AGE_CLASSES

st.set_page_config(page_title="FairVision", layout="centered")

st.title("FairVision - Age Group Classification")

st.write("""
This application predicts age groups from facial images using a CNN model trained on the FairFace dataset.

The system was also evaluated for fairness across race and gender groups.
""")

# Load model
@st.cache_resource
def load_model():

    model = FairVisionCNN()

    model.load_state_dict(
        torch.load("basic_model.pth", map_location=torch.device("cpu"))
    )

    model.eval()

    return model

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a Face Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(input_tensor)

    predictions = get_top_predictions(outputs, AGE_CLASSES)

    st.subheader("Top 3 Predictions")

    for pred in predictions:

        st.write(
            f"**{pred['class']}** : {pred['confidence']:.2f}%"
        )

st.divider()

st.subheader("System Limitations")

st.write("""
- This model is not intended for real-world surveillance use.
- Accuracy may vary across demographic groups.
- The system was developed for educational and research purposes only.
- Bias mitigation techniques may reduce overall prediction accuracy.
""")