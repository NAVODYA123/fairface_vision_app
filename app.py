import streamlit as st
import torch
from PIL import Image

from model import FairVisionCNN
from utils import preprocess_image, get_top_predictions
from class_names import AGE_CLASSES

st.set_page_config(
    page_title="FairVision Bias Analysis",
    layout="wide"
)

st.title("FairVision - Bias Detection and Mitigation")

st.write("""
This application compares:

1. Baseline CNN Model
2. Bias Mitigated CNN Model

Both models predict age groups using the FairFace dataset.
""")

# -------------------------------
# LOAD MODELS
# -------------------------------

@st.cache_resource
def load_baseline_model():

    model = FairVisionCNN()

    model.load_state_dict(
        torch.load(
            "basic_model.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model


@st.cache_resource
def load_mitigated_model():

    model = FairVisionCNN()

    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model


baseline_model = load_baseline_model()
mitigated_model = load_mitigated_model()

# -------------------------------
# IMAGE UPLOAD
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload a Face Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

    input_tensor = preprocess_image(image)

    # -------------------------------
    # PREDICTIONS
    # -------------------------------

    with torch.no_grad():

        baseline_outputs = baseline_model(input_tensor)
        mitigated_outputs = mitigated_model(input_tensor)

    baseline_predictions = get_top_predictions(
        baseline_outputs,
        AGE_CLASSES
    )

    mitigated_predictions = get_top_predictions(
        mitigated_outputs,
        AGE_CLASSES
    )

    # -------------------------------
    # SIDE BY SIDE COLUMNS
    # -------------------------------

    col1, col2 = st.columns(2)

    # BASELINE
    with col1:

        st.subheader("Baseline CNN Model")

        st.write("""
        Higher overall accuracy but larger fairness gaps.
        """)

        for pred in baseline_predictions:

            st.write(
                f"**{pred['class']}** : "
                f"{pred['confidence']:.2f}%"
            )

    # MITIGATED
    with col2:

        st.subheader("Bias Mitigated CNN Model")

        st.write("""
        Improved fairness consistency but lower accuracy.
        """)

        for pred in mitigated_predictions:

            st.write(
                f"**{pred['class']}** : "
                f"{pred['confidence']:.2f}%"
            )

# -------------------------------
# FAIRNESS COMPARISON
# -------------------------------

st.divider()

st.subheader("Fairness Comparison")

comparison_data = {
    "Metric": [
        "Overall Accuracy",
        "Race Fairness Gap",
        "Gender Fairness Gap"
    ],
    "Baseline Model": [
        "45%",
        "10.49%",
        "2.56%"
    ],
    "Mitigated Model": [
        "17%",
        "8.58%",
        "0.11%"
    ]
}

st.table(comparison_data)

# -------------------------------
# LIMITATIONS
# -------------------------------

st.divider()

st.subheader("Responsible AI Notes")

st.write("""
- This system is intended for educational and research purposes only.
- Bias mitigation can improve fairness while reducing model accuracy.
- The system should not be used in high-risk real-world environments.
- Performance may vary across demographic groups.
""")