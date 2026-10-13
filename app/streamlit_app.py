import streamlit as st
import pandas as pd
import os
import sys
from PIL import Image

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MRI_VALIDATOR_DIR = os.path.join(
    PROJECT_ROOT,
    "src",
    "mri_validator"
)

if MRI_VALIDATOR_DIR not in sys.path:
    sys.path.append(MRI_VALIDATOR_DIR)

print("PROJECT_ROOT =", PROJECT_ROOT)
print("MRI_VALIDATOR_DIR =", MRI_VALIDATOR_DIR)
print("PATH EXISTS =", os.path.exists(MRI_VALIDATOR_DIR))

CNN_DIR = os.path.join(PROJECT_ROOT, "src", "cnn")

if CNN_DIR not in sys.path:
    sys.path.append(CNN_DIR)

from translations import (
    LANGUAGES,
    get_text
)

# ---------------------------------
# Import Real Prediction Pipeline
# ---------------------------------
from predict_single_mri import (
    predict_roi_dataframe,
    get_prediction_label,
    explain_result
)

from uploaded_roi_features import (
    save_uploaded_file,
    extract_image_roi_features,
    extract_nifti_roi_features,
    save_roi_features
)

# 1. New Import Added for CNN
import importlib.util

validator_file = os.path.join(
    MRI_VALIDATOR_DIR,
    "validator_predict.py"
)

spec = importlib.util.spec_from_file_location(
    "validator_predict",
    validator_file
)

validator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator_module)

validate_mri = validator_module.validate_mri

from cnn_predict import (
    predict_image,
    get_prediction_label as get_cnn_prediction_label,
    explain_prediction,
    get_disease_details
)

# ---------------------------------
# Import Report Generator
# ---------------------------------
from report_generator import (
    generate_report
)

# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="Alzheimer Disease Detection",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------
# Theme Selection
# ---------------------------------

if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "System preference"

if "theme_menu_visible" not in st.session_state:
    st.session_state.theme_menu_visible = False

def toggle_theme_menu():
    st.session_state.theme_menu_visible = not st.session_state.theme_menu_visible

with st.sidebar:
    if st.button("Theme ⚙️", key="theme_toggle_button"):
        toggle_theme_menu()
    if st.session_state.theme_menu_visible:
        st.selectbox(
            "Select color preference",
            ["System preference", "Light mode", "Dark mode"],
            index=["System preference", "Light mode", "Dark mode"].index(
                st.session_state.theme_choice
            ),
            key="theme_choice",
        )

selected_theme = st.session_state.theme_choice

if selected_theme == "Light mode":
    theme_css = """
    <style>
    body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
    [data-testid="stHeader"], [data-testid="stToolbar"], .css-1lcbmhc,
    .css-1v3fvcr, .stButton>button, .stTextInput>div>input,
    .stTextArea>div>textarea, .stSelectbox>div>div>div>div {
        background-color: #ffffff !important;
        color: #0f1117 !important;
        border-color: #d8dee4 !important;
    }
    .stButton>button:hover {
        background-color: #e2e8f0 !important;
    }
    </style>
    """
elif selected_theme == "Dark mode":
    theme_css = """
    <style>
    body, body * {
        color: #f7fbff !important;
        background-color: transparent !important;
    }
    body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
    [data-testid="stHeader"], [data-testid="stToolbar"], .css-1lcbmhc,
    .css-1v3fvcr, .stButton>button, .stTextInput>div>input,
    .stTextArea>div>textarea, .stSelectbox>div>div>div>div,
    .stMarkdown, .stText, .stTextInput>div>label, .stCheckbox, .stRadio,
    .stMultiSelect, .stSlider, .stNumberInput {
        background-color: #0b111b !important;
        color: #f7fbff !important;
        border-color: #39475c !important;
    }
    .stButton>button:hover {
        background-color: #1f2a3a !important;
    }
    .stTextInput>div>input::placeholder,
    .stTextArea>div>textarea::placeholder {
        color: #b3c1dc !important;
    }
    </style>
    """
else:
    theme_css = """
    <style>
    @media (prefers-color-scheme: dark) {
        body, body * {
            color: #f7fbff !important;
            background-color: transparent !important;
        }
        body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
        [data-testid="stHeader"], [data-testid="stToolbar"], .css-1lcbmhc,
        .css-1v3fvcr, .stButton>button, .stTextInput>div>input,
        .stTextArea>div>textarea, .stSelectbox>div>div>div>div,
        .stMarkdown, .stText, .stTextInput>div>label, .stCheckbox, .stRadio,
        .stMultiSelect, .stSlider, .stNumberInput {
            background-color: #0b111b !important;
            color: #f7fbff !important;
            border-color: #39475c !important;
        }
        .stTextInput>div>input::placeholder,
        .stTextArea>div>textarea::placeholder {
            color: #b3c1dc !important;
        }
    }
    @media (prefers-color-scheme: light) {
        body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
        [data-testid="stHeader"], [data-testid="stToolbar"], .css-1lcbmhc,
        .css-1v3fvcr, .stButton>button, .stTextInput>div>input,
        .stTextArea>div>textarea, .stSelectbox>div>div>div>div {
            background-color: #ffffff !important;
            color: #0f1117 !important;
            border-color: #d8dee4 !important;
        }
    }
    </style>
    """

st.markdown(theme_css, unsafe_allow_html=True)

# ---------------------------------
# Sidebar
# ---------------------------------

st.sidebar.title(
    get_text(st.session_state.get("selected_language", "English"), "sidebar_title")
)

# Use session_state to allow the selectbox label to be translated
lang_default = st.session_state.get("selected_language", "English")
language = st.sidebar.selectbox(
    get_text(lang_default, "select_language"),
    list(LANGUAGES.keys()),
    index=(list(LANGUAGES.keys()).index(lang_default)
           if lang_default in LANGUAGES else 0),
    key="selected_language"
)

st.sidebar.markdown("---")

st.sidebar.metric(
    get_text(language, "validation_accuracy"),
    "67.14%"
)

st.sidebar.metric(
    get_text(language, "cnn_accuracy"),
    "91.56%"
)

st.sidebar.metric(
    get_text(language, "f1_score"),
    "57.63%"
)

st.sidebar.metric(
    get_text(language, "roc_auc"),
    "68.92%"
)

st.sidebar.metric(
    get_text(language, "brain_regions_metric"),
    "48"
)

st.sidebar.metric(
    get_text(language, "mri_scans_metric"),
    "470"
)

st.sidebar.metric(
    get_text(language, "model_metric"),
    "GAT"
)

# ---------------------------------
# Header
# ---------------------------------

st.title(
    f"{get_text(language,'app_title')}"
)

#st.markdown(
#    """
#This project uses:
#
#- MRI Brain Scans
#- Harvard Oxford Atlas
#- ROI Feature Extraction
#- Graph Construction
#- Graph Attention Networks (GAT)
#- Uncertainty Estimation
#- Explainability Analysis
#- Alzheimer's Disease Detection
#"""
#)

st.markdown("---")

# ---------------------------------
# Upload Section
# ---------------------------------

st.header(
    f"📤 {get_text(language,'upload')}"
)

uploaded_file = st.file_uploader(
    get_text(language, "uploader_label"),
    type=[
        "nii",
        "gz",
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "tiff"
    ]
)

if uploaded_file:

    st.success(
        f"{get_text(language,'upload_success')}: {uploaded_file.name}"
    )

    filename = uploaded_file.name.lower()
    uploaded_path = save_uploaded_file(
        uploaded_file
    )
    roi_df = None
    roi_path = None
    feature_source = ""
    cnn_result = None  # To flag if CNN layout metrics or table should render

    # ---------------------------------
    # 2. Image Upload Block (Replaced with CNN Integration)
    # ---------------------------------
    if filename.endswith(
        (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tiff"
        )
    ):

        image = Image.open(uploaded_path)

        st.image(
            image,
            caption="Uploaded Brain Image",
            width=250
        )

        validator_result = validate_mri(
            uploaded_path
        )

        #st.write(validator_result)

        if (
            not validator_result["is_mri"]
            and validator_result["confidence"] > 0.95
        ):

            st.error(
                get_text(language, "only_mri_allowed")
            )

            st.stop()

        cnn_result = predict_image(
            uploaded_path
        )
        final_label = get_cnn_prediction_label(
            cnn_result["label"],
            cnn_result["confidence"]
        )

        if final_label == "UNCERTAIN":

            st.warning(
                "⚠ Low confidence prediction.\n\n"
                "Please use a clearer MRI scan."
            )

            st.stop()

        prediction_label = get_cnn_prediction_label(
            cnn_result["label"],
            cnn_result["confidence"]
        )

        confidence = (
            cnn_result["confidence"] * 100
        )

        uncertainty = (
            cnn_result["uncertainty"] * 100
        )

        display_label = prediction_label

    else:

        st.info(
            get_text(language, "mri_upload_info")
        )

        try:
            roi_df = extract_nifti_roi_features(
                uploaded_path
            )
            feature_source = "MRI atlas/grid ROI features"

        except Exception as e:
            st.error(
                f"Failed to extract MRI ROI features: {e}"
            )
            st.stop()

        roi_path = save_roi_features(
            roi_df,
            uploaded_path
        )

        result = predict_roi_dataframe(
            roi_df
        )

        st.caption(
            f"{get_text(language,'prediction_source')} {feature_source} saved to {roi_path}"
        )

        prediction_label = get_prediction_label(
            result["prediction"]
        )

        confidence = (
            result["confidence"] * 100
        )

        uncertainty = (
            result["uncertainty"] * 100
        )

        if confidence < 60:
            display_label = f"Uncertain - {prediction_label}"
        else:
            display_label = prediction_label

    # ---------------------------------
    # Metrics Display (Shared block)
    # ---------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            get_text(
                language,
                "prediction"
            ),
            display_label
        )

    with col2:

        st.metric(
            get_text(
                language,
                "confidence"
            ),
            f"{confidence:.2f}%"
        )

    with col3:

        st.metric(
            get_text(
                language,
                "uncertainty"
            ),
            f"{uncertainty:.2f}%"
        )

    # ---------------------------------
    # 3. Probability Table Section (CNN Only)
    # ---------------------------------
    if cnn_result is not None:
        st.markdown("---")
        prob_df = pd.DataFrame({
            get_text(language, "class_column"): [
                get_text(language, "class_MildDemented"),
                get_text(language, "class_ModerateDemented"),
                get_text(language, "class_NonDemented"),
                get_text(language, "class_VeryMildDemented")
            ],
            get_text(language, "probability_column"): cnn_result["probabilities"]
        })

        st.subheader(
            get_text(language, "class_probabilities")
        )

        st.dataframe(
            prob_df,
            use_container_width=True
        )

    st.markdown("---")

    # ---------------------------------
    # Disease Information
    # ---------------------------------

    if cnn_result is not None:

        disease_info = get_disease_details(
            cnn_result["label"]
        )

        if disease_info:

            st.markdown("---")

            st.subheader(
                get_text(language, "disease_information")
            )

            st.write(
                get_text(language, disease_info["description"])
            )

            st.subheader(
                get_text(language, "risk_level")
            )

            st.info(
                get_text(language, disease_info["risk"])
            )

            st.subheader(
                get_text(language, "common_symptoms")
            )

            for symptom in disease_info["symptoms"]:

                st.write(
                    f"• {get_text(language, symptom)}"
                )

            st.subheader(
                get_text(language, "recommendations")
            )

            for item in disease_info["recommendations"]:

                st.write(
                    f"• {get_text(language, item)}"
                )

            st.warning(
                get_text(language, "disclaimer")
            )
    # ---------------------------------
    # Human Explanation
    # ---------------------------------

    st.subheader(
        get_text(language, "human_explanation")
    )

    if confidence < 60:

        st.warning(
            get_text(
                language,
                "uncertain"
            )
        )

        st.info(
            get_text(language, "human_unclear_message")
        )

    elif prediction_label in ["NORMAL", "NonDemented"]:

        st.success(
            get_text(
                language,
                "normal"
            )
        )

    else:

        st.warning(
            get_text(
                language,
                "alz"
            )
        )

    st.info(
        get_text(
            language,
            "disclaimer"
        )
    )

    # ---------------------------------
    # PDF Report Generation Button
    # ---------------------------------
    st.markdown("---")
    st.subheader(get_text(language, "export_analysis"))
    
    if st.button(get_text(language, "generate_pdf_report")):
        try:
            # రిపోర్ట్ సేవ్ అవ్వడానికి అవుట్‌పుట్ ఫోల్డర్ ఉందో లేదోチェック చేయడం
            os.makedirs("outputs", exist_ok=True)
            
            generate_report(
                patient_name="Patient",
                prediction=display_label,
                confidence=confidence,
                brain_regions=[
                    "Temporal Cortex",
                    "Fusiform Gyrus"
                ],
                heatmap_path=os.path.join(PROJECT_ROOT, "outputs", "heatmaps", "attention_map.png"),
                output_pdf=os.path.join(PROJECT_ROOT, "outputs", "report.pdf")
            )
            st.success("📄 Report Generated Successfully! Saved to 'outputs/report.pdf'")
            
            if os.path.exists("outputs/report.pdf"):
                with open("outputs/report.pdf", "rb") as f:
                    st.download_button(
                        label=get_text(language, "download_pdf_report"),
                        data=f,
                        file_name="Alzheimer_Detection_Report.pdf",
                        mime="application/pdf"
                    )
        except Exception as e:
            st.error(f"Failed to generate report: {e}")

# ---------------------------------
# Dataset Information
# ---------------------------------

#st.markdown("---")

#st.header(
#    f"📊 {get_text(language,'dataset')}"
#)

#try:

#    dataset = pd.read_csv(
#        "processed/master_dataset.csv"
#    )

#    c1, c2 = st.columns(2)

#    with c1:

#        st.metric(
#            "Total Samples",
#            len(dataset)
#        )

#    with c2:

#        st.metric(
#            "Features Per ROI",
#            "5"
#        )

#    st.dataframe(
#        dataset.head(10),
#        use_container_width=True
#    )

#except Exception as e:

#    st.warning(
#        f"Dataset Error: {e}"
#    )

# ---------------------------------
# ROI Importance
# ---------------------------------

st.markdown("---")

st.header(
    f"{get_text(language,'roi')}"
)

roi_path = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "heatmaps",
    "roi_importance.png"
)

if os.path.exists(
    roi_path
):
    st.image(
        roi_path,
        caption=get_text(language, 'roi'),
        use_container_width=True
    )

else:

    st.warning(
        get_text(language, "roi_image_not_found")
    )

# ---------------------------------
# Important Brain Regions
# ---------------------------------

st.markdown("---")

st.header(
    f"{get_text(language,'brain_regions')}"
)

st.write(get_text(language, "brain_regions_list"))

# Show each region as a bullet with per-item translation
regions = [
    "region_1",
    "region_2",
    "region_3",
    "region_4",
    "region_5",
    "region_6",
    "region_7",
    "region_8",
]

for r in regions:
    st.write(f"• {get_text(language, r)}")


# ---------------------------------
# Attention Map
# ---------------------------------

st.markdown("---")

st.header(
    f"🎯 {get_text(language,'attention')}"
)

attention_path = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "heatmaps",
    "attention_map.png"
)

if os.path.exists(
    attention_path
):

    st.image(
        attention_path,
        caption=get_text(language, "attention"),
        use_container_width=True
    )

else:

    st.warning(
        get_text(language, "attention_map_not_found")
    )

# ---------------------------------
# Evaluation
# ---------------------------------

st.markdown("---")

st.header(
    f"📈 {get_text(language,'evaluation')}"
)

cm_path = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "confusion_matrices",
    "confusion_matrix.png"
)

roc_path = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "confusion_matrices",
    "roc_curve.png"
)

col1, col2 = st.columns(2)

with col1:

    if os.path.exists(
        cm_path
    ):

        st.image(
            cm_path,
            caption=get_text(language, "confusion_matrix_caption")
        )

    else:

        st.warning(
            get_text(language, "confusion_matrix_missing")
        )

with col2:

    if os.path.exists(
        roc_path
    ):

        st.image(
            roc_path,
            caption=get_text(language, "roc_curve_caption")
        )

    else:

        st.warning(
            get_text(language, "roc_curve_missing")
        )

# ---------------------------------
# Uncertainty Section
# ---------------------------------

st.markdown("---")

st.header(
    f"⚡ {get_text(language, 'uncertainty_estimation')}"
)

st.info(
    get_text(language, 'uncertainty_info')
)

# ---------------------------------
