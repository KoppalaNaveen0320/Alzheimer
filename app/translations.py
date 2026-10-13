import os
from deep_translator import GoogleTranslator

# --------------------------------------------------
# Supported Languages
# --------------------------------------------------
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Bengali": "bn",
    "Odia": "or",
    "Assamese": "as",
    "Urdu": "ur",
    "Nepali": "ne",
    "Kashmiri": "ks",
    "Sindhi": "sd"
}

# --------------------------------------------------
# Base English UI Text
# --------------------------------------------------
TEXTS = {
    "app_title": "Alzheimer Disease Detection",
    "upload": "Upload Brain Scan",
    "sidebar_title": "🧠 Alzheimer GNN Dashboard",
    "select_language": "🌍 Select Language",
    "prediction": "Prediction",
    "confidence": "Confidence",
    "uncertainty": "Uncertainty",
    "dataset": "Dataset Information",
    "evaluation": "Model Evaluation",
    "roi": "ROI Importance Analysis",
    "attention": "Attention Analysis",
    "brain_regions": "Important Brain Regions",
    "class_probabilities": "Class Probabilities",
    "class_column": "Class",
    "probability_column": "Probability",
    "disease_information": "Disease Information",
    "risk_level": "Risk Level",
    "common_symptoms": "Common Symptoms",
    "recommendations": "Recommendations",
    "human_explanation": "Human Readable Explanation",
    "export_analysis": "Export Analysis",
    "generate_pdf_report": "Generate PDF Report",
    "download_pdf_report": "Download PDF Report",
    "prediction_source": "Prediction source:",
    "validation_accuracy": "Validation Accuracy",
    "cnn_accuracy": "CNN Accuracy",
    "f1_score": "F1 Score",
    "roc_auc": "ROC-AUC",
    "brain_regions_metric": "Brain Regions",
    "mri_scans_metric": "MRI Scans",
    "model_metric": "Model",
    "uploader_label": "Upload MRI Scan or Brain Image",
    "mri_upload_info": "MRI file uploaded successfully.\n\nPrediction will use extracted\nMRI ROI features.",
    "class_MildDemented": "MildDemented",
    "class_ModerateDemented": "ModerateDemented",
    "class_NonDemented": "NonDemented",
    "class_VeryMildDemented": "VeryMildDemented",
    "uncertainty_estimation": "Uncertainty Estimation",
    "uncertainty_info": "Monte Carlo Dropout is used for uncertainty estimation.\n\nLower uncertainty indicates higher prediction reliability.\n\nHigher uncertainty indicates lower prediction confidence.",
    "human_unclear_message": "The model could not clearly\ndistinguish between Normal and\nAlzheimer patterns.\n\nAdditional scans or medical\nevaluation may be required.",
    "brain_regions_list": "Top regions contributing to prediction:",
    "region_1": "Temporal Occipital Fusiform Cortex",
    "region_2": "Parietal Opercular Cortex",
    "region_3": "Occipital Fusiform Gyrus",
    "region_4": "Inferior Temporal Gyrus",
    "region_5": "Temporal Fusiform Cortex",
    "region_6": "Cingulate Gyrus",
    "region_7": "Insular Cortex",
    "region_8": "Frontal Opercular Cortex",
    "report_generated_success": "📄 Report Generated Successfully! Saved to 'outputs/report.pdf'",
    "confusion_matrix_caption": "Confusion Matrix",
    "roc_curve_caption": "ROC Curve",
    "graph_attention_caption": "Graph Attention Map",
    "total_samples": "Total Samples",
    "features_per_roi": "Features Per ROI",
    "success": "Pipeline Execution Completed Successfully",
    "normal": "No strong Alzheimer's related pattern detected.",
    "alz": "Possible Alzheimer's related changes detected.",
    "uncertain": "Prediction is uncertain.",
    "upload_success": "File uploaded successfully.",
    "mri_ready": "MRI file received and ready for analysis.",
    "image_ready": "Brain image uploaded successfully.",
    "only_mri_allowed": "❌ Only MRI Brain Images are allowed.\n\nPlease upload a valid MRI scan.",
    "roi_image_not_found": "ROI Importance image not found.",
    "attention_map_not_found": "Attention Map not found.",
    "confusion_matrix_missing": "Confusion Matrix Missing",
    "roc_curve_missing": "ROC Curve Missing",
    "model": "Graph Attention Network",
    "disclaimer": """This result is generated using an AI model.
The result should not be considered a final medical diagnosis.
Please consult a neurologist for professional evaluation."""
}

_translation_cache = {}

# --------------------------------------------------
# Translate Text
# --------------------------------------------------
def translate_text(text, language):
    if language == "English":
        return text

    if language not in LANGUAGES:
        return text

    cache_key = (language, text)

    if cache_key in _translation_cache:
        return _translation_cache[cache_key]

    try:
        translated = GoogleTranslator(
            source="auto", 
            target=LANGUAGES[language]
        ).translate(text)
        
        _translation_cache[cache_key] = translated
        return translated
        
    except Exception as e:
        return text

# --------------------------------------------------
# Get UI Text
# --------------------------------------------------
def get_text(language, key):
    text = TEXTS.get(key, key)
    return translate_text(text, language)

# --------------------------------------------------
# Get Available Languages
# --------------------------------------------------
def get_languages():
    return list(LANGUAGES.keys())

# --------------------------------------------------
# Test Block
# --------------------------------------------------
if __name__ == "__main__":
    print("Testing Translation Pipeline...\n")
    
    telugu_title = get_text("Telugu", "app_title")
    hindi_pred = get_text("Hindi", "prediction")
    tamil_success = get_text("Tamil", "success")
    
    print(f"Telugu Title : {telugu_title}")
    print(f"Hindi Pred   : {hindi_pred}")
    print(f"Tamil Success: {tamil_success}")