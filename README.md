# Alzheimer's Disease Detection using Graph Neural Networks

## Overview

This project detects Alzheimer's Disease from MRI brain scans using Graph Neural Networks (GNNs).

The pipeline includes:

* MRI preprocessing
* Harvard Oxford Brain Atlas
* ROI feature extraction
* Graph construction
* Graph Attention Network (GAT)
* Uncertainty estimation
* Explainability analysis

---

## Dataset

OASIS-1 MRI Dataset

Total Subjects: 235

Total MRI Scans Processed: 470

---

## Features Extracted

For each brain region:

* Mean Intensity
* Standard Deviation
* Minimum Intensity
* Maximum Intensity
* Region Volume

Total Regions: 48

---

## Model

Graph Attention Network (GAT)

Input Features: 5

Hidden Dimension: 64

Output Classes:

* 0 = Normal
* 1 = Alzheimer's

---

## Results

Validation Accuracy: 70%

Test Accuracy: 57.75%

Precision: 83.33%

Recall: 14.71%

F1 Score: 25%

---

## Explainability

The project identifies important brain regions associated with Alzheimer's disease, including:

* Parahippocampal Gyrus
* Temporal Fusiform Cortex
* Inferior Temporal Gyrus
* Insular Cortex
* Cingulate Gyrus

---

## Future Improvements

* Multimodal MRI Fusion
* Advanced Attention Visualization
* Clinical Decision Support Dashboard
* Larger MRI Datasets
