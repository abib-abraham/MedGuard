# MedGuard — Medicine Package Screening System

**GreenTech 2026 | Phase II Project**

**Student:** Abi B. Abraham  
**Register Number:** 23215201  
**Course:** B.Sc. Data Science & Artificial Intelligence

## 1. Project Overview

MedGuard is a web-based research prototype that performs preliminary medicine-package screening using image preprocessing, Optical Character Recognition (OCR), medicine-related text indicators, and experimental image classification.

The system helps users inspect visible information on medicine packaging. It extracts text, normalizes the OCR output, detects medicine-related indicators, and displays an experimental CNN prediction.

**Important:** MedGuard is a research prototype. It cannot certify whether a medicine is genuine or counterfeit. Its results require independent verification.

## 2. Problem Statement

Medicine packaging can contain small, rotated, reflective, blurred, or partially obscured text. These conditions can make manual inspection and reading of important details difficult.

MedGuard explores how computer vision, OCR, and machine learning can support preliminary inspection of visible medicine-package information.

## 3. Objectives

- Extract visible text from medicine-package photographs.
- Improve OCR input through image preprocessing and contrast enhancement.
- Normalize extracted text for downstream screening.
- Detect medicine-related text indicators.
- Generate an experimental image-classification result.
- Display screening findings, model output, and available batch or expiry information.
- Communicate uncertainty and limitations clearly.

## 4. Technologies Used

- **Python** — application logic and processing
- **Streamlit** — web-based user interface
- **EasyOCR** — text extraction
- **OpenCV** — image preprocessing and enhancement
- **NumPy** — image and numerical operations
- **TensorFlow/Keras** — CNN model inference
- **MobileNetV2-based CNN** — experimental image classification
- **Git and GitHub** — source-code version control

## 5. System Workflow

1. The user uploads a medicine-package photograph.
2. The image preprocessing module resizes and enhances the image.
3. EasyOCR extracts visible package text.
4. The OCR output is normalized.
5. The screening modules identify medicine-related text indicators.
6. The CNN generates an experimental image-classification output.
7. The fusion module combines available screening information.
8. Streamlit displays the findings, model score, and relevant limitations.

## 6. System Architecture

**User Interface:** `app/app.py`

**Image Preprocessing:** `src/image_preprocessing.py`

**Medicine Identification and Screening:** `src/stage1_identify.py`

**Experimental Authenticity Classification:** `src/stage2_authenticity.py`

**OCR and Text Extraction:** `src/stage3_ocr.py`

**Result Fusion:** `src/fusion.py`

**Model File:** `models/authenticity_cnn.h5`

The application coordinates these components to produce a preliminary screening report.

## 7. Installation and Execution

### Prerequisites

- Python 3.12 recommended for the development environment
- Git
- pip
- A Windows, Linux, or macOS environment compatible with the listed dependencies

### Step 1: Clone the repository

```bash
git clone https://github.com/abib-abraham/MedGuard.git
cd MedGuard
```

### Step 2: Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, consult the Python virtual-environment documentation or use Command Prompt activation instead.

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Some dependencies, particularly OCR and deep-learning libraries, may require additional platform-specific setup.

### Step 4: Run the application

```bash
python -m streamlit run app/app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

Ensure that the required model file exists at `models/authenticity_cnn.h5` before running the application.

## 8. Implementation Progress

The Phase II prototype includes:

- Streamlit-based application interface
- Medicine-package image upload
- Image preprocessing and contrast enhancement
- OCR text extraction and normalization
- Medicine-related indicator screening
- CNN-based experimental image analysis
- Combined screening output
- OCR diagnostics and available batch or expiry information

## 9. Sample Observed Result

In one documented test using a Vazidol package photograph:

- Total processing time: 46.7 seconds
- OCR processing time: 44.8 seconds
- CNN processing time: 1.9 seconds
- Detected indicators: dosage, mg, varenicline, warning
- Experimental CNN output: `Genuine`
- Reported model score: `0.6182`

These values describe one observed run, not a general performance guarantee. The model output is experimental and does not establish medicine authenticity.

Batch-number and expiry-date extraction can fail, and OCR may misread small, rotated, reflective, handwritten, or blurred text.

## 10. Dataset and Model Notes

The project documentation references the following datasets:

- [Fake vs Real Medicine Dataset — Kaggle](https://www.kaggle.com/datasets/surajkumarjha1/fake-vs-real-medicine-datasets-images)
- [Mobile-Captured Pharmaceutical Medication Packages — Kaggle](https://www.kaggle.com/datasets/aryashah2k/mobile-captured-pharmaceutical-medication-packages)

The second dataset is documented for product identification and reference matching, not as a counterfeit-classification dataset.

Verified held-out accuracy, precision, recall, F1-score, and the model's confirmed class-to-score mapping have not been established in the current project documentation. These metrics should not be claimed without a measured evaluation.

## 11. Sustainability Relevance

MedGuard has a potential connection to **United Nations Sustainable Development Goal 3 (SDG 3): Good Health and Well-Being** through its exploration of medicine-information awareness.

The project also provides an opportunity to investigate efficient image processing and model inference. Its real-world health impact and environmental benefits have not yet been measured.

## 12. Limitations

- The prototype does not authenticate medicines.
- OCR accuracy depends on image quality and package conditions.
- Batch and expiry information may be missing or incorrectly extracted.
- The CNN output is experimental and requires rigorous validation.
- No verified held-out classification metrics are currently documented.
- Users should independently verify package information with a pharmacist, manufacturer, or relevant authority.

## 13. Future Scope

- Improve OCR for rotated, reflective, and low-quality images.
- Improve batch-number and expiry-date extraction and validation.
- Evaluate the CNN using a labelled, independent test set.
- Report accuracy, precision, recall, F1-score, and a confusion matrix after evaluation.
- Explore trusted medicine-product reference data where legally and technically appropriate.
- Measure inference time, model size, memory consumption, and possible model-optimization benefits.
- Improve uncertainty handling and the clarity of screening reports.

## 14. Repository

GitHub: https://github.com/abib-abraham/MedGuard

---

**Project:** MedGuard — Medicine Package Screening System  
**Student:** Abi B. Abraham  
**Register Number:** 23215201  
**Academic Project:** GreenTech 2026 — Phase II

**Disclaimer:** This software is for research and educational purposes only. It must not be used as the sole basis for deciding whether a medicine is genuine, counterfeit, safe, or suitable for use.
