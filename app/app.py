import os
import sys
import tempfile
import time

import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from stage1_identify import screen_medicine_image
from stage2_authenticity import run_authenticity_check
from stage3_ocr import run_ocr_check
from fusion import combine_results


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="MedGuard | Medicine Screening",
    page_icon="💊",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 1.5rem;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 16px;
        margin-bottom: 1rem;
    }
    .muted {
        color: var(--text-color);
        opacity: 0.75;
        font-size: 0.9rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        padding: 12px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>💊 MedGuard</h1>
        <p><strong>Medicine Package Screening System</strong></p>
        <p class="muted">
            Prototype screening, visible-label extraction,
            and preliminary image-model analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Research prototype only. MedGuard cannot certify a medicine as "
    "genuine or counterfeit. Verify packaging and expiry details with "
    "a pharmacist or manufacturer when in doubt."
)


# --------------------------------------------------
# IMAGE UPLOAD — MULTIPLE PHOTOS
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload photos of the same medicine",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help=(
        "Upload any number of photos of the same medicine. Include "
        "different sides and close-ups of the batch number and expiry date."
    ),
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.image(
            uploaded_file,
            caption=f"Uploaded package image: {uploaded_file.name}",
            use_container_width=True,
        )

    analyze_clicked = st.button(
        "Analyze package",
        type="primary",
        use_container_width=True,
    )

    if analyze_clicked:
        image_paths = []
        start_time = time.perf_counter()

        try:
            # Save all uploaded images temporarily
            for uploaded_file in uploaded_files:
                suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    image_paths.append(temp_file.name)

            # --------------------------------------
            # PIPELINE WITH SEPARATE TIMINGS
            # --------------------------------------

            with st.spinner(
                "Reading text from all package photos and analyzing the image..."
            ):
                # Run OCR on every uploaded photo
                ocr_start = time.perf_counter()

                ocr_results = [
                    run_ocr_check(image_path)
                    for image_path in image_paths
                ]

                ocr_seconds = time.perf_counter() - ocr_start

                # Start with the first OCR result to preserve
                # the original result structure
                ocr = dict(ocr_results[0]) if ocr_results else {}

                # Combine readable text from all photos
                raw_text_parts = [
                    result.get("raw_text", "").strip()
                    for result in ocr_results
                    if result.get("raw_text", "").strip()
                ]

                normalized_text_parts = [
                    result.get("normalized_text", "").strip()
                    for result in ocr_results
                    if result.get("normalized_text", "").strip()
                ]

                ocr["raw_text"] = "\n\n".join(raw_text_parts)
                ocr["normalized_text"] = "\n\n".join(
                    normalized_text_parts
                )

                # Use the first available extracted value for each field.
                # If photos disagree, manually verify the package.
                fields_to_merge = (
                    "batch_no",
                    "expiry_raw",
                    "expiry_parsed",
                    "mfg_parsed",
                    "expiry_rule_months",
                    "expiry_used_calculation",
                    "expiry_status",
                    "confidence",
                )

                for field in fields_to_merge:
                    for result in ocr_results:
                        value = result.get(field)

                        if value not in (None, "", False):
                            ocr[field] = value
                            break

                # Screen the combined OCR text
                screening = screen_medicine_image(
                    ocr.get("raw_text", "")
                )

                authenticity = None
                cnn_seconds = 0.0

                # Keep the existing CNN workflow.
                # CNN prediction is based on the first uploaded photo only.
                if screening.get("is_medicine_candidate", False):
                    cnn_start = time.perf_counter()

                    authenticity = run_authenticity_check(
                        image_paths[0]
                    )

                    cnn_seconds = time.perf_counter() - cnn_start

                # Preserve the existing fusion workflow
                final_result = combine_results(
                    screening,
                    authenticity,
                    ocr,
                )

            elapsed = time.perf_counter() - start_time

            st.success(
                f"Analysis completed in {elapsed:.1f} seconds"
            )

            # Actual timing measurements, not estimates.
            time_col1, time_col2 = st.columns(2)

            with time_col1:
                st.metric("OCR time", f"{ocr_seconds:.1f} s")

            with time_col2:
                st.metric("CNN time", f"{cnn_seconds:.1f} s")

            st.divider()

            # --------------------------------------
            # 1. IMAGE SCREENING
            # --------------------------------------

            st.subheader("1. Image screening")

            if screening.get("is_medicine_candidate", False):
                st.success(
                    "Medicine-related text indicators detected."
                )

                indicators = screening.get(
                    "matched_indicators", []
                )

                if indicators:
                    st.caption(
                        "Recognized indicators: "
                        + ", ".join(indicators[:12])
                    )

            else:
                st.warning(
                    "Unable to verify the image from the readable text."
                )

                st.write(
                    screening.get(
                        "reason",
                        "Not enough recognizable medicine-package text.",
                    )
                )

                st.info(
                    "Try a close-up photo with the label facing the camera. "
                    "This does not prove the object is not medicine."
                )

            # --------------------------------------
            # 2. CNN OUTPUT
            # --------------------------------------

            st.subheader("2. Experimental image-model output")

            if authenticity is None:
                st.write("CNN prediction: Not run")

                st.caption(
                    "The initial text screen did not pass, so the image "
                    "model was not evaluated."
                )

            else:
                label = authenticity.get(
                    "label", "Unavailable"
                )

                score = authenticity.get("raw_score")

                model_confidence = authenticity.get("confidence")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Model label", str(label))

                with col2:
                    st.metric(
                        "Model score",
                        f"{score:.4f}"
                        if score is not None
                        else "Unavailable",
                    )

                if model_confidence is not None:
                    st.caption(
                        "Model-reported confidence: "
                        f"{model_confidence * 100:.1f}%"
                    )

                st.warning(
                    authenticity.get(
                        "warning",
                        "Experimental prediction only. Model performance "
                        "has not been independently validated.",
                    )
                )

            # --------------------------------------
            # 3. BATCH AND EXPIRY
            # --------------------------------------

            st.subheader("3. Batch and expiry information")

            if screening.get("is_medicine_candidate", False):
                batch_no = ocr.get("batch_no")
                expiry_raw = ocr.get("expiry_raw")
                expiry_parsed = ocr.get("expiry_parsed")
                mfg_parsed = ocr.get("mfg_parsed")
                expiry_rule = ocr.get("expiry_rule_months")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Batch number",
                        str(batch_no)
                        if batch_no
                        else "Not detected",
                    )

                with col2:
                    displayed_expiry = expiry_raw

                    if not displayed_expiry and expiry_parsed:
                        displayed_expiry = expiry_parsed.strftime(
                            "%d %b %Y"
                        )

                    st.metric(
                        "Expiry date",
                        str(displayed_expiry)
                        if displayed_expiry
                        else "Not detected",
                    )

                if mfg_parsed:
                    st.caption(
                        "Detected manufacturing date: "
                        f"{mfg_parsed.strftime('%b %Y')}"
                    )

                if expiry_rule:
                    st.caption(
                        f"Printed expiry rule detected: "
                        f"{expiry_rule} months from manufacture."
                    )

                if ocr.get("expiry_used_calculation"):
                    st.warning(
                        "Expiry was calculated from a detected manufacturing "
                        "date and printed duration. Verify it against the package."
                    )

                expiry_status = ocr.get(
                    "expiry_status",
                    "Not detected",
                )

                if expiry_status == "expired":
                    st.warning(
                        "The extracted expiry date appears to be in the past. "
                        "Check the printed date manually."
                    )

                elif expiry_status == "not_expired":
                    st.info(
                        "The extracted date is not in the past. "
                        "This does not establish product safety."
                    )

                else:
                    st.warning(
                        "Expiry status unknown. No reliable expiry date "
                        "was extracted."
                    )

                if not batch_no or not expiry_parsed:
                    st.caption(
                        "Missing fields may be printed elsewhere on the "
                        "package, including the crimp."
                    )

                field_indicator = ocr.get("confidence")

                if field_indicator is not None:
                    st.caption(
                        "OCR field-extraction indicator: "
                        f"{field_indicator * 100:.0f}% "
                        "(field presence only, not OCR accuracy)."
                    )

            else:
                st.info(
                    "Batch and expiry details are not presented as verified "
                    "because the initial text screen did not pass."
                )

            # --------------------------------------
            # 4. OVERALL RESULT
            # --------------------------------------

            st.divider()

            st.subheader("4. Overall screening result")

            st.markdown(
                f"**{final_result.get('verdict', 'Result unavailable')}**"
            )

            st.write(
                final_result.get(
                    "explanation",
                    "Manual verification is required.",
                )
            )

            # --------------------------------------
            # ADVANCED OCR DIAGNOSTICS
            # --------------------------------------

            with st.expander("Advanced diagnostics · OCR details"):
                raw_text = ocr.get("raw_text", "")
                normalized_text = ocr.get("normalized_text", "")

                st.markdown("**Extracted OCR text**")

                if raw_text.strip():
                    st.text_area(
                        "Raw OCR",
                        value=raw_text,
                        height=160,
                        disabled=True,
                        label_visibility="collapsed",
                        key="raw_ocr_output",
                    )

                else:
                    st.caption(
                        "No readable text was extracted."
                    )

                st.markdown(
                    "**Normalized text used for extraction**"
                )

                if normalized_text.strip():
                    st.text_area(
                        "Normalized OCR",
                        value=normalized_text,
                        height=160,
                        disabled=True,
                        label_visibility="collapsed",
                        key="normalized_ocr_output",
                    )

                else:
                    st.caption(
                        "No normalized text is available."
                    )

                st.caption(
                    "OCR can misread small, rotated, reflective, or blurry "
                    "text. Verify all extracted dates and batch numbers "
                    "against the physical package."
                )

        except Exception as error:
            st.error(f"Analysis failed: {error}")

            st.caption(
                "Check the VS Code terminal for the full traceback."
            )

        finally:
            # Delete all temporary images
            for image_path in image_paths:
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except OSError:
                        pass