from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

import os

def generate_report(
    patient_name,
    prediction,
    confidence,
    brain_regions,
    heatmap_path,
    output_pdf
):

    doc = SimpleDocTemplate(
        output_pdf
    )

    styles = (
        getSampleStyleSheet()
    )

    content = []

    content.append(
        Paragraph(
            "Alzheimer MRI Report",
            styles["Title"]
        )
    )

    content.append(
        Spacer(1,12)
    )

    content.append(
        Paragraph(
            f"Patient: {patient_name}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Prediction: {prediction}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Confidence: "
            f"{confidence:.2f}%",
            styles["Normal"]
        )
    )

    content.append(
        Spacer(1,12)
    )

    content.append(
        Paragraph(
            "Important Brain Regions",
            styles["Heading2"]
        )
    )

    for region in brain_regions:

        content.append(
            Paragraph(
                region,
                styles["Normal"]
            )
        )

    content.append(
        Spacer(1,12)
    )

    if os.path.exists(
        heatmap_path
    ):
        content.append(
            Image(
                heatmap_path,
                width=400,
                height=250
            )
        )

    content.append(
        Spacer(1,12)
    )

    content.append(
        Paragraph(
            """
Doctor Summary:

This AI system detected
patterns associated with
Alzheimer's disease.

This report should be used
for screening purposes only.
Clinical evaluation is
recommended.
""",
            styles["Normal"]
        )
    )

    doc.build(content)

    return output_pdf