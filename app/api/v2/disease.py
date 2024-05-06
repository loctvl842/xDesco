import base64
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from io import BytesIO

from fastapi import APIRouter
from PIL import Image
from pydantic import BaseModel

from ai_kit import load_models
from ai_kit.jsw import calculate_diff, get_JSW
from ai_kit.utils import combine_prob_jsw, read_image

router = APIRouter(prefix="/disease", tags=["Disease"])


class DiagnoseBody(BaseModel):
    image: str


@router.post("/diagnose")
async def diagnose(body: DiagnoseBody):
    seg_model, classif_model, rf, anomaly_extractor = load_models()

    # Add padding if necessary
    base64_image_string = body.image.split(",")[1]
    contents = base64.b64decode(base64_image_string)

    # Decode base64-encoded string
    img = read_image(contents)

    mask = seg_model.segment(img)

    left_distances, right_distances = get_JSW(mask, dim=10, verbose=0)
    jsw_m, jsw_mm = calculate_diff(left_distances, right_distances)

    probability = classif_model.predict(img)
    prob_jsw = combine_prob_jsw(probability, jsw_m, jsw_mm)

    # output tra ve FE
    output = rf.predict([prob_jsw])[0]
    anomaly_map = anomaly_extractor.extract(mask, img, verbose=0)

    plt.imshow(img, cmap="gray")
    plt.imshow(anomaly_map, cmap="turbo", alpha=0.3)
    plt.axis("off")

    # Save the rendered image to a buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buffer.seek(0)

    # Convert the buffer to base64
    base64_encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return {"class": int(output), "anomaly_map": base64_encoded_image}
