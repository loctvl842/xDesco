import base64
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

    # Assuming anomaly_map is a TensorFlow tensor
    # Convert TensorFlow tensor to NumPy array
    anomaly_map_np = anomaly_map.numpy()

    # Normalize the array to be between 0 and 255 (if necessary)
    anomaly_map_np = (
        (anomaly_map_np - np.min(anomaly_map_np)) / (np.max(anomaly_map_np) - np.min(anomaly_map_np))
    ) * 255

    # Convert the NumPy array to PIL image
    anomaly_map_img = Image.fromarray(anomaly_map_np.astype("uint8"))

    # Convert the PIL image to bytes
    img_byte_array = BytesIO()
    anomaly_map_img.save(img_byte_array, format="PNG")
    img_bytes = img_byte_array.getvalue()

    # Encode bytes to base64
    base64_encoded_image = base64.b64encode(img_bytes).decode("utf-8")

    # print(output, anomaly_map)
    # print(type(output))
    # print(type(anomaly_map))

    return {"class": int(output), "anomaly_map": base64_encoded_image}
