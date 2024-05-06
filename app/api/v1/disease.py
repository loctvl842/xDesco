import base64
from io import BytesIO

from fastapi import APIRouter
from PIL import Image

from ai_kit import load_models
from ai_kit.jsw import calculate_diff, get_JSW
from ai_kit.utils import combine_prob_jsw

router = APIRouter(prefix="/disease", tags=["Disease"])


@router.post("/diagnose")
async def diagnose(base64_data: str):
    seg_model, classif_model, rf, anomaly_extractor = load_models()

    contents = base64.b64decode(base64_data)
    img = Image.open(BytesIO(contents))

    mask = seg_model.segment(img)

    left_distances, right_distances = get_JSW(mask, dim=10, verbose=0)
    jsw_m, jsw_mm = calculate_diff(left_distances, right_distances)

    probability = classif_model.predict(img)
    prob_jsw = combine_prob_jsw(probability, jsw_m, jsw_mm)

    # output tra ve FE
    output = rf.predict([prob_jsw])[0]
    anomaly_map = anomaly_extractor.extract(mask, img, verbose=0)
    print(output, anomaly_map)
    print(type(output))
    print(type(anomaly_map))

    return {"class": int(output)}
