from fastapi import APIRouter, File, UploadFile

from ai_kit import load_models
from ai_kit.utils import read_image, combine_prob_jsw
from ai_kit.jsw import get_JSW, calculate_diff

router = APIRouter(prefix="/disease", tags=["Disease"])


@router.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    seg_model, classif_model, rf, anomaly_extractor = load_models()
    contents = await file.read()
    print(contents)
    img = read_image(contents)

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
