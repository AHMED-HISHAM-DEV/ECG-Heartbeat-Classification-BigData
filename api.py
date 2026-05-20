# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np, joblib

# تحميل الموديل 
app     = FastAPI()  
# hena p7mel el files el ana pnetha henak
model   = joblib.load("ecg_model.pkl")
scaler  = joblib.load("ecg_scaler.pkl")
sel     = joblib.load("ecg_selector.pkl")

CLASSES = {0:"Normal", 1:"Type1", 2:"Type2", 3:"Type3", 4:"Type4"}

# hena pykol le el api eny mestany list of numbers 
class ECGInput(BaseModel):
    signal: list[float]

# end point 
@app.post("/predict")
def predict(data: ECGInput):
    # hena p2a el request el gau  men el mobile shel meno el column el zayada 
    X    = sel.transform([data.signal])
    # pa5le el valueus men 0 le 1
    X    = scaler.transform(X)
    
    pred = model.predict(X)[0]
    
    conf = model.predict_proba(X)[0].max()
    return {
        "prediction": int(pred),
        "label":      CLASSES[pred],
        "confidence": f"{conf:.2%}"
    }