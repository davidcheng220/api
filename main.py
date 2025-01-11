from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file 
from tensorflow.keras.utils import load_img 
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow import expand_dims
from tensorflow.nn import softmax
import numpy as np

import numpy as np
from numpy import argmax
from numpy import max
from numpy import array
from json import dumps
from uvicorn import run
from typing import Annotated
import os
import io
from PIL import Image
from fastapi.responses import RedirectResponse

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

class_predictions = array([
'battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash'
])
#load model
model = load_model("model_fixed.h5")
#get method summary
@app.get("/")  
async def root():
     return RedirectResponse(url="/docs#")

def predict_image(img: Image.Image):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

#post method to get prediction, 當檔案傳輸的時候會使用post
@app.post("/predict/")
async def get_image_prediction(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read()))
    prediction = predict_image(img)
    confidence = prediction[0]
#     class_names = class_predictions
#   # 获取所有类别和对应置信度
#     all_predictions = [
#         {"class_name": class_names[i], "confidence": float(confidence[i])}
#         for i in range(len(class_names))
#     ]
    
#     # 根据置信度排序（可选）
#     all_predictions = sorted(all_predictions, key=lambda x: x["confidence"], reverse=True)
    
#     return {"predictions": all_predictions}
    predicted_index = np.argmax(confidence)
    class_names = class_predictions
    predicted_class = class_names[predicted_index]
    confidence = confidence[(predicted_index)]
    return {"class_name": predicted_class, "confidence": float(confidence)}
    
@app.post("/predict-all/")
async def get_image_prediction(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read()))
    prediction = predict_image(img)
    confidence = prediction[0]
    class_names = class_predictions
  # 获取所有类别和对应置信度
    all_predictions = [
        {"class_name": class_names[i], "confidence": float(confidence[i])}
        for i in range(len(class_names))
    ]
    
    # 根据置信度排序（可选）
    all_predictions = sorted(all_predictions, key=lambda x: x["confidence"], reverse=True)
    
    return {"predictions": all_predictions}


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    run(app, host="0.0.0.0", port=port)
