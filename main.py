from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file 
from tensorflow.keras.utils import load_img 
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow import expand_dims
from tensorflow.nn import softmax
import numpy as np
from numpy import argmax
from numpy import max
from numpy import array
from json import dumps
from uvicorn import run
from typing import Annotated
from PIL import Image
import os
import io
from vertexai.generative_models import GenerativeModel, Part
from fastapi.responses import RedirectResponse
import json
from pathlib import Path

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
     return RedirectResponse(url="/docs")

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

@app.post("/gcp-llm/")
async def get_image_prediction(file: UploadFile):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcpkey.json'
    
    model = GenerativeModel('gemini-1.5-flash-002')

    generationconfig = {
                        'temperature': 0,
                        'response_mime_type': 'application/json'
    }

    prompt = '''
    告訴我這是這是甚麼類的垃圾而已 (例如：寶特瓶, 鋁箔包, 塑膠, 牛奶盒, 一般垃圾)並輸出為json
    task1 : 例如看到寶特瓶, 就說寶特瓶
    task2: 如果不是垃圾, 說一般垃圾 JSON,
    例如1：  {"class_name": "plastic", 
            "recycle": Ture,
            "confidence": 0.98}
    例如2：  {"class_name": "一般垃圾", 
            "recycle": False,
            "confidence": 0.99}
    '''

    IMGDIR = Path("images")
    file_path = IMGDIR / file.filename

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    with open(file_path, "rb") as f:
        # data = Part.from_image(Image.load_from_file(str(file_path)))
        data = Part.from_data(data=f.read(), mime_type='image/jpeg')
        response1 = model.generate_content([prompt,data], generation_config=generationconfig)

    res = json.loads(response1.text)
    
    return res

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    run(app, host="0.0.0.0", port=port)
