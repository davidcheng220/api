from fastapi import FastAPI, UploadFile, File, HTTPException, Query 
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
# from tensorflow.keras.utils import get_file 
# from tensorflow.keras.utils import load_img 
# from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
# from tensorflow import expand_dims
# from tensorflow.nn import softmax
import numpy as np
# from numpy import argmax
# from numpy import max
from numpy import array
# from json import dumps
from uvicorn import run
# from typing import Annotated
from PIL import Image
import os
import io
from vertexai.generative_models import GenerativeModel, Part
from fastapi.responses import RedirectResponse, StreamingResponse
import json
from pathlib import Path
from gcp import WasteService
# from yoloapi import *
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
import mysql.connector 

app = FastAPI()
service = WasteService()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    # HTTPSRedirectMiddleware,
    # allow_host=["*"]
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

# def predict_image(img: Image.Image):
#     img = img.resize((224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     prediction = model.predict(img_array)
#     return prediction

# #post method to get prediction, 當檔案傳輸的時候會使用post
# @app.post("/predict/")
# async def get_image_prediction(file: UploadFile):
#     img = Image.open(io.BytesIO(await file.read()))
#     prediction = predict_image(img)
#     confidence = prediction[0]

#     predicted_index = np.argmax(confidence)
#     class_names = class_predictions
#     predicted_class = class_names[predicted_index]
#     confidence = confidence[(predicted_index)]
#     return {"class_name": predicted_class, "confidence": float(confidence)}
    
# @app.post("/predict-all/")
# async def get_image_prediction(file: UploadFile):
#     img = Image.open(io.BytesIO(await file.read()))
#     prediction = predict_image(img)
#     confidence = prediction[0]
#     class_names = class_predictions
#   # 所有可信度
#     all_predictions = [
#         {"class_name": class_names[i], "confidence": float(confidence[i])}
#         for i in range(len(class_names))
#     ]
    
#     # 序列化由高到低
#     all_predictions = sorted(all_predictions, key=lambda x: x["confidence"], reverse=True)
    
#     return {"predictions": all_predictions}

@app.post("/gcp-llm/")
async def get_image_prediction(file: UploadFile):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcpkey.json'
    
    model = GenerativeModel('gemini-1.5-pro-002')

    generationconfig = {
                        'temperature': 0,
                        'response_mime_type': 'application/json'
    }

    prompt = '''
    告訴我這是這是甚麼類的垃圾而已 (例如：寶特瓶, 鋁箔包, 塑膠, 牛奶盒, 一般垃圾, 玻璃, 瓶子)且敘述信心分數輸出為json
    task1 : 例如看到寶特瓶, 就說寶特瓶
    task2: 如果不是垃圾, 說一般垃圾 JSON
    task3: 是便當盒, 說便當盒
    task4: 信心度請依照你的感覺並說0.90~0.70之間
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

@app.post("/classify/")
async def classify_waste(file: UploadFile):
    """Endpoint to classify waste from uploaded images"""
    # Create images directory if it doesn't exist
    IMGDIR = Path("images")
    IMGDIR.mkdir(exist_ok=True)
    
    # Save uploaded file temporarily
    temp_path = IMGDIR / file.filename
    # try:
    content = await file.read()
    with open(temp_path, "wb") as f:
        f.write(content)
    
    # Process the image
    result = service.classify(str(temp_path))
    return result
    # finally:
    #     # Clean up temporary file
    #     if temp_path.exists():
    #         os.unlink(temp_path)



def get_db_connection():
    """
    請根據實際情況修改此函式，例如使用 mysql.connector.connect 來連線至 MySQL 資料庫。
    """
    try:
        conn = mysql.connector.connect(
            host="database-1.cnwykisoqb94.ap-northeast-1.rds.amazonaws.com",
            user="admin",
            password="#Tibame01",
            database="trash_2"
        )
        return conn
    except mysql.connector.Error as err:
        raise Exception(f"資料庫連線錯誤: {err}")
    
# 1. 查詢桶子
@app.get("/get_location")
async def get_location():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("select distinct(trash_loc) as 'bins' from trash_box;")
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return {'result': result}

# 2. 查詢桶子的時間
@app.get("/bin_time")
async def get_bin_time(bin1: str = Query(...)):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute(
                   """
                   SELECT * FROM trash_box i1 JOIN sensor s1 ON i1.bin_id = s1.trash_box_bin_id 
                   WHERE i1.trash_loc = %s 
                   ORDER BY s1.identify_time DESC
                   """, (bin1,)
                   )
    
    result = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return {"identify_time": [row["identify_time"] for row in result]}

# 3. 查詢一般垃圾的用量(full/not full)
@app.get("/for_general_is_full")
async def is_full(bin1: str = Query(...), time: str = Query(...)):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # 執行 SQL 查詢
    cursor.execute(
                    """
                    select results from trash_box i1 join sensor s1 on i1.bin_id = s1.trash_box_bin_id 
                    where trash_loc=%s and (identify_time = %s or identify_time < %s) 
                    and bin_name='一般垃圾' 
                    order by identify_time limit 1;
                    """,
                    (bin1, time, time)
    )

    result = cursor.fetchone()
    
    cursor.close()
    conn.close()

    return {"result": result["results"] if result else "No data found"}

# 4. 查詢資源回收的用量(full/not full)
@app.get("/for_recycle_is_full")
async def for_recycle_is_full(bin1: str = Query(...), time: str = Query(...)):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # 執行 SQL 查詢
    cursor.execute(
                    """
                    select results from trash_box i1 join sensor s1 on i1.bin_id = s1.trash_box_bin_id 
                    where trash_loc=%s and (identify_time = %s or identify_time < %s) 
                    and bin_name='資源回收' 
                    order by identify_time limit 1;
                    """,
                    (bin1, time, time)
    )

    result = cursor.fetchone()
    
    cursor.close()
    conn.close()

    return {"result": result["results"] if result else "No data found"}

# 5. 查詢今天一般垃圾的滿的狀況
@app.get("/for_general_others_time", response_model=dict)
def for_general_others_time(day: str = Query(..., description="查詢的日期，格式例如 '2025-01-01'")):
    result = []
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(f"SELECT distinct s1.identify_time FROM trash_box b1 JOIN sensor s1 ON b1.bin_id = s1.trash_box_bin_id JOIN img i1 ON s1.trash_box_bin_id = i1.trash_box_bin_id where i1.trash_category = '一般垃圾' and DATE(s1.identify_time) = '{day}' and s1.results='滿' and TIME(s1.identify_time) BETWEEN '09:00:00' AND '10:59:59';")
    nine_to_eleven = cursor.fetchall()
    cursor.execute(f"SELECT distinct s1.identify_time FROM trash_box b1 JOIN sensor s1 ON b1.bin_id = s1.trash_box_bin_id JOIN img i1 ON s1.trash_box_bin_id = i1.trash_box_bin_id where i1.trash_category = '一般垃圾' and DATE(s1.identify_time) = '{day}' and s1.results='滿' and TIME(s1.identify_time) BETWEEN '11:00:00' AND '12:59:59';")
    eleven_to_one = cursor.fetchall()
    cursor.execute(f"SELECT distinct s1.identify_time FROM trash_box b1 JOIN sensor s1 ON b1.bin_id = s1.trash_box_bin_id JOIN img i1 ON s1.trash_box_bin_id = i1.trash_box_bin_id where i1.trash_category = '一般垃圾' and DATE(s1.identify_time) = '{day}' and s1.results='滿' and TIME(s1.identify_time) BETWEEN '13:00:00' AND '14:59:59';")
    one_to_three = cursor.fetchall()
    cursor.execute(f"SELECT distinct s1.identify_time FROM trash_box b1 JOIN sensor s1 ON b1.bin_id = s1.trash_box_bin_id JOIN img i1 ON s1.trash_box_bin_id = i1.trash_box_bin_id where i1.trash_category = '一般垃圾' and DATE(s1.identify_time) = '{day}' and s1.results='滿' and TIME(s1.identify_time) BETWEEN '15:00:00' AND '17:00:00';")
    three_to_five = cursor.fetchall()


    if len(nine_to_eleven) >0:
        nine_to_eleven = '09:00:00- 11:00:00'
        result.append(nine_to_eleven)
    
    if len(eleven_to_one) > 0:
        eleven_to_one = '11:00:00- 13:00:00'
        result.append(eleven_to_one)
    
    if len(one_to_three) >0:
        one_to_three = '13:00:00- 15:00:00'
        result.append(one_to_three)
    
    if len(three_to_five) >0:
        three_to_five = '15:00:00- 17:00:00'
        result.append(three_to_five)
        
    if len(three_to_five) ==0:
        a = None
        result.append(a)
        
    return {'result':result}

@app.get("/for_recycle_others_time", response_model=dict)
def for_recycel_others_time(day: str = Query(..., description="查詢的日期，格式例如 '2025-01-01'")):
    result = []
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(f"SELECT distinct s1.identify_time FROM trash_box b1 JOIN sensor s1 ON b1.bin_id = s1.trash_box_bin_id JOIN img i1 ON s1.trash_box_bin_id = i1.trash_box_bin_id where i1.trash_category = '資源回收' and DATE(s1.identify_time) = '{day}' and s1.results='滿' and TIME(s1.identify_time) BETWEEN '09:00:00' AND '10:59:59';")
    nine_to_eleven = cursor.fetchall()
    cursor.execute(f"SELECT distinct s1.identify_time FROM trash_box b1 JOIN sensor s1 ON b1.bin_id = s1.trash_box_bin_id JOIN img i1 ON s1.trash_box_bin_id = i1.trash_box_bin_id where i1.trash_category = '資源回收' and DATE(s1.identify_time) = '{day}' and s1.results='滿' and TIME(s1.identify_time) BETWEEN '11:00:00' AND '12:59:59';")
    eleven_to_one = cursor.fetchall()
    cursor.execute(f"SELECT distinct s1.identify_time FROM trash_box b1 JOIN sensor s1 ON b1.bin_id = s1.trash_box_bin_id JOIN img i1 ON s1.trash_box_bin_id = i1.trash_box_bin_id where i1.trash_category = '資源回收' and DATE(s1.identify_time) = '{day}' and s1.results='滿' and TIME(s1.identify_time) BETWEEN '13:00:00' AND '14:59:59';")
    one_to_three = cursor.fetchall()
    cursor.execute(f"SELECT distinct s1.identify_time FROM trash_box b1 JOIN sensor s1 ON b1.bin_id = s1.trash_box_bin_id JOIN img i1 ON s1.trash_box_bin_id = i1.trash_box_bin_id where i1.trash_category = '資源回收' and DATE(s1.identify_time) = '{day}' and s1.results='滿' and TIME(s1.identify_time) BETWEEN '15:00:00' AND '17:00:00';")
    three_to_five = cursor.fetchall()


    if len(nine_to_eleven) >0:
        nine_to_eleven = '09:00:00- 11:00:00'
        result.append(nine_to_eleven)
    
    if len(eleven_to_one) > 0:
        eleven_to_one = '11:00:00- 13:00:00'
        result.append(eleven_to_one)
    
    if len(one_to_three) >0:
        one_to_three = '13:00:00- 15:00:00'
        result.append(one_to_three)
    
    if len(three_to_five) >0:
        three_to_five = '15:00:00- 17:00:00'
        result.append(three_to_five)
        
    if len(three_to_five) ==0:
        a = None
        result.append(a)
        
    return {'result':result}
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    run(app, host="0.0.0.0", port=port)
