# from gcp import WasteService
# service = WasteService()

import json
import os
from vertexai.generative_models import GenerativeModel, Part, Image

def img_classify(image_path, credentiasls='gcpkey.json'):
    # return service.classify(image_path)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentiasls

    model = GenerativeModel('gemini-1.5-flash-002')

    generationconfig = {
                        'temperature': 0,
                        'response_mime_type': 'application/json'
    }

    prompt = '''
    告訴我這是這是甚麼類的垃圾而已 (例如：寶特瓶, 鋁箔包, 塑膠, 牛奶盒, 一般垃圾, 玻璃)且敘述信心分數輸出為json
    task1 : 例如看到寶特瓶, 就說寶特瓶
    task2: 如果不是垃圾, 說一般垃圾 JSON,
    task3: 是便當盒, 說便當盒
    task4: 回傳結果是英文
    例如1：  {"class_name": "plastic", 
            "recycle": Ture,
            "confidence": 0.98}
    例如2：  {"class_name": "一般垃圾", 
            "recycle": False,
            "confidence": 0.99}
    '''
    image_part = Part.from_image(Image.load_from_file(image_path))

    response = model.generate_content([prompt, image_part], generation_config=generationconfig)
    data = json.loads(response.text)
    return data['class_name'], data['recycle']

if __name__ == '__main__':
    print(img_classify('images/image.jpg'))