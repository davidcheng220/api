import json
import os
from pathlib import Path
from typing import Union
from fastapi import UploadFile
from vertexai.generative_models import GenerativeModel, Part

class WasteService:
    def __init__(self, credentials_path: str = 'gcpkey.json'):
        
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        self.model = GenerativeModel('gemini-1.5-flash-002')
        self.config = {
            'temperature': 0,
            'response_mime_type': 'application/json'
        }
        
        self.prompt = '''
        告訴我這是這是甚麼類的垃圾而已 (例如：寶特瓶, 鋁箔包, 塑膠, 牛奶盒, 一般垃圾, 玻璃, 便當盒)且敘述信心分數輸出為json
        task1: 例如看到寶特瓶, 就說寶特瓶
        task2: 如果不是垃圾, 說一般垃圾 JSON,
        task3: 是便當盒, 說便當盒
        例如1：  {"class_name": "plastic", 
                "recycle": Ture,
                "confidence": 0.98}
        例如2：  {"class_name": "一般垃圾", 
                "recycle": False,
                "confidence": 0.99}
        '''

    def process_image(self, image_data: bytes) -> dict:
        """Process image data and return classification"""
        data = Part.from_data(data=image_data, mime_type='image/jpeg')
        response = self.model.generate_content([self.prompt, data], 
                                            generation_config=self.config)
        return json.loads(response.text)

    def classify(self, file: Union[str, UploadFile]) -> dict:
        """Classify waste from file path or UploadFile"""
        if isinstance(file, str):
            if not os.path.exists(file):
                raise FileNotFoundError(f"File not found: {file}")
            with open(file, "rb") as f:
                return self.process_image(f.read())
        
        elif isinstance(file, UploadFile):
            IMGDIR = Path("images")
            IMGDIR.mkdir(exist_ok=True)
            
            file_path = IMGDIR / file.filename
            content = file.file.read()
            
            with open(file_path, "wb") as f:
                f.write(content)
                
            try:
                with open(file_path, "rb") as f:
                    return self.process_image(f.read())
            finally:
                if file_path.exists():
                    file_path.unlink()
        else:
            raise TypeError("Input must be either a file path string or an UploadFile object")