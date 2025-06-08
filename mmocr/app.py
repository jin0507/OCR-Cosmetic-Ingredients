import os
import io
import redis
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os.path as osp
import mmcv
import mmengine
from mmocr.apis.inferencers import MMOCRInferencer
from typing import List, Union, Sequence, Optional
import traceback
import math
import cv2 

InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]

# Initialize FastAPI app
app = FastAPI(title="MMOCR Service")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to Redis
redis_host = os.environ.get("REDIS_HOST", "localhost")
redis_client = redis.Redis(host=redis_host, port=6379, db=0)


class ResizeWrapper:
    """Wrapper cho MMOCRInferencer để resize ảnh thành 640x640 trước khi inference."""

    def __init__(self, 
                 det: Optional[Union[str, dict]] = None,
                 det_weights: Optional[str] = None,
                 rec: Optional[Union[str, dict]] = None,
                 rec_weights: Optional[str] = None,
                 kie: Optional[Union[str, dict]] = None,
                 kie_weights: Optional[str] = None,
                 device: Optional[str] = None) -> None:
        """Khởi tạo wrapper.
        
        Args:
            det: Cấu hình hoặc tên model detection
            det_weights: Đường dẫn đến model detection weights
            rec: Cấu hình hoặc tên model recognition
            rec_weights: Đường dẫn đến model recognition weights
            kie: Cấu hình hoặc tên model KIE
            kie_weights: Đường dẫn đến model KIE weights
            device: Thiết bị để inference
        """
        self.inferencer = MMOCRInferencer(
            det=det,
            det_weights=det_weights,
            rec=rec,
            rec_weights=rec_weights,
            kie=kie,
            kie_weights=kie_weights,
            device=device
        )
    
    def __call__(self, 
                 inputs,
                 batch_size: int = 1,
                 det_batch_size: Optional[int] = None,
                 rec_batch_size: Optional[int] = None,
                 kie_batch_size: Optional[int] = None,
                 out_dir: str = 'results/',
                 return_vis: bool = False,
                 save_vis: bool = False,
                 save_pred: bool = False,
                 **kwargs):
        """Thực hiện inference với resize ảnh trước.
        
        Args:
            inputs: Đường dẫn ảnh hoặc thư mục
            Các tham số khác giống với MMOCRInferencer.__call__
        """
        inputs_list = self._convert_inputs_to_list(inputs)
        
        # Resize ảnh
        resized_inputs = self._resize_inputs(inputs_list)
        
        return self.inferencer(
            inputs=resized_inputs,
            batch_size=batch_size,
            det_batch_size=det_batch_size,
            rec_batch_size=rec_batch_size,
            kie_batch_size=kie_batch_size,
            out_dir=out_dir,
            return_vis=return_vis,
            save_vis=save_vis,
            save_pred=save_pred,
            **kwargs
        )
    
    def _convert_inputs_to_list(self, inputs):
        """Chuyển đổi inputs thành danh sách."""
        if isinstance(inputs, str):
            if osp.isdir(inputs):
                files = [osp.join(inputs, f) for f in os.listdir(inputs) 
                        if osp.isfile(osp.join(inputs, f))]
                return files
            else:
                return [inputs]
        
        if not isinstance(inputs, (list, tuple)):
            return [inputs]
            
        return inputs
    
    def _resize_inputs(self, inputs_list):
        """Resize ảnh thành 640x640."""
        resized_inputs = []
        
        for item in inputs_list:
            if isinstance(item, str):
                # Đọc ảnh từ đường dẫn
                img_bytes = mmengine.fileio.get(item)
                img = mmcv.imfrombytes(img_bytes)
                
                # Resize ảnh thành 640x640
                img = mmcv.imresize(img, (640, 640))
                
                # Tạo thư mục tạm nếu cần thiết
                temp_dir = 'temp_resized'
                os.makedirs(temp_dir, exist_ok=True)
                
                # Lưu ảnh đã resize
                img_name = osp.basename(item)
                temp_path = osp.join(temp_dir, f"resized_{img_name}")
                mmcv.imwrite(img, temp_path)
                
                resized_inputs.append(temp_path)
            elif isinstance(item, np.ndarray):
                # Resize ảnh numpy array
                img = mmcv.imresize(item, (640, 640))
                resized_inputs.append(img)
            else:
                raise TypeError(f"Không hỗ trợ kiểu dữ liệu {type(item)}")
        
        return resized_inputs


# Initialize MMOCR model
print("Loading MMOCR models...")

# Directory to model and config
DET_CONFIG = "configs/text_det/checkpoint/dbnetpp/20250417_165101/20250417_165101/vis_data/config.py"
DET_WEIGHTS = "configs/text_det/checkpoint/dbnetpp/epoch_200.pth"
REC_CONFIG = "configs/text_rec/checkpoint/abinet/20250417_165704/20250417_165704/vis_data/config.py"
REC_WEIGHTS = "configs/text_rec/checkpoint/abinet/epoch_100.pth"

mmocr_model = ResizeWrapper(
    det=DET_CONFIG,
    det_weights=DET_WEIGHTS,
    rec=REC_CONFIG,
    rec_weights=REC_WEIGHTS,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

print(f"MMOCR models loaded successfully! Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")


class RecognizeRequest(BaseModel):
    preprocessed_image_id: str


@app.get("/")
def read_root():
    return {"message": "MMOCR Service is running"}

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "mmocr"}

def is_valid_float(value):
    """Kiểm tra xem một giá trị float có hợp lệ cho JSON không"""
    return isinstance(value, float) and not (math.isnan(value) or math.isinf(value))

@app.post("/recognize")
async def recognize_text(request: RecognizeRequest):
    try:
        image_bytes = redis_client.get(f"preprocessed:{request.preprocessed_image_id}")
        if not image_bytes:
            raise HTTPException(status_code=404, detail="Preprocessed image not found")
        
        temp_dir = "temp_images"
        os.makedirs(temp_dir, exist_ok=True)
        temp_image_path = os.path.join(temp_dir, f"{request.preprocessed_image_id}.png")
        
        with open(temp_image_path, "wb") as f:
            f.write(image_bytes)
        
        original_image = cv2.imread(temp_image_path)
        original_height, original_width = original_image.shape[:2]
        
        result = mmocr_model(
            temp_image_path,
            return_vis=False,
            save_vis=False,
            save_pred=False
        )
        
        ocr_results = []
        if result and 'predictions' in result and len(result['predictions']) > 0:
            predictions = result['predictions'][0]
            
            if 'det_polygons' in predictions and 'rec_texts' in predictions:
                det_polygons = predictions['det_polygons']
                rec_texts = predictions['rec_texts']
                rec_scores = predictions.get('rec_scores', [1.0] * len(rec_texts))
                
                for idx, (poly, text, score) in enumerate(zip(det_polygons, rec_texts, rec_scores)):
                    try:
                        score_value = float(score)
                        if not is_valid_float(score_value):
                            score_value = 0.0  
                    except:
                        score_value = 0.0
                    
                    bbox = []
                    try:
                        if isinstance(poly, np.ndarray):
                            points = poly.tolist()
                        elif isinstance(poly, list):
                            points = poly
                        else:
                            points = list(poly)  
                        
                        # Kích thước ảnh đã resize là 640x640
                        resize_width, resize_height = 640, 640
                        width_ratio = original_width / resize_width
                        height_ratio = original_height / resize_height
                        
                        # Chuyển đổi tọa độ về ảnh gốc
                        if points and isinstance(points[0], (list, tuple)):
                            for p in points:
                                x = p[0] * width_ratio
                                y = p[1] * height_ratio
                                bbox.append([int(x), int(y)])
                        else:
                            for i in range(0, len(points), 2):
                                if i+1 < len(points):
                                    x = points[i] * width_ratio
                                    y = points[i+1] * height_ratio
                                    bbox.append([int(x), int(y)])
                    except Exception as e:
                        print(f"Error formatting polygon: {e}")
                        bbox = []
                    
                    # Thêm vào kết quả với thông tin về kích thước gốc
                    ocr_results.append({
                        "text": text if isinstance(text, str) else str(text),
                        "confidence": score_value,
                        "bbox": bbox,
                        "region_id": idx,
                        "original_size": [original_height, original_width]
                    })
        
        # Clean up temporary file
        try:
            os.remove(temp_image_path)
        except:
            pass
        print(ocr_results)
        return {"result": ocr_results}
    
    except Exception as e:
        print(f"Error in recognize_text: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error recognizing text: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5003, reload=True)