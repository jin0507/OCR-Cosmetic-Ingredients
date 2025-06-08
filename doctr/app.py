import os
import io
import redis
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.models.preprocessor import PreProcessor
from doctr.models.predictor import OCRPredictor
from doctr.models import vitstr_base, linknet_resnet50
from doctr.datasets import VOCABS
import cv2
import traceback

# Initialize FastAPI app
app = FastAPI(title="DocTR OCR Service")

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

# Load detection model
print("Loading DocTR detection model...")
det_model = linknet_resnet50(pretrained=False)
det_model.load_state_dict(torch.load("checkpoints/detection/linknet_resnet50_20250525-201345.pt", map_location="cpu"))
if torch.cuda.is_available():
    det_model = det_model.cuda()

# Load recognition model
print("Loading DocTR recognition model...")
reco_model = vitstr_base(pretrained=False)
reco_model.load_state_dict(torch.load("checkpoints/recognition/vitstr_base_20250514-042745.pt", map_location="cpu"))
if torch.cuda.is_available():
    reco_model = reco_model.cuda()

# Set up detection predictor
det_predictor = DetectionPredictor(
    PreProcessor(
        (1024, 1024),
        batch_size=1,
        mean=(0.798, 0.785, 0.772),
        std=(0.264, 0.2749, 0.287)
    ),
    det_model
)

# Set up recognition predictor
reco_predictor = RecognitionPredictor(
    PreProcessor(
        (32, 128),
        preserve_aspect_ratio=True,
        batch_size=32,
        mean=(0.694, 0.695, 0.693),
        std=(0.299, 0.296, 0.301)
    ),
    reco_model
)

# Combine both predictors
predictor = OCRPredictor(det_predictor, reco_predictor)

print("DocTR models loaded successfully!")


class RecognizeRequest(BaseModel):
    preprocessed_image_id: str


def ensure_rgb_image(image: np.ndarray) -> np.ndarray:
    """
    Ensure the image is in RGB format with 3 channels.
    
    Args:
        image (np.ndarray): Input image
    
    Returns:
        np.ndarray: RGB image with 3 channels
    """
    # Nếu ảnh là grayscale
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Nếu ảnh có 4 kênh (RGBA)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Nếu ảnh là BGR (OpenCV format)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    raise ValueError(f"Unsupported image format with {image.shape[2]} channels")


@app.get("/")
def read_root():
    return {"message": "DocTR OCR Service is running"}

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "doctr"}


@app.post("/recognize")
async def recognize_text(request: RecognizeRequest):
    try:
        # Lấy ảnh từ Redis
        image_bytes = redis_client.get(f"preprocessed:{request.preprocessed_image_id}")
        if not image_bytes:
            raise HTTPException(status_code=404, detail="Preprocessed image not found")
        
        # Chuyển đổi thành ảnh numpy
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        # Lưu kích thước gốc của ảnh
        original_height, original_width = image.shape[:2]
        
        # Đảm bảo ảnh là RGB
        image = ensure_rgb_image(image)
        
        # Thực hiện OCR
        result = predictor([image])
        
        # Chuyển kết quả thành định dạng JSON-serializable
        ocr_results = []
        try:
            for page_idx, page in enumerate(result.pages):
                for block_idx, block in enumerate(page.blocks):
                    for line_idx, line in enumerate(block.lines):
                        line_text = " ".join([word.value for word in line.words])
                        
                        confidences = [word.confidence for word in line.words]
                        line_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                        
                        geometry = line.geometry
                        
                        bbox = []
                        
                        if isinstance(geometry, np.ndarray):
                            for point in geometry:
                                bbox.append([int(point[0] * original_width), int(point[1] * original_height)])
                        else:
                            # Straight bbox
                            (xmin, ymin), (xmax, ymax) = geometry
                            bbox = [
                                [int(xmin * original_width), int(ymin * original_height)],
                                [int(xmax * original_width), int(ymin * original_height)],
                                [int(xmax * original_width), int(ymax * original_height)],
                                [int(xmin * original_width), int(ymax * original_height)]
                            ]
                        
                        region_id = len(ocr_results)
                        
                        ocr_results.append({
                            "text": line_text,
                            "confidence": float(line_confidence),
                            "bbox": bbox,
                            "region_id": region_id,  
                            "block_idx": block_idx,
                            "line_idx": line_idx,
                            "word_idx": -1,  
                            "original_size": [original_height, original_width]
                        })
        except Exception as e:
            print(f"Error formatting results: {e}")
            traceback.print_exc()
        
        print(f"DocTR found {len(ocr_results)} text regions")
        if ocr_results:
            print(f"Sample result: {ocr_results[0]}")
        print(ocr_results)
        return {"result": ocr_results}
    
    except Exception as e:
        print(f"Error in OCR process: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error recognizing text: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)