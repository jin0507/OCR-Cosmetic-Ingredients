import os
import cv2
import torch
import numpy as np
import json
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from collections import defaultdict, OrderedDict
import io
import base64
from fastapi import FastAPI, HTTPException, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis
import traceback
import time
from typing import List, Optional, Dict, Any
from east_model import East

# ====== CONFIG ======
# Các đường dẫn linh hoạt để tải model
east_model_paths = [
    "/app/epoch_90_checkpoint.pth.tar",
    "./epoch_90_checkpoint.pth.tar",
]

trocr_model_paths = [
    "/app/seq2seq_model_printed/checkpoint-166002",
    "./seq2seq_model_printed/checkpoint-166002",
]

trocr_model_name = "microsoft/trocr-base-printed"

# Tham số phát hiện văn bản
score_map_thresh = 0.6
nms_thres = 0.5
merge_lines_enabled = False
y_overlap_threshold = 0.5

# Tham số mở rộng box - tối ưu hóa cho nhận dạng văn bản chính xác hơn
box_expand_ratio = 0.03  # Tăng nhẹ để bắt văn bản tốt hơn
horizontal_expand_factor = 0.2  # Tăng để bắt được các từ đầy đủ hơn

# Cấu hình GPU
gpu_ids = [0] if torch.cuda.is_available() else []
means = [100, 100, 100]
MAX_BATCH_SIZE = 16
MAX_REGIONS = 300

# Sử dụng CUDA nếu có
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Đang sử dụng thiết bị: {device}")

# Khởi tạo FastAPI
app = FastAPI(title="TrOCR Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kết nối Redis
redis_host = os.environ.get("REDIS_HOST", "localhost")
redis_client = redis.Redis(host=redis_host, port=6379, db=0)

# Định nghĩa các models
class RecognizeRequest(BaseModel):
    preprocessed_image_id: str

class RecognitionResult(BaseModel):
    text: str
    bbox: List[List[int]]
    region_id: int

class RecognizeResponse(BaseModel):
    result: List[RecognitionResult]

# Biến toàn cục cho models
east_model = None
processor = None
trocr_model = None

def resize_image(image, size=1024):
    """Resize ảnh cho EAST model với kích thước cố định hình vuông"""
    h, w = image.shape[:2]
    
    # Sử dụng kích thước vuông cố định để đảm bảo tính nhất quán
    target_height = size
    target_width = size
    
    # Resize về kích thước cố định
    resized = cv2.resize(image, (target_width, target_height))
    
    # Tính tỷ lệ để chuyển đổi tọa độ ngược lại
    h_ratio = h / target_height
    w_ratio = w / target_width
    
    return resized, (h_ratio, w_ratio)

def get_boxes_with_scores(score, geo, score_thresh=score_map_thresh):
    """Trích xuất boxes và scores từ đầu ra của EAST model"""
    score_map = score.detach()[0, 0].cpu().numpy()
    geo_map = geo.detach()[0].permute(1, 2, 0).cpu().numpy()
    
    yxs = np.argwhere(score_map > score_thresh)
    
    boxes = []
    scores = []
    
    for y, x in yxs:
        distances = geo_map[y, x, :4]  # top, right, bottom, left
        top, right, bottom, left = distances

        cx, cy = x * 4.0, y * 4.0  
        x0, y0 = int(cx - left), int(cy - top)
        x1, y1 = int(cx + right), int(cy + bottom)
        
        boxes.append([x0, y0, x1, y1])
        scores.append(score_map[y, x])
    
    return boxes, scores

def nms(boxes, scores, nms_threshold=nms_thres):
    """Non-Maximum Suppression để loại bỏ các box chồng lấp"""
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x1 - x0) * (y1 - y0)
    order = np.argsort(scores)[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx0 = np.maximum(x0[i], x0[order[1:]])
        yy0 = np.maximum(y0[i], y0[order[1:]])
        xx1 = np.minimum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        
        w = np.maximum(0.0, xx1 - xx0)
        h = np.maximum(0.0, yy1 - yy0)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        inds = np.where(iou <= nms_threshold)[0]
        order = order[inds + 1]
    
    return keep

def expand_box(box, expand_ratio=box_expand_ratio, horizontal_factor=horizontal_expand_factor, 
              image_width=None, image_height=None):
    """Mở rộng bounding box theo tỷ lệ được chỉ định đồng thời giữ trong giới hạn ảnh"""
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    
    # Áp dụng các hệ số mở rộng khác nhau theo chiều ngang và dọc
    x_expand = width * expand_ratio * (1 + horizontal_factor)
    y_expand = height * expand_ratio
    
    new_x0 = max(0, int(x0 - x_expand))
    new_y0 = max(0, int(y0 - y_expand))
    new_x1 = int(x1 + x_expand)
    new_y1 = int(y1 + y_expand)
    
    # Đảm bảo box nằm trong giới hạn ảnh
    if image_width:
        new_x1 = min(new_x1, image_width)
    if image_height:
        new_y1 = min(new_y1, image_height)
    
    return [new_x0, new_y0, new_x1, new_y1]

def convert_to_polygon_bbox(box, image_width=None, image_height=None):
    """
    Chuyển đổi bounding box từ định dạng [x1, y1, x2, y2] sang định dạng đa giác [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    phù hợp cho việc vẽ và hiển thị trên UI.
    
    Args:
        box: Bounding box định dạng [x1, y1, x2, y2]
        image_width: Chiều rộng của ảnh để đảm bảo box nằm trong giới hạn
        image_height: Chiều cao của ảnh để đảm bảo box nằm trong giới hạn
        
    Returns:
        Bounding box định dạng đa giác: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    """
    # Đảm bảo box có định dạng chính xác [x1, y1, x2, y2]
    if not (isinstance(box, list) and len(box) == 4):
        # Nếu định dạng không chính xác, trả về một hình chữ nhật mặc định
        print(f"Cảnh báo: Định dạng box không hợp lệ: {box}, sử dụng mặc định")
        return [[0, 0], [100, 0], [100, 100], [0, 100]]
    
    x1, y1, x2, y2 = box
    
    # Đảm bảo tọa độ là số nguyên
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Đảm bảo tọa độ nằm trong giới hạn ảnh
    if image_width is not None:
        x1 = max(0, min(x1, image_width - 1))
        x2 = max(0, min(x2, image_width))
    
    if image_height is not None:
        y1 = max(0, min(y1, image_height - 1))
        y2 = max(0, min(y2, image_height))
    
    # Tạo đa giác theo thứ tự đồng hồ bắt đầu từ góc trên bên trái
    polygon = [
        [int(x1), int(y1)],  # top-left
        [int(x2), int(y1)],  # top-right
        [int(x2), int(y2)],  # bottom-right
        [int(x1), int(y2)]   # bottom-left
    ]
    
    return polygon

def ocr_crop(image, box, processor, trocr_model):
    """Thực hiện OCR trên một vùng text được crop sử dụng TrOCR"""
    expanded_box = expand_box(box, 
                             expand_ratio=box_expand_ratio, 
                             horizontal_factor=horizontal_expand_factor,
                             image_width=image.shape[1], 
                             image_height=image.shape[0])
    
    x0, y0, x1, y1 = expanded_box
    crop = image[y0:y1, x0:x1]
    
    # Kiểm tra crop trống hoặc không hợp lệ
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return "", 0.0  # Trả về text trống và confidence bằng 0
    
    # Chuyển đổi sang PIL và resize về kích thước phù hợp cho TrOCR
    # Bảo toàn tỷ lệ khung hình tốt hơn cho nhận dạng văn bản
    try:
        pil_img = Image.fromarray(crop)
        # Tính toán kích thước bảo toàn tỷ lệ khung hình
        aspect = crop.shape[1] / crop.shape[0]  # width / height
        if aspect > 4:  # Text rất rộng
            new_width = min(128, crop.shape[1])  # Giới hạn chiều rộng cho văn bản rất rộng
            new_height = max(32, int(new_width / aspect))
        else:
            new_height = 32  # Chiều cao cố định để xử lý nhất quán
            new_width = max(32, min(128, int(new_height * aspect)))  # Bảo toàn tỷ lệ nhưng giới hạn chiều rộng tối đa
            
        pil = pil_img.resize((new_width, new_height), resample=Image.BILINEAR)
        
        # Xử lý ảnh với TrOCR
        inputs = processor(images=pil, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            outputs = trocr_model.generate(inputs)
        
        # Giải mã text đã dự đoán
        text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Ước tính độ tin cậy (sử dụng độ dài như một heuristic đơn giản)
        # Có thể thêm cách đo độ tin cậy tinh vi hơn ở đây
        confidence = min(0.9, max(0.5, len(text) / 30)) if text else 0.0
        
        return text, confidence
    except Exception as e:
        print(f"Lỗi trong quá trình xử lý OCR: {e}")
        return "", 0.0

def detect_text_boxes(image, east_model, processor, ocr_model, 
                     score_thresh=score_map_thresh, 
                     nms_thresh=nms_thres,
                     merge_lines=merge_lines_enabled,
                     y_overlap_threshold=y_overlap_threshold,
                     return_resized_info=True):
    """
    Pipeline hoàn chỉnh để phát hiện văn bản với EAST + OCR sử dụng TrOCR
    
    Args:
        image: ảnh RGB đầu vào
        east_model: model EAST đã load
        processor: TrOCR processor
        ocr_model: model TrOCR
        score_thresh: ngưỡng cho điểm tin cậy của EAST
        nms_thresh: ngưỡng cho non-maximum suppression
        merge_lines: có gộp các box cùng dòng không
        y_overlap_threshold: ngưỡng chồng lấp theo chiều y để xác định các box thuộc cùng dòng
        return_resized_info: có trả về thông tin ảnh đã resize không
        
    Returns:
        valid_boxes: các box văn bản cuối cùng
        all_texts: văn bản được nhận dạng cho mỗi box
        all_confidences: điểm tin cậy cho mỗi dự đoán
        resized_info: thông tin về ảnh đã resize (nếu return_resized_info=True)
    """
    # Lưu kích thước ảnh gốc
    orig_height, orig_width = image.shape[:2]
    resized, (h_ratio, w_ratio) = resize_image(image, size=1024)
    # Log thông tin kích thước
    print(f"Original image: {orig_width}x{orig_height}")
    print(f"Resize ratios: h_ratio={h_ratio}, w_ratio={w_ratio}")
    
    # Chuẩn bị tensor cho EAST model
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    # Chạy EAST model để lấy điểm số vùng text và hình học
    with torch.no_grad():
        score, geo = east_model(tensor)
    
    # Trích xuất boxes và scores từ đầu ra của EAST
    boxes, scores = get_boxes_with_scores(score, geo, score_thresh)
    print(f"Found {len(boxes)} text boxes before NMS")
    
    # Áp dụng non-maximum suppression
    if len(boxes) > 0:
        keep_indices = nms(boxes, scores, nms_thresh)
        nms_boxes = [boxes[i] for i in keep_indices]
        nms_scores = [scores[i] for i in keep_indices]
        print(f"After NMS: {len(nms_boxes)} text boxes")
    else:
        nms_boxes = []
        nms_scores = []
        print("No text boxes detected")
    
    # Chuyển đổi boxes về không gian ảnh đã resize
    resized_boxes = []
    for box in nms_boxes:
        resized_box = np.array(box, dtype=np.int32)
        resized_boxes.append(resized_box)
    
    # Nhận dạng văn bản trong mỗi box
    all_texts = []
    all_confidences = []
    valid_boxes = []
    valid_resized_boxes = []  # Lưu lại boxes trong không gian ảnh đã resize
    
    print("🔍 Performing OCR on detected boxes...")
    for i, box in enumerate(resized_boxes):
        # Trích xuất văn bản từ box
        text, confidence = ocr_crop(resized, box, processor, ocr_model)
        
        if text.strip():
            all_texts.append(text)
            all_confidences.append(confidence)
            
            # Chuyển đổi box từ không gian ảnh đã resize về không gian ảnh gốc
            x0, y0, x1, y1 = box
            
            # Áp dụng tỷ lệ chính xác để chuyển đổi tọa độ về ảnh gốc
            orig_x0 = int(x0 * w_ratio)
            orig_y0 = int(y0 * h_ratio)
            orig_x1 = int(x1 * w_ratio)
            orig_y1 = int(y1 * h_ratio)
            
            # Đảm bảo tọa độ nằm trong giới hạn ảnh
            orig_x0 = max(0, min(orig_x0, orig_width - 1))
            orig_y0 = max(0, min(orig_y0, orig_height - 1))
            orig_x1 = max(0, min(orig_x1, orig_width))
            orig_y1 = max(0, min(orig_y1, orig_height))
            
            orig_box = [orig_x0, orig_y0, orig_x1, orig_y1]
            
            valid_boxes.append(orig_box)
            valid_resized_boxes.append(box)
            # print(f"Box {i+1}/{len(resized_boxes)}: \"{text}\"")
            # print(f"  Resized box: {box}")
            # print(f"  Original box: {orig_box}")
        else:
            print(f"Box {i+1}/{len(resized_boxes)}: Empty text, skipping")
    
    # Trả về kết quả với hoặc không có thông tin resize
    if return_resized_info:
        resized_info = {
            "original_size": (orig_width, orig_height),
            "resize_ratios": (w_ratio, h_ratio),
            "resized_boxes": valid_resized_boxes
        }
        return valid_boxes, all_texts, all_confidences, resized_info
    else:
        return valid_boxes, all_texts, all_confidences

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

def initialize_models():
    """Khởi tạo EAST và TrOCR models với xử lý đường dẫn cải tiến và khôi phục lỗi"""
    global east_model, processor, trocr_model
    
    try:
        # Tải EAST model
        east_model_loaded = False
        for path in east_model_paths:
            if os.path.exists(path):
                try:
                    print(f"[INFO] Đang tải EAST model từ {path}")
                    east_model = East().to(device)
                    
                    checkpoint = torch.load(path, map_location=device)
                    if 'state_dict' in checkpoint:
                        # Xử lý state dict với tiền tố 'module.' (từ DataParallel)
                        new_state_dict = OrderedDict()
                        for k, v in checkpoint['state_dict'].items():
                            name = k.replace('module.', '')
                            new_state_dict[name] = v
                        east_model.load_state_dict(new_state_dict)
                    else:
                        east_model.load_state_dict(checkpoint)
                        
                    east_model.eval()
                    print(f"[INFO] EAST model đã được tải thành công từ {path}")
                    east_model_loaded = True
                    break
                except Exception as e:
                    print(f"[ERROR] Không thể tải EAST model từ {path}: {e}")
                    traceback.print_exc()
        
        if not east_model_loaded:
            print("[WARNING] Không thể tải EAST model từ bất kỳ đường dẫn nào")
            return False
        
        # Tải TrOCR model
        trocr_model_loaded = False
        for path in trocr_model_paths:
            if os.path.exists(path):
                try:
                    print(f"[INFO] Đang tải TrOCR model từ {path}")
                    processor = TrOCRProcessor.from_pretrained(trocr_model_name)
                    trocr_model = VisionEncoderDecoderModel.from_pretrained(path).to(device)
                    trocr_model.eval()
                    print(f"[INFO] TrOCR model đã được tải thành công từ {path}")
                    trocr_model_loaded = True
                    break
                except Exception as e:
                    print(f"[ERROR] Không thể tải TrOCR model từ {path}: {e}")
                    traceback.print_exc()
        
        if not trocr_model_loaded:
            # Quay lại tải từ Hugging Face
            try:
                print("[INFO] Đang tải model TrOCR mặc định từ Hugging Face")
                processor = TrOCRProcessor.from_pretrained(trocr_model_name)
                trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_model_name).to(device)
                trocr_model.eval()
                print("[INFO] Model TrOCR mặc định đã được tải thành công")
                trocr_model_loaded = True
            except Exception as e:
                print(f"[ERROR] Không thể tải model TrOCR mặc định: {e}")
                traceback.print_exc()
                return False
        
        print(f"[INFO] Tất cả models đã được tải thành công. Sử dụng thiết bị: {device}")
        return east_model_loaded and trocr_model_loaded
            
    except Exception as e:
        print(f"[ERROR] Lỗi khi khởi tạo models: {str(e)}")
        traceback.print_exc()
        return False

# Khởi tạo model
if not initialize_models():
    print("[ERROR] Không thể khởi tạo models, dịch vụ có thể không hoạt động chính xác")

# Các endpoints
@app.get("/")
def read_root():
    return {"message": "TrOCR Service is running"}

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "trocr"}

@app.post("/recognize")
async def recognize_text(request: RecognizeRequest):
    """Nhận dạng văn bản trong ảnh đã được tiền xử lý"""
    try:
        if processor is None or trocr_model is None or east_model is None:
            if not initialize_models():
                raise HTTPException(status_code=500, detail="OCR models not initialized properly")
                
        # Lấy ảnh đã qua tiền xử lý từ Redis
        image_bytes = redis_client.get(f"preprocessed:{request.preprocessed_image_id}")
        if not image_bytes:
            raise HTTPException(status_code=404, detail="Preprocessed image not found")
        
        # Giải mã ảnh
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Chuyển sang định dạng RGB để xử lý
        image_rgb = ensure_rgb_image(image)
        
        # Phát hiện và nhận dạng văn bản
        boxes, texts, confidences, resized_info = detect_text_boxes(
            image_rgb, east_model, processor, trocr_model,
            score_thresh=score_map_thresh,
            nms_thresh=nms_thres,
            return_resized_info=True
        )

        # Chuyển đổi tất cả boxes sang định dạng đa giác
        polygon_boxes = []
        for box in boxes:
            polygon = convert_to_polygon_bbox(
                box, 
                image_width=image_rgb.shape[1], 
                image_height=image_rgb.shape[0]
            )
            polygon_boxes.append(polygon)
        
        # Tạo kết quả phản hồi
        ocr_results = []
        for i, (polygon, text, confidence) in enumerate(zip(polygon_boxes, texts, confidences)):
            ocr_results.append({
                        "text": text,
                        "confidence": float(confidence),
                        "bbox": polygon,
                        "region_id": i, 
                    })
        # return results

        # In ra cấu trúc kết quả để debug
        print("Return structure:")
        print(ocr_results)
        
        return {"result": ocr_results}
            
    except Exception as e:
        print(f"[ERROR] Lỗi khi nhận dạng văn bản: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error recognizing text: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5002, reload=True)