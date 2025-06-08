import openai
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import io
import time
import uuid
import httpx
import redis
import logging
from typing import Optional, List, Dict, Any
import json
import numpy as np
import cv2
import asyncio
import traceback
import csv
import random

# Thêm import cho static frontend
from static_frontend import mount_static_frontend

# Cấu hình OpenAI API key
OPENAI_API_KEY = "API key"
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo FastAPI
app = FastAPI(title="OCR Pipeline API")

# Cấu hình CORS
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

# Service URLs từ biến môi trường
PREPROCESSING_URL = os.environ.get("PREPROCESSING_URL", "http://preprocessing:5000")
DOCTR_URL = os.environ.get("DOCTR_URL", "http://doctr:5001")
TROCR_URL = os.environ.get("TROCR_URL", "http://trocr:5002")
MMOCR_URL = os.environ.get("MMOCR_URL", "http://mmocr:5003")

# Cấu hình timeout - Tăng lên để xử lý OCR với nhiều vùng văn bản
REQUEST_TIMEOUT = 180.0  # Tăng timeout lên 180 giây

# Mount frontend HTML
mount_static_frontend(app)

# Model definitions
class OCRRequest(BaseModel):
    image_id: str
    model: str = "doctr"  # Default model

class OCRResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

class GPTRequest(BaseModel):
    extracted_text: str
    prompt: Optional[str] = None

# Hàm tiện ích retry cho các HTTP request
async def fetch_with_retry(client, method, url, **kwargs):
    """Thực hiện HTTP request với cơ chế retry khi lỗi và health check."""
    max_retries = 8  # Increased max retries
    initial_delay = 5  # Increased initial delay
    max_delay = 60  # Increased max delay
    last_exception = None
    service_name = url.split("//")[1].split(":")[0]
    
    # Avoid unnecessary health checks on subsequent attempts
    health_check_attempted = False
    
    for attempt in range(max_retries):
        try:
            # Try health check first for services we know about, but only on first attempt
            if not health_check_attempted and any(service in url for service in ["preprocessing", "doctr", "trocr", "mmocr"]):
                try:
                    health_check_attempted = True
                    # Get base URL for health check
                    base_url = url.split("/")[0] + "//" + url.split("/")[2]
                    health_url = f"{base_url}/health"
                    logger.info(f"Checking health for {service_name} at {health_url}")
                    
                    health_timeout = 10.0  # Increased timeout for health check
                    health_response = await client.get(health_url, timeout=health_timeout)
                    
                    if health_response.status_code == 200:
                        logger.info(f"Health check passed for {service_name}")
                    else:
                        logger.warning(f"Health check failed for {service_name} with status code {health_response.status_code}")
                except Exception as e:
                    logger.warning(f"Health check failed for {service_name}: {str(e)}")
            
            # Add longer timeout for the main request
            if 'timeout' not in kwargs:
                kwargs['timeout'] = 60.0

            # Main request
            if method.lower() == "get":
                response = await client.get(url, **kwargs)
            elif method.lower() == "post":
                response = await client.post(url, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            if response.status_code >= 400:
                logger.warning(f"Request failed with status {response.status_code} for {service_name}: {response.text}")
                if response.status_code >= 500:
                    raise httpx.HTTPError(f"Server error: {response.status_code}")
                return response
                
            return response
            
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout, 
                httpx.WriteTimeout, httpx.PoolTimeout, httpx.NetworkError,
                httpx.ProxyError, httpx.HTTPError) as e:
            last_exception = e
            
            # Use exponential backoff with initial delay
            delay = min(max_delay, initial_delay * (2 ** attempt) + random.uniform(0, 1))
            
            logger.warning(
                f"Connection to {service_name} failed (attempt {attempt+1}/{max_retries}): {str(e)}. "
                f"Retrying in {delay:.2f} seconds..."
            )
            
            if attempt == max_retries - 1:
                logger.error(
                    f"All {max_retries} attempts failed for {service_name}. "
                    f"Make sure the service is running and healthy. Error: {str(e)}"
                )
                raise httpx.ConnectError(
                    f"Could not connect to {service_name} after {max_retries} attempts. "
                    "Please check if the service is running."
                ) from e
                
            await asyncio.sleep(delay)
    
    raise last_exception if last_exception else RuntimeError("Unknown error in fetch_with_retry")

# Function to chat with GPT
def chat_with_gpt(extracted_text: str, image_name: str = "", prompt: Optional[str] = None):
    structured_prompt = """
Bạn hãy phân tích và trình bày thông tin về một sản phẩm dưỡng da dựa trên danh sách thành phần mà tôi cung cấp. Kết quả cần chia thành 4 phần rõ ràng như sau:

# 1. THÀNH PHẦN CHÍNH 
Liệt kê tất cả từng thành phần có trong danh sách, mỗi dòng gồm:  
[Tên thành phần] – [Giải thích ngắn gọn công dụng chính, đơn giản, dễ hiểu].  
Giải thích nên ưu tiên tác dụng phổ biến và đơn giản hóa từ chuyên môn phức tạp nếu có.

# 2. CÔNG DỤNG CHÍNH
Tóm tắt 4–6 công dụng nổi bật của sản phẩm dưới dạng bullet point biểu tượng như sau:  

🌟 Làm sáng da  
💧 Dưỡng ẩm  
🧴 Chống lão hóa  
🛡️ Tăng cường hàng rào bảo vệ da  
🧘‍♀️ Làm dịu và mềm da  

(Chọn và diễn giải tùy thuộc vào thành phần thật sự có)

# 3. PHÙ HỢP VỚI LOẠI DA NÀO  
Gồm các loại da phổ biến: Da khô, Da thường, Da hỗn hợp, Da dầu, Da nhạy cảm.  
Với mỗi loại, đánh dấu bằng ✅ (phù hợp), ⚠️ (cần cân nhắc), hoặc ❌ (không phù hợp).  
Giải thích ngắn gọn lý do phù hợp hay không, dựa vào thành phần sản phẩm.

# 4. CẢNH BÁO & LƯU Ý  
Nếu trong thành phần có các hoạt chất dễ gây kích ứng như Retinol, AHA, BHA,... hãy thêm cảnh báo.  
Gợi ý một số cảnh báo phổ biến:  

⚠️ Có thể gây kích ứng nếu da nhạy cảm hoặc lần đầu dùng  
☀️ Cần dùng kem chống nắng ban ngày  
❌ Không kết hợp với các sản phẩm AHA/BHA nồng độ cao  
🧪 Nên test vùng da nhỏ trước khi dùng toàn mặt

Hãy trình bày rõ ràng, gọn gàng, đúng format, dễ đọc. Chỉ đưa ra thông tin theo đúng 4 phần trên. Đừng thêm lời khuyên, giới thiệu hoặc tổng kết thừa. Sử dụng định dạng Markdown đúng cách với tiêu đề (#) và khoảng cách hợp lý giữa các phần.
"""
    
    # user_prompt = prompt if prompt else structured_prompt
    
    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful skincare analysis assistant."},
            {"role": "user", "content": extracted_text + "\n\n" + structured_prompt}
        ],
        temperature=0.7,
    )
    
    response_text = response.choices[0].message.content
    
    # Trả về văn bản định dạng Markdown trực tiếp, không cần phân tích JSON
    return response_text

@app.post("/analyze-with-gpt/")
async def analyze_with_gpt(request: GPTRequest):
    """
    Analyze extracted text using GPT model
    """
    try:
        # Sử dụng prompt đã cập nhật trong hàm chat_with_gpt
        analysis = chat_with_gpt(request.extracted_text, prompt=request.prompt)
        
        # Trả về phản hồi định dạng Markdown trực tiếp
        return {"response": analysis}
    except Exception as e:
        logger.error(f"Error calling GPT API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")

# API endpoints
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image for processing.
    Returns an image_id that can be used to request OCR processing.
    """
    try:
        # Đọc nội dung hình ảnh
        contents = await file.read()
        
        # Tạo ID duy nhất cho hình ảnh
        image_id = str(uuid.uuid4())
        
        # Lưu hình ảnh vào Redis (thời hạn 30 phút)
        redis_client.set(f"image:{image_id}", contents, ex=1800)
        
        # Lưu thêm tên file gốc vào Redis
        redis_client.set(f"image_filename:{image_id}", file.filename, ex=1800)
        
        return {"image_id": image_id, "filename": file.filename}
    
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading image: {str(e)}")

@app.post("/process/", response_model=OCRResponse)
async def process_image(background_tasks: BackgroundTasks, request: OCRRequest):
    """
    Process an uploaded image with the specified OCR model.
    """
    # Kiểm tra model
    if request.model.lower() not in ["doctr", "trocr", "mmocr"]:
        raise HTTPException(status_code=400, detail="Invalid model selection. Choose 'doctr', 'trocr', or 'mmocr'")
    
    # Kiểm tra hình ảnh có tồn tại không
    if not redis_client.exists(f"image:{request.image_id}"):
        raise HTTPException(status_code=404, detail="Image not found. Please upload the image first.")
    
    # Tạo job ID
    job_id = str(uuid.uuid4())
    
    # Lưu trạng thái job vào Redis
    redis_client.hset(
        f"job:{job_id}",
        mapping={
            "status": "pending",
            "image_id": request.image_id,
            "model": request.model,
            "created_at": time.time()
        }
    )
    redis_client.expire(f"job:{job_id}", 3600)  # Hết hạn sau 1 giờ
    
    # Khởi động xử lý background
    background_tasks.add_task(process_image_task, job_id, request.image_id, request.model)
    
    return {"job_id": job_id, "status": "pending"}

@app.get("/status/{job_id}", response_model=OCRResponse)
async def get_job_status(job_id: str):
    """
    Check the status of an OCR processing job.
    """
    # Kiểm tra job có tồn tại không
    if not redis_client.exists(f"job:{job_id}"):
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Lấy trạng thái job
    job_data = redis_client.hgetall(f"job:{job_id}")
    
    # Chuyển từ byte sang string
    job_data = {k.decode('utf-8'): v.decode('utf-8') for k, v in job_data.items()}
    
    response = {"job_id": job_id, "status": job_data["status"]}
    
    if job_data["status"] == "completed":
        # Lấy kết quả từ Redis
        result_json = redis_client.get(f"result:{job_id}")
        if result_json:
            response["result"] = json.loads(result_json)
    
    elif job_data["status"] == "failed":
        response["error"] = job_data.get("error", "Unknown error")
    
    return response
    
@app.get("/models")
async def list_models():
    """
    List available OCR models
    """
    return {
        "models": [
            {"id": "doctr", "name": "DocTR", "description": "Document Text Recognition model"},
            {"id": "trocr", "name": "TrOCR", "description": "Transformer-based OCR model"},
            {"id": "mmocr", "name": "MMOCR", "description": "OpenMMLab Text Detection and Recognition"}
        ]
    }

# Background task to process the image
# Background task to process the image
async def process_image_task(job_id: str, image_id: str, model: str):
    """
    Background task to process an image through the OCR pipeline.
    """
    try:
        # Cập nhật trạng thái job
        redis_client.hset(f"job:{job_id}", "status", "processing")
        
        # Lấy dữ liệu hình ảnh từ Redis
        image_data = redis_client.get(f"image:{image_id}")
        if not image_data:
            raise Exception("Image data not found in Redis")
        
        # Lấy tên file gốc từ Redis
        original_filename = redis_client.get(f"image_filename:{image_id}")
        if original_filename:
            original_filename = original_filename.decode('utf-8')
            # Lưu tên file gốc vào job data
            redis_client.hset(f"job:{job_id}", "original_filename", original_filename)
        
        # Step 1: Gửi hình ảnh đến preprocessing service
        logger.info(f"Sending image {image_id} to preprocessing service")
        
        # Create client with increased timeout
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            # Step 1: Preprocessing
            preprocessed_image_id = None
            preprocessing_attempts = 0
            max_preprocessing_attempts = 3
            
            while preprocessing_attempts < max_preprocessing_attempts:
                try:
                    preprocessing_attempts += 1
                    logger.info(f"Preprocessing attempt {preprocessing_attempts}/{max_preprocessing_attempts}")
                    
                    preprocessing_response = await fetch_with_retry(
                        client, 
                        "post",
                        f"{PREPROCESSING_URL}/preprocess",
                        files={"file": ("image.jpg", image_data)},
                        timeout=REQUEST_TIMEOUT
                    )
                    
                    if preprocessing_response.status_code != 200:
                        error_msg = f"Preprocessing failed: {preprocessing_response.text}"
                        logger.error(error_msg)
                        if preprocessing_attempts >= max_preprocessing_attempts:
                            raise Exception(error_msg)
                        await asyncio.sleep(5)  # Wait before retry
                        continue
                    
                    # Lấy ID ảnh đã tiền xử lý
                    preprocessed_image_id = preprocessing_response.json()["preprocessed_image_id"]
                    
                    # Lưu ID ảnh đã tiền xử lý vào job
                    redis_client.hset(f"job:{job_id}", "preprocessed_image_id", preprocessed_image_id)
                    logger.info(f"Preprocessing successful, image ID: {preprocessed_image_id}")
                    break
                    
                except Exception as e:
                    logger.error(f"Preprocessing error: {str(e)}")
                    if preprocessing_attempts >= max_preprocessing_attempts:
                        raise Exception(f"Preprocessing failed after {max_preprocessing_attempts} attempts: {str(e)}")
                    await asyncio.sleep(5)  # Wait before retry
            
            if not preprocessed_image_id:
                raise Exception("Failed to preprocess image after multiple attempts")
            
            # Step 2: Send to appropriate OCR service
            ocr_url = {
                "doctr": f"{DOCTR_URL}/recognize",
                "trocr": f"{TROCR_URL}/recognize",
                "mmocr": f"{MMOCR_URL}/recognize"
            }[model.lower()]
            
            logger.info(f"Sending preprocessed image to {model} service")
            
            # Longer timeout for TrOCR
            request_timeout = REQUEST_TIMEOUT * 2 if model.lower() == "trocr" else REQUEST_TIMEOUT
            
            # OCR processing with retries
            ocr_attempts = 0
            max_ocr_attempts = 3
            
            while ocr_attempts < max_ocr_attempts:
                try:
                    ocr_attempts += 1
                    logger.info(f"OCR attempt {ocr_attempts}/{max_ocr_attempts} with {model} service")
                    
                    ocr_response = await fetch_with_retry(
                        client,
                        "post",
                        ocr_url,
                        json={"preprocessed_image_id": preprocessed_image_id},
                        timeout=request_timeout
                    )
                    
                    if ocr_response.status_code != 200:
                        error_msg = f"OCR processing failed: {ocr_response.text}"
                        logger.error(error_msg)
                        if ocr_attempts >= max_ocr_attempts:
                            raise Exception(error_msg)
                        await asyncio.sleep(5)  # Wait before retry
                        continue
                    
                    # Parse result
                    result = None
                    try:
                        # Xử lý trường hợp response.text có thể là thuộc tính hoặc phương thức
                        if hasattr(ocr_response, 'text'):
                            if callable(ocr_response.text):
                                # Nếu là phương thức
                                try:
                                    result_text = await ocr_response.text()
                                    result = json.loads(result_text)
                                except TypeError:
                                    # Nếu ocr_response.text không phải callable
                                    if isinstance(ocr_response.text, str):
                                        result = json.loads(ocr_response.text)
                                    else:
                                        # Trường hợp khác
                                        result = ocr_response.json()
                            else:
                                # Nếu là thuộc tính
                                if isinstance(ocr_response.text, str):
                                    result = json.loads(ocr_response.text)
                                else:
                                    # Trường hợp khác
                                    result = ocr_response.json()
                        else:
                            # Trả về trực tiếp
                            result = ocr_response.json()
                        
                        print(f"OCR response structure: {list(result.keys()) if isinstance(result, dict) else 'Not a dictionary'}")
                        
                    except Exception as e:
                        logger.error(f"Error parsing OCR response: {str(e)}")
                        traceback.print_exc()
                        if ocr_attempts >= max_ocr_attempts:
                            raise Exception(f"Error parsing OCR response: {str(e)}")
                        await asyncio.sleep(5)  # Wait before retry
                        continue
                    
                    if not isinstance(result, dict) or "result" not in result:
                        error_msg = f"Invalid OCR response format: {result}"
                        logger.error(error_msg)
                        if ocr_attempts >= max_ocr_attempts:
                            raise Exception(error_msg)
                        await asyncio.sleep(5)  # Wait before retry
                        continue
                    
                    # Log details about the result
                    print(f"Result contains {len(result['result'])} text regions")
                    if result['result'] and len(result['result']) > 0:
                        print(f"First result item: {result['result'][0]}")
                    
                    # Save result to Redis with longer expiration (2 hours)
                    redis_client.set(f"result:{job_id}", json.dumps(result["result"]), ex=7200)
                    print(f"Saved results to Redis with key: result:{job_id}")
                    
                    # Update job status to completed
                    redis_client.hset(f"job:{job_id}", "status", "completed")
                    logger.info(f"Job {job_id} completed successfully with {model} OCR service")
                    return
                    
                except Exception as e:
                    logger.error(f"OCR error with {model} service: {str(e)}")
                    if ocr_attempts >= max_ocr_attempts:
                        raise Exception(f"OCR processing failed after {max_ocr_attempts} attempts: {str(e)}")
                    await asyncio.sleep(5)  # Wait before retry
            
            # If we got here without returning, all attempts failed
            raise Exception(f"All OCR attempts with {model} service failed")
    
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        logger.error(traceback.format_exc())
        # Store detailed error information in Redis
        redis_client.hset(
            f"job:{job_id}",
            mapping={
                "status": "failed",
                "error": str(e),
                "error_time": time.time()
            }
        )
        # Set expiration for failed jobs (1 hour)
        redis_client.expire(f"job:{job_id}", 3600)

@app.get("/annotated-image/{job_id}")
async def get_annotated_image(job_id: str):
    """
    Return the preprocessed image with bounding boxes drawn on it
    """
    try:
        logger.info(f"Annotation request received for job_id: {job_id}")
        
        # Kiểm tra job có tồn tại không
        if not redis_client.exists(f"job:{job_id}"):
            logger.error(f"Job ID not found: {job_id}")
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Lấy thông tin job
        job_data = redis_client.hgetall(f"job:{job_id}")
        job_data = {k.decode('utf-8'): v.decode('utf-8') for k, v in job_data.items()}
        logger.info(f"Job data: {job_data}")
        
        if job_data["status"] != "completed":
            raise HTTPException(status_code=400, detail="Job not completed yet")
        
        # Lấy model được sử dụng để biết kích thước resize
        model = job_data.get("model", "doctr").lower()
        
        # Lấy ID ảnh đã tiền xử lý từ job
        preprocessed_image_id = job_data.get("preprocessed_image_id")
        if not preprocessed_image_id:
            # Log the error and provide a detailed message
            logger.error(f"Preprocessed image ID not found for job {job_id}")
            raise HTTPException(status_code=404, detail="Preprocessed image ID not found in job data. The preprocessing step may have failed.")
        
        # Lấy ảnh đã tiền xử lý từ Redis
        preprocessed_image_bytes = redis_client.get(f"preprocessed:{preprocessed_image_id}")
        if not preprocessed_image_bytes:
            logger.error(f"Preprocessed image not found in Redis for ID: {preprocessed_image_id}")
            # Attempt to recover with alternative methods (if available)
            original_image_id = job_data.get("image_id")
            if original_image_id:
                logger.info(f"Attempting to fall back to original image {original_image_id}")
                original_image_bytes = redis_client.get(f"image:{original_image_id}")
                if original_image_bytes:
                    preprocessed_image_bytes = original_image_bytes
                    logger.info("Successfully retrieved original image as fallback")
                else:
                    logger.error(f"Original image also not found for ID: {original_image_id}")
                    raise HTTPException(status_code=404, detail="Preprocessed image not found in storage. The data may have expired.")
            else:
                raise HTTPException(status_code=404, detail="Preprocessed image not found in storage. The data may have expired.")
        
        # Chuyển đổi sang numpy array
        try:
            nparr = np.frombuffer(preprocessed_image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Failed to decode image from bytes")
                raise HTTPException(status_code=500, detail="Failed to decode image data")
                
            # Log image dimensions for debugging
            logger.info(f"Decoded image dimensions: {img.shape}")
        except Exception as e:
            logger.error(f"Error decoding image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error decoding image: {str(e)}")
        
        # Kích thước resize dựa trên model
        if model == "mmocr":
            resize_width, resize_height = 640, 640
        else:  # DocTR hoặc TrOCR đều dùng 1024
            resize_width, resize_height = 1024, 1024
        
        # Trích xuất kích thước thực tế của ảnh để biết tỷ lệ
        actual_height, actual_width = img.shape[:2]
        
        # Lấy kết quả OCR để vẽ bounding box
        result_json = redis_client.get(f"result:{job_id}")
        if not result_json:
            logger.error(f"OCR result not found for job_id: {job_id}")
            raise HTTPException(status_code=404, detail="OCR result not found. The result may have expired or the OCR process failed.")
        
        try:
            result = json.loads(result_json)
            logger.info(f"Drawing {len(result)} bounding boxes")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing OCR result JSON: {str(e)}")
            raise HTTPException(status_code=500, detail="Invalid OCR result format")
        
        # Create a copy of the image for drawing
        draw_img = img.copy()
        
        # Vẽ bounding box lên ảnh
        drawn_boxes = 0
        for item in result:
            if "bbox" in item and item["bbox"]:  # Kiểm tra bbox không rỗng
                bbox = item["bbox"]
                text = item.get("text", "")
                
                # Đảm bảo định dạng bbox hợp lệ
                if bbox and isinstance(bbox, list) and len(bbox) >= 4:
                    try:
                        # Ghi log để debug
                        logger.info(f"Drawing bbox: {bbox}")
                        
                        # Chuyển đổi thành numpy array
                        bbox_np = np.array(bbox, np.int32)
                        
                        # Đảm bảo bbox_np có đủ điểm và đúng định dạng
                        if bbox_np.shape[0] >= 4:
                            # FIX: Đảm bảo reshape đúng định dạng mà OpenCV yêu cầu
                            bbox_np = bbox_np.reshape((-1, 1, 2))
                            
                            # Vẽ đường viền (đỏ)
                            cv2.polylines(draw_img, [bbox_np], True, (0, 0, 255), 2)
                            drawn_boxes += 1
                            
                            # Vẽ text
                            if len(bbox) > 0:
                                # Lấy tọa độ góc trên bên trái để vẽ text
                                x, y = bbox[0]
                                # Đặt text ở vị trí phù hợp, tránh ra ngoài ảnh
                                x = max(0, x)
                                y = max(20, y)  # Đảm bảo đủ không gian cho text
                                # Use a better font scale based on image size
                                font_scale = max(0.3, min(1.0, actual_width / 1000))
                                cv2.putText(draw_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
                        else:
                            logger.warning(f"Skipping bbox with insufficient points: {bbox}")
                    except Exception as e:
                        logger.error(f"Error drawing bbox: {e}")
                        traceback.print_exc()  # Thêm dòng này để ghi chi tiết lỗi
        
        logger.info(f"Successfully drew {drawn_boxes} bounding boxes out of {len(result)} results")
        
        try:
            # Chuyển đổi ảnh thành byte stream
            success, processed_bytes = cv2.imencode('.png', draw_img)
            
            # Kiểm tra kết quả encoding
            if not success:
                logger.error("Failed to encode annotated image")
                raise HTTPException(status_code=500, detail="Error encoding image")
                
            buffer = processed_bytes.tobytes()
            
            # Trả về ảnh
            return StreamingResponse(io.BytesIO(buffer), media_type="image/png")
        except Exception as e:
            logger.error(f"Error creating image response: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error creating image response: {str(e)}")
    
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        logger.error(f"Error generating annotated image: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/export-csv/{job_id}")
async def export_csv(job_id: str):
    """Export OCR and analysis results to CSV"""
    try:
        # Get OCR result
        ocr_result = redis_client.get(f"result:{job_id}")
        if not ocr_result:
            raise HTTPException(status_code=404, detail="Result not found")
            
        # Get image name
        job_info = redis_client.hgetall(f"job:{job_id}")
        image_id = job_info.get(b"image_id").decode("utf-8")

        # Lấy tên file gốc từ job data
        original_filename = job_info.get(b"original_filename")
        if original_filename:
            original_filename = original_filename.decode("utf-8")
        else:
            # Nếu không có tên file gốc, sử dụng image_id
            original_filename = image_id
        
        # Extract text
        result_data = json.loads(ocr_result)
        extracted_text = "\n".join([item["text"] for item in result_data])
        
        # Get GPT analysis
        gpt_response = chat_with_gpt(extracted_text, image_id)
        
        # Phân tích nội dung từ định dạng Markdown để xuất CSV
        # Chuẩn bị dữ liệu cho CSV
        ingredients_list = ""
        benefits_list = ""
        skin_types_list = ""
        warnings_list = ""
        
        # Chuyển đổi văn bản phản hồi thành các phần riêng biệt
        import re
        
        # Tách phản hồi thành từng dòng
        lines = gpt_response.split('\n')
        current_section = None
        
        # Xác định các mẫu tiêu đề cho từng phần
        sections_patterns = {
            "ingredients": ["THÀNH PHẦN CHÍNH", "INGREDIENTS"],
            "benefits": ["CÔNG DỤNG CHÍNH", "BENEFITS"],
            "skin_types": ["PHÙ HỢP VỚI LOẠI DA NÀO", "SKIN_TYPES"],
            "warnings": ["CẢNH BÁO & LƯU Ý", "WARNINGS", "CẢNH BÁO VÀ LƯU Ý"]
        }
        
        section_content = {
            "ingredients": [],
            "benefits": [],
            "skin_types": [],
            "warnings": []
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Kiểm tra xem dòng hiện tại có phải là tiêu đề không
            is_header = False
            for section, patterns in sections_patterns.items():
                if any(pattern.lower() in line.lower() for pattern in patterns):
                    current_section = section
                    is_header = True
                    break
            
            # Nếu không phải tiêu đề và thuộc một phần đã biết
            if not is_header and current_section:
                # Loại bỏ số thứ tự nếu có (như "1.", "2.")
                clean_line = re.sub(r'^\d+\.\s*', '', line)
                section_content[current_section].append(clean_line)
        
        # Chuyển các mảng thành chuỗi
        ingredients_list = "\n".join(section_content["ingredients"])
        benefits_list = "\n".join(section_content["benefits"])
        skin_types_list = "\n".join(section_content["skin_types"])
        warnings_list = "\n".join(section_content["warnings"])
        
        # Làm sạch định dạng Markdown
        # Làm sạch định dạng Markdown và emoji
        def clean_markdown_and_emoji(text):
            # Xóa dấu gạch đầu dòng và khoảng trắng
            lines = []
            for line in text.split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    line = line[2:]
                elif line.startswith('-'):
                    line = line[1:]
                elif line.startswith('* '):
                    line = line[2:]
                elif line.startswith('*'):
                    line = line[1:]
                
                # Xóa các định dạng Markdown khác
                line = re.sub(r'(\*\*|__)(.*?)(\*\*|__)', r'\2', line)  # Bold
                line = re.sub(r'(\*|_)(.*?)(\*|_)', r'\2', line)  # Italic
                line = re.sub(r'~~(.*?)~~', r'\1', line)  # Strikethrough
                line = re.sub(r'`(.*?)`', r'\1', line)  # Inline code
                line = re.sub(r'^#+\s*', '', line)  # Headers
                
                # Xóa emoji thông dụng
                emoji_pattern = re.compile(
                    "["
                    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                    "\U0001F300-\U0001F5FF"  # symbols & pictographs
                    "\U0001F600-\U0001F64F"  # emoticons
                    "\U0001F680-\U0001F6FF"  # transport & map symbols
                    "\U0001F700-\U0001F77F"  # alchemical symbols
                    "\U0001F780-\U0001F7FF"  # Geometric Shapes
                    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                    "\U0001FA00-\U0001FA6F"  # Chess Symbols
                    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                    "\U00002702-\U000027B0"  # Dingbats
                    "\U000024C2-\U0001F251" 
                    "]+", flags=re.UNICODE)
                
                line = emoji_pattern.sub(r'', line)
                
                # Xóa các biểu tượng đánh dấu cụ thể
                line = line.replace('✅', '')  # ✅
                line = line.replace('❌', '')  # ❌
                line = line.replace('⚠️', '')  # ⚠️
                line = line.replace('☀️', '')  # ☀️
                line = line.replace('☑', '')  # ☑
                line = line.replace('✔', '')  # ✔
                
                # Loại bỏ khoảng trắng thừa
                line = re.sub(r'\s+', ' ', line).strip()
                
                if line.strip():
                    lines.append(line)
            return "\n".join(lines)
        
        # Áp dụng làm sạch định dạng và emoji
        ingredients_list = clean_markdown_and_emoji(ingredients_list)
        benefits_list = clean_markdown_and_emoji(benefits_list)
        skin_types_list = clean_markdown_and_emoji(skin_types_list)
        warnings_list = clean_markdown_and_emoji(warnings_list)
        
        # Đảm bảo không có cột nào trống
        if not ingredients_list.strip():
            ingredients_list = "Không có thông tin về thành phần"
        if not benefits_list.strip():
            benefits_list = "Không có thông tin về công dụng"
        if not skin_types_list.strip():
            skin_types_list = "Không có thông tin về loại da phù hợp"
        if not warnings_list.strip():
            warnings_list = "Không có cảnh báo"
        
        json_data = {
            "Tên ảnh": original_filename,
            "Thành phần": ingredients_list,
            "Công dụng": benefits_list,
            "Da phù hợp": skin_types_list,
            "Lưu ý": warnings_list
        }

        # Ghi log dữ liệu JSON
        with open("json_data.json", "a") as log_file:
            log_file.write(json.dumps(json_data, ensure_ascii=False) + "\n")

        # Tạo CSV in-memory
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Tên ảnh", "Thành phần", "Công dụng", "Da phù hợp", "Lưu ý"])
        writer.writerow([
            original_filename,
            ingredients_list,
            benefits_list,
            skin_types_list,
            warnings_list
        ])
        
        # Lấy nội dung chuỗi và chuyển thành bytes với encoding UTF-8
        csv_content = output.getvalue().encode('utf-8-sig')  # utf-8-sig bao gồm BOM
        
        # Tạo BytesIO từ bytes
        bytes_io = io.BytesIO(csv_content)
        
        return StreamingResponse(
            bytes_io,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename=analysis_{job_id}.csv"}
        )
        
    except Exception as e:
        logger.error(f"Error exporting CSV: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error exporting CSV: {str(e)}")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
