import os
import io
import uuid
import redis
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove
from craft_text_detector import Craft

# Initialize FastAPI app
app = FastAPI(title="Image Preprocessing Service")

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

# Cấu hình
CONFIG = {
    "padding_size": 50, 
    "text_threshold": 0.8, 
    "link_threshold": 0.4, 
    "low_text": 0.3, 
}

# Initialize CRAFT text detector
try:
    craft_detector = Craft(
        crop_type="poly", 
        cuda=True, 
        text_threshold=CONFIG["text_threshold"], 
        link_threshold=CONFIG["link_threshold"], 
        low_text=CONFIG["low_text"],
        weight_path_craft_net="/app/craft_text_detector/weights/craft_mlt_25k.pth",
        weight_path_refine_net="/app/craft_text_detector/weights/craft_refiner_CTW1500.pth",
    )
    print("CRAFT detector initialized successfully")
except Exception as e:
    print(f"Error initializing CRAFT detector: {e}")
    craft_detector = None


class ImageProcessor:
    """Image processing class for text extraction and optimization."""
    
    @staticmethod
    def ensure_rgb(image: np.ndarray) -> np.ndarray:
        """Ensure image is in RGB format."""
        if len(image.shape) == 2:  # Grayscale
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
    
    @staticmethod
    def remove_background(image: np.ndarray) -> np.ndarray:
        """Remove background from an image."""
        return remove(image)
    
    @staticmethod
    def crop_around_content(image: np.ndarray) -> np.ndarray:
        """Crop image around the main content."""
        gray = image[:,:,0] if image.ndim > 2 else image
        
        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        x, y, w, h = cv2.boundingRect(thresholded)
        
        if image.ndim > 2:
            return image[y:y+h, x:x+w]
        else:
            output = gray[y:y+h, x:x+w]
            return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def add_padding(image: np.ndarray, padding_size: int = None) -> np.ndarray:
        """Add padding around the image."""
        padding = padding_size if padding_size is not None else CONFIG["padding_size"]
        
        # Create appropriately shaped empty array
        if len(image.shape) == 2:  # Grayscale
            padded = np.zeros((
                image.shape[0] + 2 * padding, 
                image.shape[1] + 2 * padding
            ), dtype=np.uint8)
        else:  # Color
            padded = np.zeros((
                image.shape[0] + 2 * padding, 
                image.shape[1] + 2 * padding, 
                image.shape[2]
            ), dtype=np.uint8)
        
        # Place original image in the center of the padded one
        padded[
            padding:padding + image.shape[0], 
            padding:padding + image.shape[1]
        ] = image
        
        return padded
    
    @staticmethod
    def detect_text_regions(image: np.ndarray) -> list:
        """Detect text regions in the image using CRAFT."""
        if craft_detector is None:
            print("CRAFT detector not available, skipping text detection")
            return []
            
        # Detect text with CRAFT
        try:
            detection_result = craft_detector.detect_text(image)
            text_boxes = detection_result['boxes']
        except Exception as e:
            print(f"Error detecting text: {e}")
            return []
            
        regions = []
        for box in text_boxes:
            if len(box) >= 3:  # Need at least 3 points for a polygon
                # Convert to contour format
                points = np.array(box, dtype=np.int32).reshape(-1, 2)
                
                # Get minimum area rectangle
                rect = cv2.minAreaRect(points)
                box_vertices = cv2.boxPoints(rect)
                box_vertices = np.array(box_vertices, dtype=np.int32)
                
                regions.append(box_vertices)
        
        return regions
    
    @staticmethod
    def rotate_to_horizontal(image: np.ndarray, regions: list) -> np.ndarray:
        """Rotate image to make text horizontal based on detected regions."""
        if not regions:
            return image
            
        # Find largest region by area
        largest_region = max(regions, key=cv2.contourArea)
        
        # Calculate orientation
        [vx, vy, x, y] = cv2.fitLine(largest_region, cv2.DIST_L2, 0, 0.01, 0.01)
        angle_rad = np.arctan2(vy, vx)
        angle_deg = np.degrees(angle_rad)[0]
        
        # Skip if angle is already close to horizontal
        if abs(angle_deg - 90) < 1:
            return image
        
        print(f"Rotating image by {angle_deg:.2f} degrees")
            
        # Rotate image
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, scale=1.0)
        
        return cv2.warpAffine(image, rotation_matrix, (width, height))
    
    @classmethod
    def process_image(cls, image: np.ndarray) -> np.ndarray:
        """Process an image through the full pipeline."""
        # Ensure image is RGB
        image = cls.ensure_rgb(image)
        
        print("Removing background...")
        # Remove background
        image_no_bg = cls.remove_background(image)
        
        print("Adding padding...")
        # Add padding
        padded_image = cls.add_padding(image_no_bg)
        
        print("Detecting text regions...")
        # Detect text regions
        text_regions = cls.detect_text_regions(padded_image)
        
        # Rotate if needed
        if text_regions:
            print(f"Found {len(text_regions)} text regions, rotating if needed...")
            rotated_image = cls.rotate_to_horizontal(padded_image, text_regions)
        else:
            print("No text regions found, skipping rotation")
            rotated_image = padded_image
        
        print("Cropping around content...")
        # Crop around content
        final_image = cls.crop_around_content(rotated_image)
            
        return final_image


@app.get("/")
def read_root():
    return {"message": "Image Preprocessing Service is running"}

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "preprocessing"}


@app.post("/preprocess")
async def preprocess_image(file: UploadFile = File(...)):
    """
    Preprocess an image for OCR.
    Returns a preprocessed_image_id that can be used by OCR services.
    """
    try:
        print(f"Received file: {file.filename}")
        # Read image content
        contents = await file.read()
        
        # Convert to numpy array for processing
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        print(f"Image shape: {image.shape}")
        
        # Process the image
        processed_image = ImageProcessor.process_image(image)
        
        # Convert back to bytes
        _, processed_bytes = cv2.imencode('.png', processed_image)
        processed_bytes = processed_bytes.tobytes()
        
        # Generate a unique ID for the processed image
        preprocessed_image_id = str(uuid.uuid4())
        
        # Store in Redis (with 30 minute expiration)
        redis_client.set(f"preprocessed:{preprocessed_image_id}", processed_bytes, ex=1800)
        
        print(f"Preprocessing complete, ID: {preprocessed_image_id}")
        return {"preprocessed_image_id": preprocessed_image_id}
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error preprocessing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)