from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

# Đường dẫn đến thư mục gốc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def mount_static_frontend(app: FastAPI):
    """
    Cấu hình để phục vụ frontend HTML
    """
    # Tạo thư mục static nếu chưa tồn tại
    static_dir = os.path.join(BASE_DIR, "static")
    templates_dir = os.path.join(BASE_DIR, "templates")
    
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(templates_dir, exist_ok=True)
    
    # Lưu HTML vào thư mục templates
    with open(os.path.join(templates_dir, "index.html"), "w", encoding="utf-8") as f:
        with open("frontend.html", "r", encoding="utf-8") as src:
            f.write(src.read())
    
    # Mount thư mục static
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # Tạo templates
    templates = Jinja2Templates(directory=templates_dir)
    
    # Route để hiển thị frontend
    @app.get("/", response_class=HTMLResponse)
    async def get_html(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})