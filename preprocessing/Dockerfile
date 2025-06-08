FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # - Common packages
    build-essential \
    pkg-config \
    curl \
    wget \
    software-properties-common \
    unzip \
    git \
    # - Python compilation packages
    tar make gcc zlib1g-dev libffi-dev libssl-dev liblzma-dev libbz2-dev libsqlite3-dev \
    # - DocTR packages
    libgl1-mesa-dev libsm6 libxext6 libxrender-dev libpangocairo-1.0-0 \
    python3-opencv \
    # - matplotlib packages
    python3-tk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.11
ARG PYTHON_VERSION=3.11.11
RUN wget http://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar -zxf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    mkdir -p /opt/python/ && \
    ./configure --prefix=/opt/python --enable-optimizations && \
    make && \
    make install && \
    cd .. && \
    rm Python-$PYTHON_VERSION.tgz && \
    rm -r Python-$PYTHON_VERSION

ENV PATH=/opt/python/bin:$PATH

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    /opt/python/bin/python3 get-pip.py && \
    rm get-pip.py

# Verify Python and pip versions
RUN /opt/python/bin/python3 --version && \
    /opt/python/bin/pip3 --version

# Upgrade pip
RUN /opt/python/bin/pip3 install --upgrade pip

# Install PyTorch with CUDA support
RUN /opt/python/bin/pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set CUDA environment variables
ENV PYTHONPATH="/app"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV NUMBA_THREADING_LAYER=omp

# Create working directories
WORKDIR /app
RUN mkdir -p /app/craft_text_detector /root/.craft_text_detector/weights /root/.u2net

# Copy model weights and configurations
COPY craft_text_detector /app/craft_text_detector/
COPY models/craft_mlt_25k.pth /root/.craft_text_detector/weights/craft_mlt_25k.pth
COPY models/craft_refiner_CTW1500.pth /root/.craft_text_detector/weights/craft_refiner_CTW1500.pth
COPY models/u2net.onnx /root/.u2net/u2net.onnx

# Copy application code and requirements
COPY requirements.txt /app/
COPY app.py /app/

# Install application dependencies
RUN /opt/python/bin/pip3 install -r requirements.txt
RUN pip install --upgrade tbb

# Expose port
EXPOSE 5000

# Start the FastAPI application
CMD ["/opt/python/bin/python3", "app.py"]