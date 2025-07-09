# 使用輕量級 Python 映像作為基底（支援 TensorFlow）
FROM python:3.11-slim

# 安裝必要套件
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 複製所有專案檔案到容器中
COPY . /app

# 安裝 Python 套件
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 自動下載模型（分割模型 segmentation.h5）
RUN python3 -c "import os, gdown; \
  seg_path = 'segmentation.h5'; \
  cls_path = 'analysis.h5'; \
  seg_id = '1d44Rt7ihKTdkdhn2grWh6nlEJ4k4OUKh'; \
  cls_id = '1G6q_AKZi7MyNJAX9b9pIOkvT8DRwpwMU'; \
  \
  print('📥 檢查 segmentation.h5'); \
  os.path.exists(seg_path) or gdown.download(f'https://drive.google.com/uc?id={seg_id}', seg_path, quiet=False); \
  print('📥 檢查 analysis.h5'); \
  os.path.exists(cls_path) or gdown.download(f'https://drive.google.com/uc?id={cls_id}', cls_path, quiet=False)"

# 開放埠口 Cloud Run 預設是 8080
EXPOSE 8080

# 啟動應用，指定 api.py 裡的 app 物件
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "api:app"]
