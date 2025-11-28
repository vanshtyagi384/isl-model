FROM python:3.10-slim

# System packages needed for numpy / pillow / cv2
RUN apt-get clean && apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY re.txt .  
RUN pip install --no-cache-dir -r re.txt

COPY . .

EXPOSE 10000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
