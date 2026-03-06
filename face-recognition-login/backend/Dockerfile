FROM continuumio/miniconda3

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y -c conda-forge python=3.11 dlib cmake && \
    pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    pillow \
    opencv-python-headless \
    numpy \
    ultralytics \
    face_recognition \
    face-recognition-models && \
    conda clean -afy

EXPOSE 10000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]