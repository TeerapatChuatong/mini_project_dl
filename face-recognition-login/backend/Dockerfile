FROM continuumio/miniconda3

WORKDIR /app

COPY . /app

RUN conda install -y -c conda-forge python=3.11 dlib face_recognition cmake && \
    pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart pillow opencv-python numpy ultralytics && \
    conda clean -afy

EXPOSE 10000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]