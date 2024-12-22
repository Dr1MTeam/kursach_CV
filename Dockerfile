FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get clean && \
rm -rf /var/lib/apt/lists/*

#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]