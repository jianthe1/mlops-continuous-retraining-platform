FROM python:3.10-slim

WORKDIR /app

# COPY requirements.txt from root context
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# COPY main.py and model.pkl from app/ directory into current WORKDIR (/app)
COPY app/main.py .
COPY app/model.pkl .  

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]