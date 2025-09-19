FROM python:3.10-slim

WORKDIR /app

# Add this line to install the missing system library for LightGBM
RUN apt-get update && apt-get install -y libgomp1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8888
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]