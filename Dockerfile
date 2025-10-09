FROM python:3.10-slim

WORKDIR /app

# Add this line to install the missing system library for LightGBM
RUN apt-get update && apt-get install -y libgomp1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- ADD THESE LINES HERE ---
# Copy the trained model and the DuckDB database into the image
# This is done before "COPY . ." to better leverage layer caching.
COPY ./output /app/output
COPY ./models /app/models
# --------------------------

COPY . .

EXPOSE 8000 8888
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]