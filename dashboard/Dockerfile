FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Copy data files from parent if they exist
COPY ../_h_batch_production_data.xlsx ./ 2>/dev/null || true
COPY ../_h_batch_process_data.xlsx ./ 2>/dev/null || true

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
