# Use a full image to ensure all build tools and libraries are present
FROM python:3.10-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the upload folder exists relative to the app
RUN mkdir -p ../backend/uploads

EXPOSE 5000

CMD ["python", "main.py"]
