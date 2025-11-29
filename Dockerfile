FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git

COPY requirements.txt .
RUN pip install -r requirements.txt

# Install Django for validation (optional)
RUN pip install django

COPY . .

