# Build frontend
FROM node:18-alpine as frontend-build

WORKDIR /frontend
COPY podcraft/package*.json ./
RUN npm install
COPY podcraft/ .
RUN npm run build

# Build backend
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY server/ .

# Create necessary directories
RUN mkdir -p audio_storage transcripts

# Copy frontend build
COPY --from=frontend-build /frontend/dist /usr/share/nginx/html

# Copy nginx configuration
COPY podcraft/nginx.conf /etc/nginx/conf.d/default.conf

# Expose ports
EXPOSE 7860

# Copy startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Start both services
CMD ["/start.sh"] 