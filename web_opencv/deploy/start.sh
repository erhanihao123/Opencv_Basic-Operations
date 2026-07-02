#!/bin/bash

set -e

echo "======================================"
echo "  Image Annotation Tool - Deployment"
echo "======================================"

if [ ! -d "../logs" ]; then
    mkdir -p ../logs
    echo "Created logs directory"
fi

if [ ! -d "../backend/models" ]; then
    mkdir -p ../backend/models
    echo "Created models directory"
fi

echo ""
echo "Starting services with Docker Compose..."
echo ""

cd "$(dirname "$0")"
docker-compose up -d

echo ""
echo "Waiting for services to start..."
sleep 10

echo ""
echo "Checking service status..."
docker-compose ps

echo ""
echo "======================================"
echo "  Deployment Complete!"
echo "======================================"
echo ""
echo "Access the application at:"
echo "  http://<your-server-ip>"
echo ""
echo "To check logs:"
echo "  docker-compose logs -f"
echo ""
echo "To stop services:"
echo "  docker-compose down"