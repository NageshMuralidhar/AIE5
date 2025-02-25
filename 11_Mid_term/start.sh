#!/bin/bash

# Start nginx in background
nginx

# Start FastAPI application
uvicorn main:app --host 0.0.0.0 --port 7860 