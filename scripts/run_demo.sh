#!/bin/bash
# Run RaeLM Demo - Complete system startup script

set -e

echo "🚀 Starting RaeLM Document Understanding Platform"
echo "================================================="

# Check if infrastructure is running
if ! docker ps | grep -q postgres; then
    echo "Starting infrastructure..."
    make start-infra
    sleep 5
fi

# Start API server
echo "Starting FastAPI server..."
cd /Users/nainy/Documents/tennr-realm
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

sleep 3

# Start Celery worker
echo "Starting Celery worker..."
celery -A inference.tasks worker --loglevel=info &
CELERY_PID=$!

sleep 2

# Start Streamlit demo
echo "Starting Streamlit demo..."
streamlit run ui/demo_app.py --server.port 8501 &
STREAMLIT_PID=$!

sleep 2

echo ""
echo "✅ RaeLM is running!"
echo "===================="
echo "📄 Demo UI: http://localhost:8501"
echo "🔌 API: http://localhost:8000"
echo "📊 API Docs: http://localhost:8000/docs"
echo "📈 Metrics: http://localhost:8000/metrics"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "echo 'Stopping services...'; kill $API_PID $CELERY_PID $STREAMLIT_PID 2>/dev/null" EXIT INT TERM

wait

