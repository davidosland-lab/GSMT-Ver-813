#!/bin/bash
# Development server with auto-reload

echo "🔧 Starting development server..."
echo "📱 Frontend: http://localhost:8000/"
echo "📚 API Docs: http://localhost:8000/api/docs"
echo ""

python3 -c "
import uvicorn
from app import app

if __name__ == '__main__':
    uvicorn.run(
        'app:app',
        host='0.0.0.0',
        port=8000,
        reload=True,
        log_level='info'
    )
"
