#!/bin/bash
# Health check script

echo "🩺 Checking application health..."

response=$(curl -s -w "%{http_code}" http://localhost:8000/api/health -o /tmp/health_response.json)

if [ "$response" = "200" ]; then
    echo "✅ Application is healthy"
    cat /tmp/health_response.json | python3 -m json.tool
else
    echo "❌ Application health check failed (HTTP $response)"
    if [ -f /tmp/health_response.json ]; then
        cat /tmp/health_response.json
    fi
fi
