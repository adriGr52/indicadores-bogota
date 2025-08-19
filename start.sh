#!/bin/bash
echo "Starting Fecundidad Temprana API..."
echo "Port: $PORT"
echo "Database URL: ${DATABASE_URL:0:50}..."

# Verificar que el puerto esté configurado
if [ -z "$PORT" ]; then
    echo "Warning: PORT not set, using 8000"
    export PORT=8000
fi

# Iniciar la aplicación
exec uvicorn main:app --host 0.0.0.0 --port $PORT --log-level info
