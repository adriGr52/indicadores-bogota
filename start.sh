#!/bin/bash

echo "🚀 Starting Fecundidad Temprana API..."
echo "Environment: $(echo $RAILWAY_ENVIRONMENT || echo 'local')"
echo "Port: $PORT"
echo "Database URL: ${DATABASE_URL:0:50}..."

# Verificar que el puerto esté configurado
if [ -z "$PORT" ]; then
    echo "⚠️ Warning: PORT not set, using 8000"
    export PORT=8000
fi

# Verificar Python version
echo "🐍 Python version: $(python --version)"

# Verificar dependencias críticas
echo "📦 Checking critical dependencies..."
python -c "import fastapi, uvicorn, sqlalchemy, pandas; print('✅ Core dependencies OK')" || {
    echo "❌ Missing critical dependencies"
    exit 1
}

# Verificar estructura de archivos
echo "📁 Checking file structure..."
if [ -f "main.py" ]; then
    echo "✅ main.py found"
else
    echo "❌ main.py not found"
    exit 1
fi

if [ -f "dashboard_compatible.html" ]; then
    echo "✅ dashboard_compatible.html found"
else
    echo "⚠️ dashboard_compatible.html not found - will use fallback"
fi

# Crear directorio de logs si no existe
mkdir -p logs

# Set environment variables
export PYTHONPATH="/app:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Verificar que la aplicación se puede importar
echo "🧪 Testing app import..."
python -c "from main import app; print('✅ App imported successfully')" || {
    echo "❌ Failed to import app"
    exit 1
}

# Ejecutar tests básicos si estamos en desarrollo
if [ "$RAILWAY_ENVIRONMENT" != "production" ]; then
    echo "🧪 Running basic tests..."
    python -c "
import sys
try:
    from main import SessionLocal
    db = SessionLocal()
    result = db.execute('SELECT 1').scalar()
    db.close()
    print('✅ Database connection test passed')
    sys.exit(0)
except Exception as e:
    print(f'⚠️ Database test failed: {e}')
    print('📝 Will continue anyway (might use SQLite fallback)')
    sys.exit(0)
    "
fi

echo "🎯 Starting FastAPI server..."
echo "📍 Server will be available at: http://0.0.0.0:$PORT"

# Iniciar la aplicación con configuración optimizada
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --log-level info \
    --access-log \
    --use-colors \
    --loop uvloop
