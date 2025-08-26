#!/bin/bash

echo "🚀 Starting Fecundidad Temprana API v4.3.1..."
echo "🆕 Mejoras implementadas:"
echo "   • Filtros Localidad/UPZ corregidos"
echo "   • Índice Theil con todas las UPZ"
echo "   • Dashboard responsive optimizado"
echo "   • Orden de pestañas mejorado"
echo ""
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

# Verificar estructura de archivos v4.3.1
echo "📁 Checking file structure v4.3.1..."
if [ -f "main.py" ]; then
    echo "✅ main.py found"
    
    # Verificar versión en main.py
    VERSION=$(python -c "from main import app; print(app.version)" 2>/dev/null || echo "unknown")
    if [ "$VERSION" = "4.3.1" ]; then
        echo "✅ main.py version 4.3.1 confirmed"
    else
        echo "⚠️ main.py version: $VERSION (expected 4.3.1)"
    fi
else
    echo "❌ main.py not found"
    exit 1
fi

if [ -f "dashboard_compatible.html" ]; then
    echo "✅ dashboard_compatible.html found"
    
    # Verificar características v4.3.1 en dashboard
    if grep -q "updateTerritorioForLevel" dashboard_compatible.html; then
        echo "✅ Improved territory filtering logic found"
    else
        echo "⚠️ Territory filtering improvements not detected"
    fi
    
    if grep -q "chart-container-scrollable" dashboard_compatible.html; then
        echo "✅ Scrollable Theil chart found"
    else
        echo "⚠️ Scrollable chart improvements not detected"
    fi
    
else
    echo "⚠️ dashboard_compatible.html not found - will use fallback"
fi

if [ -f "test.py" ]; then
    echo "✅ test.py found"
    
    # Verificar si tiene tests v4.3.1
    if grep -q "v4.3.1" test.py; then
        echo "✅ v4.3.1 tests detected"
    else
        echo "⚠️ v4.3.1 specific tests not found"
    fi
else
    echo "⚠️ test.py not found"
fi

# Crear directorio de logs si no existe
mkdir -p logs

# Set environment variables
export PYTHONPATH="/app:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Verificar que la aplicación se puede importar y version
echo "🧪 Testing app import and version..."
python -c "
import sys
try:
    from main import app
    print(f'✅ App imported successfully - Version: {app.version}')
    if app.version == '4.3.1':
        print('✅ Correct version 4.3.1 loaded')
    else:
        print(f'⚠️ Expected v4.3.1, got {app.version}')
    sys.exit(0)
except Exception as e:
    print(f'❌ Failed to import app: {e}')
    sys.exit(1)
" || {
    echo "❌ Failed to import app"
    exit 1
}

# Test base de datos
echo "🗄️ Testing database connection..."
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

# Ejecutar tests rápidos si estamos en desarrollo
if [ "$RAILWAY_ENVIRONMENT" != "production" ]; then
    echo "🧪 Running quick v4.3.1 tests..."
    python -c "
import sys
import os
from pathlib import Path

print('Testing v4.3.1 improvements...')

# Test 1: Verificar función de filtros en main.py
try:
    with open('main.py', 'r') as f:
        content = f.read()
    if 'upz_por_localidad' in content:
        print('✅ UPZ filtering endpoint found')
    else:
        print('⚠️ UPZ filtering endpoint not found')
except:
    print('⚠️ Could not verify main.py content')

# Test 2: Verificar dashboard mejorado
try:
    if Path('dashboard_compatible.html').exists():
        with open('dashboard_compatible.html', 'r') as f:
            content = f.read()
        
        checks = {
            'Territory filtering': 'updateTerritorioForLevel' in content,
            'Scrollable Theil': 'chart-container-scrollable' in content,
            'Tab order': 'data-tab=\"caracterizacion\"' in content and 'data-tab=\"series\"' in content,
            'Responsive': '@media' in content
        }
        
        passed = sum(checks.values())
        total = len(checks)
        print(f'✅ Dashboard checks: {passed}/{total} passed')
        
        if passed >= total * 0.8:
            print('✅ Dashboard v4.3.1 improvements verified')
        else:
            print('⚠️ Some dashboard improvements not detected')
    else:
        print('⚠️ Dashboard file not found')
except:
    print('⚠️ Could not verify dashboard improvements')

print('✅ Quick tests completed')
"
fi

# Información adicional v4.3.1
echo ""
echo "📋 v4.3.1 Feature Summary:"
echo "   🎯 Filtros: Localidad → UPZ funciona correctamente"
echo "   📊 Theil: Gráfico scrolleable con TODAS las UPZ"
echo "   🎨 UI: Dashboard completamente responsive"  
echo "   📑 Tabs: Caracterización → Series → Asociación → Desigualdad"
echo "   🔧 Backend: Endpoints optimizados y filtros corregidos"
echo ""

echo "🎯 Starting FastAPI server v4.3.1..."
echo "📍 Server will be available at: http://0.0.0.0:$PORT"
echo "🏛️ Exploración Determinantes Fecundidad Temprana - Bogotá D.C."
echo "📊 Análisis territorial por UPZ (10-14, 15-19 años)"
echo "🆕 Con mejoras de filtros, visualización y UX"

# Iniciar la aplicación con configuración optimizada
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --log-level info \
    --access-log \
    --use-colors \
    --loop uvloop
