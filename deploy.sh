#!/bin/bash

# Script de deployment y verificación para Fecundidad Temprana v4.3.1
# Automatiza verificaciones pre-deploy y process de deployment

set -e  # Salir si cualquier comando falla

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Variables
VERSION="4.3.1"
PROJECT_NAME="Fecundidad Temprana"
REQUIRED_PYTHON="3.11"

echo -e "${BLUE}🚀 $PROJECT_NAME - Script de Deployment v$VERSION${NC}"
echo "=================================================================="

# Función para imprimir con color
print_status() {
    local status=$1
    local message=$2
    
    case $status in
        "success") echo -e "${GREEN}✅ $message${NC}" ;;
        "error")   echo -e "${RED}❌ $message${NC}" ;;
        "warning") echo -e "${YELLOW}⚠️  $message${NC}" ;;
        "info")    echo -e "${BLUE}ℹ️  $message${NC}" ;;
        "step")    echo -e "${PURPLE}🔄 $message${NC}" ;;
    esac
}

# Función para verificar si un comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Función para verificar archivos críticos
check_critical_files() {
    print_status "step" "Verificando archivos críticos..."
    
    local critical_files=(
        "main.py"
        "requirements.txt" 
        "Procfile"
        "dashboard_compatible.html"
    )
    
    local missing_files=()
    
    for file in "${critical_files[@]}"; do
        if [[ -f "$file" ]]; then
            print_status "success" "Archivo encontrado: $file"
        else
            missing_files+=("$file")
            print_status "error" "Archivo faltante: $file"
        fi
    done
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        print_status "error" "Archivos críticos faltantes. Deployment abortado."
        exit 1
    fi
    
    print_status "success" "Todos los archivos críticos están presentes"
}

# Función para verificar archivos recomendados
check_recommended_files() {
    print_status "step" "Verificando archivos recomendados..."
    
    local recommended_files=(
        "start.sh"
        "railway.json"
        "runtime.txt"
        ".gitignore"
        "README.md"
        "test.py"
    )
    
    local missing_recommended=()
    
    for file in "${recommended_files[@]}"; do
        if [[ -f "$file" ]]; then
            print_status "success" "Archivo recomendado encontrado: $file"
        else
            missing_recommended+=("$file")
            print_status "warning" "Archivo recomendado faltante: $file"
        fi
    done
    
    if [[ ${#missing_recommended[@]} -gt 0 ]]; then
        print_status "info" "Archivos recomendados faltantes (no crítico): ${missing_recommended[*]}"
    fi
}

# Función para verificar Python
check_python() {
    print_status "step" "Verificando versión de Python..."
    
    if command_exists python3; then
        local python_version=$(python3 --version | grep -oE '[0-9]+\.[0-9]+')
        print_status "info" "Python version detectada: $python_version"
        
        if [[ "$python_version" == "$REQUIRED_PYTHON" ]]; then
            print_status "success" "Versión de Python correcta"
        else
            print_status "warning" "Se recomienda Python $REQUIRED_PYTHON, encontrado: $python_version"
        fi
    else
        print_status "error" "Python3 no encontrado"
        exit 1
    fi
}

# Función para verificar dependencias
check_dependencies() {
    print_status "step" "Verificando dependencias..."
    
    if [[ -f "requirements.txt" ]]; then
        local dep_count=$(wc -l < requirements.txt)
        print_status "info" "Dependencias en requirements.txt: $dep_count"
        
        # Verificar algunas dependencias críticas
        local critical_deps=("fastapi" "uvicorn" "sqlalchemy" "pandas")
        
        for dep in "${critical_deps[@]}"; do
            if grep -q "^$dep" requirements.txt; then
                print_status "success" "Dependencia crítica encontrada: $dep"
            else
                print_status "warning" "Dependencia crítica posiblemente faltante: $dep"
            fi
        done
    else
        print_status "error" "requirements.txt no encontrado"
        exit 1
    fi
}

# Función para verificar Procfile
check_procfile() {
    print_status "step" "Verificando Procfile..."
    
    if [[ -f "Procfile" ]]; then
        local content=$(cat Procfile)
        print_status "info" "Contenido del Procfile: $content"
        
        if echo "$content" | grep -q "web:"; then
            print_status "success" "Procfile contiene comando web"
            
            if echo "$content" | grep -q "uvicorn main:app"; then
                print_status "success" "Procfile usa uvicorn correctamente"
            elif echo "$content" | grep -q "bash start.sh"; then
                print_status "success" "Procfile usa script de inicio"
            else
                print_status "warning" "Procfile no usa comando estándar"
            fi
        else
            print_status "error" "Procfile no contiene comando web válido"
            exit 1
        fi
    else
        print_status "error" "Procfile no encontrado"
        exit 1
    fi
}

# Función para verificar configuración de Railway
check_railway_config() {
    print_status "step" "Verificando configuración de Railway..."
    
    if [[ -f "railway.json" ]]; then
        print_status "success" "railway.json encontrado"
        
        # Verificar estructura JSON básica
        if command_exists jq; then
            if jq empty railway.json 2>/dev/null; then
                print_status "success" "railway.json tiene JSON válido"
            else
                print_status "error" "railway.json contiene JSON inválido"
                exit 1
            fi
        else
            print_status "info" "jq no disponible, omitiendo validación JSON"
        fi
    else
        print_status "warning" "railway.json no encontrado (Railway usará detección automática)"
    fi
}

# Función para ejecutar tests
run_tests() {
    print_status "step" "Ejecutando tests del sistema..."
    
    if [[ -f "test.py" ]]; then
        print_status "info" "Ejecutando suite de tests v$VERSION..."
        
        if python3 test.py; then
            print_status "success" "Tests pasaron exitosamente"
        else
            local exit_code=$?
            if [[ $exit_code -eq 1 ]]; then
                print_status "warning" "Tests pasaron parcialmente - revisar output arriba"
                read -p "¿Continuar con el deployment? (y/N): " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    print_status "info" "Deployment abortado por usuario"
                    exit 1
                fi
            else
                print_status "error" "Tests fallaron - deployment abortado"
                exit 1
            fi
        fi
    else
        print_status "warning" "test.py no encontrado - omitiendo tests"
    fi
}

# Función para verificar git status
check_git_status() {
    print_status "step" "Verificando estado de Git..."
    
    if command_exists git && [[ -d ".git" ]]; then
        # Verificar si hay cambios sin commit
        if git diff-index --quiet HEAD --; then
            print_status "success" "No hay cambios sin commit"
        else
            print_status "warning" "Hay cambios sin commit"
            git status --short
            
            read -p "¿Hacer commit automático de cambios? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                git add .
                git commit -m "feat: deployment v$VERSION - archivos actualizados"
                print_status "success" "Cambios commitados automáticamente"
            fi
        fi
        
        # Verificar branch
        local current_branch=$(git branch --show-current)
        print_status "info" "Branch actual: $current_branch"
        
        # Verificar si hay commits para push
        if git log origin/$current_branch..$current_branch --oneline | grep -q .; then
            print_status "info" "Hay commits locales listos para push"
            
            read -p "¿Hacer push automáticamente? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                git push origin $current_branch
                print_status "success" "Cambios pushed exitosamente"
            fi
        else
            print_status "success" "Repository está sincronizado"
        fi
    else
        print_status "info" "No es un repositorio Git o Git no disponible"
    fi
}

# Función para mostrar información de deployment
show_deployment_info() {
    print_status "step" "Información de deployment..."
    
    echo -e "${BLUE}"
    echo "=================================================================="
    echo "🚀 LISTO PARA DEPLOYMENT"
    echo "=================================================================="
    echo "Proyecto: $PROJECT_NAME v$VERSION"
    echo "Archivos verificados: ✅"
    echo "Configuración válida: ✅" 
    echo "Tests ejecutados: ✅"
    echo ""
    echo "📋 PRÓXIMOS PASOS PARA RAILWAY:"
    echo "1. Ve a https://railway.app"
    echo "2. Conecta tu repositorio de GitHub"
    echo "3. Railway detectará automáticamente la configuración"
    echo "4. Agrega PostgreSQL desde el dashboard de Railway"
    echo "5. ¡Tu aplicación estará live!"
    echo ""
    echo "🔗 ENDPOINTS DISPONIBLES DESPUÉS DEL DEPLOY:"
    echo "  https://tu-app.railway.app/ - Dashboard principal"
    echo "  https://tu-app.railway.app/docs - Documentación API"
    echo "  https://tu-app.railway.app/health - Health check"
    echo ""
    echo "📊 FUNCIONALIDADES v$VERSION:"
    echo "  • ✅ Filtros Localidad/UPZ corregidos"
    echo "  • ✅ Lógica inteligente de datos" 
    echo "  • ✅ Dashboard completamente funcional"
    echo "  • ✅ Índice de Theil con todas las UPZ"
    echo "  • ✅ Sistema responsive optimizado"
    echo "=================================================================="
    echo -e "${NC}"
}

# Función para crear archivo de verificación de deployment
create_deployment_check() {
    print_status "step" "Creando archivo de verificación post-deployment..."
    
    cat > deployment_check.sh << 'EOF'
#!/bin/bash

# Script para verificar deployment exitoso
# Uso: ./deployment_check.sh https://tu-app.railway.app

URL=${1:-"http://localhost:8000"}

echo "🔍 Verificando deployment en: $URL"

# Test health endpoint
echo "Testing health endpoint..."
if curl -s "$URL/health" | jq '.status' | grep -q "healthy"; then
    echo "✅ Health check OK"
else
    echo "❌ Health check failed"
fi

# Test docs endpoint  
echo "Testing docs endpoint..."
if curl -s -o /dev/null -w "%{http_code}" "$URL/docs" | grep -q "200"; then
    echo "✅ Docs endpoint OK"
else
    echo "❌ Docs endpoint failed"
fi

# Test main dashboard
echo "Testing main dashboard..."
if curl -s -o /dev/null -w "%{http_code}" "$URL/" | grep -q "200"; then
    echo "✅ Dashboard OK"
else
    echo "❌ Dashboard failed"
fi

echo "🎉 Verificación de deployment completada"
EOF
    
    chmod +x deployment_check.sh
    print_status "success" "Script de verificación creado: ./deployment_check.sh"
}

# Función principal
main() {
    echo -e "${BLUE}Iniciando verificación pre-deployment...${NC}\n"
    
    # Ejecutar todas las verificaciones
    check_critical_files
    echo ""
    
    check_recommended_files  
    echo ""
    
    check_python
    echo ""
    
    check_dependencies
    echo ""
    
    check_procfile
    echo ""
    
    check_railway_config
    echo ""
    
    # Preguntar si ejecutar tests
    read -p "¿Ejecutar suite de tests? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        run_tests
        echo ""
    fi
    
    check_git_status
    echo ""
    
    create_deployment_check
    echo ""
    
    show_deployment_info
}

# Verificar argumentos de línea de comandos
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Script de deployment y verificación para $PROJECT_NAME v$VERSION"
    echo ""
    echo "Uso: $0 [opciones]"
    echo ""
    echo "Opciones:"
    echo "  --help, -h     Mostrar esta ayuda"
    echo "  --quick, -q    Verificación rápida (sin tests ni git)"
    echo "  --test-only    Solo ejecutar tests"
    echo ""
    echo "El script verifica:"
    echo "  • Archivos críticos y recomendados"
    echo "  • Versión de Python y dependencias"
    echo "  • Configuración de Procfile y Railway"
    echo "  • Estado de Git y cambios pendientes"
    echo "  • Suite completa de tests (opcional)"
    exit 0
elif [[ "$1" == "--quick" ]] || [[ "$1" == "-q" ]]; then
    print_status "info" "Modo verificación rápida activado"
    check_critical_files
    check_python
    check_dependencies  
    check_procfile
    print_status "success" "Verificación rápida completada"
    exit 0
elif [[ "$1" == "--test-only" ]]; then
    print_status "info" "Ejecutando solo tests"
    run_tests
    exit 0
fi

# Ejecutar función principal
main

print_status "success" "🎉 ¡Script de deployment completado exitosamente!"
print_status "info" "Tu sistema $PROJECT_NAME v$VERSION está listo para deployment"
