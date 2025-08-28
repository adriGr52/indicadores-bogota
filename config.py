"""
Configuración centralizada para Fecundidad Temprana API v4.3.1
Contiene todas las configuraciones, constantes y parámetros del sistema
"""

import os
from pathlib import Path
from typing import List, Dict

# Versión del sistema
VERSION = "4.3.1"
APP_NAME = "Exploración Determinantes Fecundidad Temprana - Bogotá D.C."
DESCRIPTION = "Análisis integral por UPZ con filtros corregidos y funcionalidad completa"

# Configuración de base de datos
DATABASE_CONFIG = {
    "url": os.getenv("DATABASE_URL", "sqlite:///fecundidad_temprana.db"),
    "echo": os.getenv("DB_ECHO", "false").lower() == "true",
    "pool_pre_ping": True,
    "pool_recycle": 300,
    "connect_timeout": 10
}

# Configuración del servidor
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": int(os.getenv("PORT", 8000)),
    "workers": int(os.getenv("WORKERS", 1)),
    "log_level": os.getenv("LOG_LEVEL", "info"),
    "access_log": True,
    "use_colors": True
}

# Configuración de CORS
CORS_CONFIG = {
    "allow_origins": ["*"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"]
}

# Cohortes válidas para análisis
COHORTES_VALIDAS = {"10-14", "15-19"}

# Columnas requeridas en Excel
COLUMNAS_REQUERIDAS = [
    "Indicador_Nombre", 
    "Valor", 
    "Unidad_Medida"
]

# Columnas opcionales en Excel
COLUMNAS_OPCIONALES = [
    "origen_archivo", "archivo_hash", "Dimensión", "Tipo_Medida",
    "Nivel_Territorial", "ID Localidad", "Nombre Localidad", 
    "ID_UPZ", "Nombre_UPZ", "Área Geográfica", "Año_Inicio",
    "Periodicidad", "Poblacion Base", "Semaforo", 
    "Grupo Etario Asociado", "Sexo", "Tipo de Unidad", 
    "Tipo de Unidad Observación", "Fuente", "URL_Fuente (Opcional)"
]

# Mapeo de columnas (Excel -> Base de datos)
COLUMN_MAPPING = {
    'Dimensión': 'dimension', 
    'Área Geográfica': 'area_geografica', 
    'Tipo de Unidad Observación': 'observacion', 
    'URL_Fuente (Opcional)': 'url_fuente'
}

# Configuración de archivos
FILE_CONFIG = {
    "allowed_extensions": [".xlsx", ".xls"],
    "max_file_size": 50 * 1024 * 1024,  # 50MB
    "upload_folder": "uploads",
    "temp_folder": "temp"
}

# Configuración de logging
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["console"]
}

# Límites de la aplicación
LIMITS = {
    "max_records_per_upload": 100000,
    "max_chart_points": 1000,
    "max_export_records": 50000,
    "query_timeout": 30
}

# Configuración de Chart.js
CHART_CONFIG = {
    "max_labels": 50,
    "colors": {
        "primary": "#2563eb",
        "secondary": "#64748b", 
        "success": "#10b981",
        "warning": "#f59e0b",
        "error": "#ef4444",
        "info": "#06b6d4"
    },
    "default_height": 400,
    "responsive": True
}

# Patrones para extracción de grupos de edad
EDAD_PATTERNS = {
    "10-14": [
        r"10\s*[-aá]\s*14",
        r"10\D*14", 
        r"niñas de 10 a 14",
        r"10 a 14 años"
    ],
    "15-19": [
        r"15\s*[-aá]\s*19",
        r"15\D*19",
        r"mujeres de 15 a 19", 
        r"15 a 19 años"
    ]
}

# Palabras clave para identificar indicadores de fecundidad
FECUNDIDAD_KEYWORDS = [
    "fecund", "natalidad", "nacimiento", 
    "maternidad", "embarazo", "gestación"
]

# Valores considerados como NaN
NAN_VALUES = {
    "", "nan", "nd", "no_data", "none", "null", 
    "n/a", "na", "not available", "sin datos"
}

# Configuración de índices de base de datos
DATABASE_INDEXES = [
    ("idx_localidad_indicador", ["nombre_localidad", "indicador_nombre"]),
    ("idx_upz_grupo", ["nombre_upz", "grupo_etario_asociado"]),
    ("idx_nivel_año", ["nivel_territorial", "año_inicio"])
]

# Configuración de health check
HEALTH_CONFIG = {
    "timeout": 5,
    "required_fields": ["status", "version", "database", "registros", "timestamp"]
}

# Configuración de Theil
THEIL_CONFIG = {
    "min_territories": 2,
    "categories": {
        "baja": 0.1,
        "moderada": 0.3
    }
}

# Configuración de correlación
CORRELATION_CONFIG = {
    "min_common_territories": 3,
    "significance_level": 0.05,
    "categories": {
        "muy_debil": 0.3,
        "debil": 0.5,
        "moderada": 0.7,
        "fuerte": 1.0
    }
}

# Mensajes de error estándar
ERROR_MESSAGES = {
    "no_data": "Sin datos para los filtros especificados",
    "invalid_file": "Archivo no válido. Use archivos .xlsx o .xls",
    "missing_columns": "Columnas faltantes en el archivo",
    "upload_failed": "Error procesando archivo",
    "insufficient_data": "Insuficientes datos para el análisis",
    "no_variation": "Una de las variables no tiene variación",
    "db_error": "Error de conectividad con la base de datos"
}

# Configuración de desarrollo
DEV_CONFIG = {
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "reload": os.getenv("RELOAD", "false").lower() == "true",
    "test_mode": os.getenv("TEST_MODE", "false").lower() == "true"
}

# Rutas de archivos importantes
PATHS = {
    "dashboard": Path("dashboard_compatible.html"),
    "uploads": Path("uploads"),
    "temp": Path("temp"),
    "logs": Path("logs"),
    "exports": Path("exports")
}

def get_database_url() -> str:
    """Obtiene URL de base de datos con fallback a SQLite"""
    url = DATABASE_CONFIG["url"]
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url

def is_production() -> bool:
    """Determina si estamos en producción"""
    return os.getenv("RAILWAY_ENVIRONMENT") is not None

def get_cors_origins() -> List[str]:
    """Obtiene orígenes permitidos para CORS"""
    if is_production():
        # En producción, podrías querer ser más específico
        return ["*"]  # Cambiar según necesidades
    return ["*"]

def validate_config() -> Dict[str, bool]:
    """Valida la configuración del sistema"""
    checks = {
        "database_url_set": bool(os.getenv("DATABASE_URL")),
        "port_valid": 1 <= SERVER_CONFIG["port"] <= 65535,
        "dashboard_exists": PATHS["dashboard"].exists(),
        "required_env_vars": all(
            os.getenv(var) for var in ["PORT"] if is_production()
        )
    }
    return checks

# Configuración específica para Railway
RAILWAY_CONFIG = {
    "builder": "NIXPACKS",
    "start_command": "bash start.sh",
    "healthcheck_path": "/health",
    "healthcheck_timeout": 120,
    "restart_policy": "ON_FAILURE",
    "max_retries": 10
}

# Metadatos de la aplicación
APP_METADATA = {
    "name": APP_NAME,
    "version": VERSION,
    "description": DESCRIPTION,
    "author": "Equipo de Análisis Territorial",
    "license": "Uso Gubernamental",
    "repository": "github.com/tu-usuario/fecundidad-temprana",
    "tags": ["fecundidad", "bogota", "analisis-territorial", "upz", "salud-publica"]
}

# Configuración de exportación
EXPORT_CONFIG = {
    "formats": ["json", "csv", "excel"],
    "max_records": LIMITS["max_export_records"],
    "chunk_size": 1000,
    "temp_lifetime": 3600  # 1 hora
}

def get_app_config() -> Dict:
    """Retorna configuración completa de la aplicación"""
    return {
        "app": {
            "name": APP_NAME,
            "version": VERSION,
            "description": DESCRIPTION
        },
        "server": SERVER_CONFIG,
        "database": DATABASE_CONFIG,
        "cors": CORS_CONFIG,
        "file_handling": FILE_CONFIG,
        "limits": LIMITS,
        "development": DEV_CONFIG,
        "is_production": is_production()
    }

if __name__ == "__main__":
    # Test de configuración
    import json
    config = get_app_config()
    validation = validate_config()
    
    print("🔧 Configuración de la Aplicación v4.3.1")
    print("=" * 50)
    print(json.dumps(config, indent=2, default=str))
    print("\n🔍 Validación de Configuración:")
    for check, result in validation.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}: {result}")
