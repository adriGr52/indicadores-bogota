# Exploración Determinantes Fecundidad Temprana - Bogotá D.C.

## Análisis territorial por UPZ (Cohortes 10-14, 15-19 años) - v4.3.0

Sistema web integral para el análisis de determinantes de fecundidad temprana en Bogotá D.C., con funcionalidades avanzadas de caracterización territorial, correlaciones estadísticas, medición de desigualdad y series temporales.

---

## **Características Principales**

### **Análisis Disponibles:**
- **Caracterización Territorial**: Estadísticas descriptivas por UPZ y localidad
- **Análisis de Asociación**: Correlaciones de Pearson y Spearman
- **Índice de Theil**: Medición de desigualdad territorial
- **Series Temporales**: Evolución de indicadores por UPZ

### 🎯 **Funcionalidades Clave:**
- ✅ Carga y validación de archivos Excel
- ✅ Dashboard interactivo responsive
- ✅ API REST completa con documentación automática
- ✅ Filtros por localidad, UPZ, año y cohortes
- ✅ Visualizaciones dinámicas con Chart.js
- ✅ Exportación de resultados

---

## **Deployment en Railway**

### **Paso 1: Preparar archivos**
Asegúrate de tener todos estos archivos en tu repositorio:
```
📁 proyecto/
├── 📄 main.py                     # API principal
├── 📄 dashboard_compatible.html   # Dashboard frontend
├── 📄 requirements.txt           # Dependencias Python
├── 📄 runtime.txt                # Versión Python
├── 📄 railway.json               # Configuración Railway
├── 📄 Procfile                   # Comandos deployment
├── 📄 start.sh                   # Script de inicio
├── 📄 test.py                    # Suite de testing
├── 📄 .gitignore                 # Archivos a ignorar
└── 📄 README.md                  # Documentación
```

### **Paso 2: Configurar Railway**
1. **Conectar repositorio**: Ve a [railway.app](https://railway.app) y conecta tu repo de GitHub
2. **Agregar PostgreSQL**: En el dashboard de Railway, añade el plugin PostgreSQL
3. **Variables de entorno**: Se configuran automáticamente con el plugin

### **Paso 3: Deploy automático**
Railway detectará automáticamente:
- `railway.json` para configuración
- `requirements.txt` para dependencias
- El comando de start desde `railway.json`

---

## 🛠️ **Configuración Local**

### **Requisitos:**
- Python 3.11.9
- PostgreSQL (opcional, usa SQLite por defecto)

### **Instalación:**
```bash
# 1. Clonar repositorio
git clone <tu-repo>
cd fecundidad-temprana

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar tests
python test.py

# 5. Iniciar aplicación
python main.py
# O usar el script: ./start.sh
```

### **Variables de entorno (opcionales):**
```bash
export PORT=8000
export DATABASE_URL=postgresql://user:pass@localhost/dbname
```

---

## 📁 **Estructura de Datos**

### **Archivo Excel esperado:**
El sistema espera un archivo `consolidado_indicadores_fecundidad.xlsx` con estas columnas:

#### **Columnas requeridas:**
- `Indicador_Nombre`: Nombre del indicador
- `Valor`: Valor numérico del indicador  
- `Unidad_Medida`: Unidad de medida

#### **Columnas opcionales:**
- `Nivel_Territorial`: LOCALIDAD/UPZ
- `Nombre Localidad`: Nombre de la localidad
- `Nombre_UPZ`: Nombre de la UPZ
- `Año_Inicio`: Año del dato
- `Grupo Etario Asociado`: Grupo de edad
- Y más... (ver `/debug/columns` para lista completa)

---

##  **Endpoints de la API**

### ** Principales:**
- `GET /` - Dashboard principal
- `GET /health` - Estado del sistema
- `GET /metadatos` - Información del dataset
- `POST /upload/excel` - Carga de archivos

### ** Análisis:**
- `GET /caracterizacion` - Estadísticas por territorio
- `GET /analisis/asociacion` - Correlaciones entre indicadores
- `GET /analisis/theil` - Índice de desigualdad territorial
- `GET /datos/series` - Series temporales

### ** Geografía:**
- `GET /geografia/upz_por_localidad` - UPZ por localidad

### ** Documentación:**
- `GET /docs` - Swagger UI (documentación interactiva)
- `GET /openapi.json` - Schema OpenAPI

---

##  **Testing**

```bash
# Ejecutar suite completa de tests
python test.py

# Los tests verifican:
✅ Importación de módulos
✅ Conectividad de base de datos  
✅ Funcionamiento de endpoints
✅ Validación de datos
✅ Detección de cohortes
✅ Limpieza de datos
```

---

## 📊 **Dashboard Interactivo**

### **Características:**
- * Diseño responsive**: Funciona en desktop, tablet y móvil
- ** Gráficos dinámicos**: Barras, dispersión, líneas temporales
- ** Filtros avanzados**: Por indicador, territorio, año
- ** Optimización móvil**: Interfaz adaptativa

### **Tabs disponibles:**
1. **Caracterización**: Estadísticas descriptivas
2. **Asociación**: Correlaciones entre variables  
3. ** Desigualdad**: Índice de Theil
4. **Series**: Evolución temporal

---

## 🗄️ **Base de Datos**

### **Modelo principal:**
```sql
tabla: indicadores_fecundidad
├── id (PK)
├── indicador_nombre
├── valor
├── unidad_medida
├── nivel_territorial
├── nombre_localidad  
├── nombre_upz
├── año_inicio
├── grupo_etario_asociado
└── ... (30+ campos)
```

### **Índices optimizados:**
- `idx_localidad_indicador`
- `idx_upz_grupo` 
- `idx_nivel_año`

---

##  **Funcionalidades Especiales**

### ** Detección automática de cohortes:**
El sistema identifica automáticamente grupos etarios:
- **10-14 años**: Niñas 
- **15-19 años**: Adolescentes

### ** Análisis estadísticos:**
- Estadísticas descriptivas completas
- Correlaciones de Pearson y Spearman
- Índice de Theil para desigualdad
- Coeficientes de variación

### **🛡 Validación de datos:**
- Limpieza automática de valores nulos
- Normalización de texto
- Validación de tipos de datos
- Manejo de errores robusto

---

##  **Solución de Problemas**

### **Error de dependencias:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### **Error de base de datos:**
- El sistema usa SQLite como fallback automático
- Para PostgreSQL, verifica la variable `DATABASE_URL`

### **Error de archivo Excel:**
- Verifica que el archivo tenga las columnas requeridas
- Usa `/debug/columns` para ver la estructura esperada

### **Tests fallando:**
```bash
# Verificar configuración
python -c "from main import app; print('✅ OK')"

# Verificar base de datos
python -c "from main import SessionLocal; db=SessionLocal(); print('✅ DB OK')"
```

---

##  **Contribuciones**

El proyecto está diseñado para ser extensible:

1. **Nuevos análisis**: Agregar endpoints en `main.py`
2. **Nuevas visualizaciones**: Modificar `dashboard_compatible.html`
3. **Nuevos tests**: Agregar funciones a `test.py`

---

##  **Licencia**

Este proyecto está desarrollado para análisis de políticas públicas en Bogotá D.C.

---

## **Enlaces Útiles**

- [Railway Documentation](https://docs.railway.app/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Chart.js Documentation](https://www.chartjs.org/docs/)

---

##  **Soporte**

Para reportar issues o solicitar nuevas funcionalidades, utiliza el sistema de issues del repositorio.

---

** Versión 4.3.0** - Sistema optimizado sin funcionalidad de brechas, enfocado en análisis territorial por UPZ.

se debe cumplir con un requisito mínimo de plantilla
