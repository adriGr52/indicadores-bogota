# Exploración Determinantes Fecundidad Temprana - Bogotá D.C.

## Sistema de Análisis Territorial por UPZ (v4.3.1 - Totalmente Funcional)

Sistema web integral para el análisis de determinantes de fecundidad temprana en Bogotá D.C., con funcionalidades completas de caracterización territorial, correlaciones estadísticas, medición de desigualdad y series temporales.

---

## 🎯 **¿Qué hace este sistema?**

Este sistema permite analizar datos de fecundidad temprana en Bogotá D.C. a nivel territorial (localidades y UPZ), proporcionando:

- **📊 Caracterización territorial**: Estadísticas descriptivas por territorio
- **📈 Series temporales**: Evolución de indicadores a lo largo del tiempo  
- **🔗 Análisis de correlación**: Relaciones entre diferentes indicadores
- **⚖️ Índice de desigualdad**: Medición de desigualdad territorial usando el índice de Theil
- **👥 Análisis por cohortes**: Comparación entre grupos 10-14 y 15-19 años

---

## 🆕 **Novedades v4.3.1 - Completamente Funcional**

### ✅ **Problemas Solucionados:**
- **Filtros Localidad/UPZ CORREGIDOS**: Ahora funciona la lógica Localidad → UPZ correctamente
- **Lógica Inteligente**: Si no hay datos UPZ, automáticamente busca datos de localidad
- **Índice de Theil Completo**: Gráfico scrolleable que muestra TODAS las UPZ/localidades
- **Dashboard Totalmente Funcional**: Interfaz responsive que realmente carga y muestra datos
- **Backend Optimizado**: Endpoints con manejo robusto de datos y fallback inteligente

### 🎮 **Flujo de Uso Corregido:**
1. **Carga datos** → Selecciona archivo Excel y carga
2. **Selecciona localidad** → Automáticamente filtra UPZ de esa localidad
3. **Elige "Todas"** → Muestra todas las unidades del nivel correspondiente
4. **Sistema inteligente** → Si no hay datos específicos, busca datos más generales

---

## 🚀 **Instalación y Deploy**

### **Opción 1: Deploy Rápido en Railway**

1. **Subir archivos a GitHub**:
   ```bash
   git clone <tu-repositorio>
   cd fecundidad-temprana
   # Copiar todos los archivos actualizados v4.3.1
   git add .
   git commit -m "feat: Sistema v4.3.1 completamente funcional"
   git push
   ```

2. **Conectar a Railway**:
   - Ve a [railway.app](https://railway.app)
   - Conecta tu repositorio de GitHub
   - Railway detectará automáticamente la configuración

3. **Agregar PostgreSQL**:
   - En el dashboard de Railway: "+ New" → "Database" → "Add PostgreSQL"
   - Railway configurará automáticamente la variable `DATABASE_URL`

4. **¡Listo!** - Tu sistema estará disponible en la URL que Railway te proporcione

### **Opción 2: Instalación Local**

```bash
# 1. Clonar y configurar
git clone <tu-repositorio>
cd fecundidad-temprana
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar tests
python test.py

# 4. Iniciar aplicación
python main.py
# O usar: ./start.sh
```

---

## 📋 **Cómo Usar el Sistema**

### **1. Cargar Datos**
- Sube un archivo Excel con tus indicadores de fecundidad
- El sistema esperará las columnas: `Indicador_Nombre`, `Valor`, `Unidad_Medida`
- Columnas opcionales: `Nombre Localidad`, `Nombre_UPZ`, `Año_Inicio`, etc.

### **2. Análisis Disponibles**

#### **📊 Caracterización Territorial**
- Selecciona un indicador
- Escoge localidad específica o "Todas"
- El sistema muestra estadísticas descriptivas y gráficos
- **Lógica inteligente**: Si no hay datos UPZ, muestra datos de localidad

#### **📈 Series Temporales**
- Selecciona indicador y territorio
- Ve la evolución temporal con análisis de tendencias
- Calcula variaciones y direcciones de cambio

#### **🎯 Análisis de Asociación**
- Selecciona dos indicadores para correlacionar
- Obtén coeficientes de Pearson y Spearman
- Visualiza diagramas de dispersión

#### **⚖️ Índice de Desigualdad de Theil**
- Mide la desigualdad territorial de un indicador
- **Novedad v4.3.1**: Gráfico scrolleable con TODAS las UPZ
- Interpretación automática (Baja/Moderada/Alta)

### **3. Filtros Inteligentes**
- **Localidad → UPZ**: Al seleccionar localidad, se filtran automáticamente las UPZ
- **"Todas"**: Muestra todas las unidades del nivel seleccionado  
- **Fallback automático**: Si no hay datos específicos, busca datos más generales

---

## 🏗️ **Estructura del Proyecto**

```
fecundidad-temprana/
├── 📄 main.py                     # API FastAPI principal
├── 📄 dashboard_compatible.html   # Dashboard web funcional
├── 📄 requirements.txt           # Dependencias Python
├── 📄 runtime.txt                # Versión de Python (3.11.9)
├── 📄 railway.json               # Configuración Railway
├── 📄 start.sh                   # Script de inicio
├── 📄 Procfile                   # Comando de deploy
├── 📄 test.py                    # Suite de testing completa
├── 📄 .gitignore                 # Archivos a ignorar
└── 📄 README.md                  # Esta documentación
```

---

## 🔗 **API Endpoints v4.3.1**

### **🏠 Principales**
- `GET /` → Dashboard principal
- `GET /health` → Estado del sistema
- `POST /upload/excel` → Cargar datos Excel
- `GET /metadatos` → Información del dataset

### **📊 Análisis**
- `GET /caracterizacion` → Estadísticas por territorio con lógica inteligente
- `GET /analisis/theil` → Índice de desigualdad (TODAS las unidades)
- `GET /analisis/asociacion` → Correlaciones entre indicadores
- `GET /datos/series` → Series temporales
- `GET /brechas/cohortes` → Análisis por grupos etarios

### **🗺️ Geografía**
- `GET /geografia/upz_por_localidad` → UPZ filtradas por localidad

### **📚 Documentación**
- `GET /docs` → Swagger UI (documentación interactiva)
- `GET /openapi.json` → Schema OpenAPI

---

## 🧪 **Testing**

El sistema incluye una suite completa de tests:

```bash
python test.py
```

**Tests incluidos:**
- ✅ Conectividad de base de datos
- ✅ Funcionamiento de endpoints  
- ✅ Filtrado UPZ por localidad
- ✅ Lógica inteligente de fallback
- ✅ Índice de Theil completo
- ✅ Estructura del dashboard
- ✅ Validación de datos
- ✅ Procesamiento de Excel

---

## 📊 **Dashboard Funcional**

### **🎨 Características del Dashboard:**
- **Responsive**: Funciona en móviles, tablets y desktop
- **Interactivo**: Gráficos dinámicos con Chart.js
- **Intuitivo**: Filtros que se actualizan automáticamente
- **Inteligente**: Manejo de casos sin datos

### **📱 Pestañas del Dashboard:**
1. **Caracterización** (Principal) → Estadísticas descriptivas
2. **Series Temporales** → Evolución en el tiempo
3. **Asociación** → Correlaciones entre variables  
4. **Desigualdad** → Índice de Theil con scroll

### **🔄 Flujo de Filtros:**
```
Seleccionar Localidad → Se filtran UPZ automáticamente
Seleccionar "Todas" → Muestra todas las unidades
Sistema Inteligente → Busca datos donde estén disponibles
```

---

## 🗄️ **Base de Datos**

### **Modelo de Datos:**
```sql
tabla: indicadores_fecundidad
├── indicador_nombre (string) 
├── valor (float)
├── unidad_medida (string)
├── nivel_territorial (LOCALIDAD/UPZ)
├── nombre_localidad (string)
├── nombre_upz (string, nullable)
├── año_inicio (integer, nullable)
├── grupo_etario_asociado (string, nullable)
└── ... (25+ campos adicionales)
```

### **Lógica Inteligente:**
El sistema implementa una lógica de fallback:
1. Busca datos a nivel UPZ primero
2. Si no hay datos UPZ, busca a nivel localidad
3. Siempre muestra los datos más específicos disponibles

---

## ⚙️ **Configuración**

### **Variables de Entorno:**
```bash
PORT=8000                    # Puerto del servidor
DATABASE_URL=postgresql://... # URL de PostgreSQL (Railway lo configura automáticamente)
PYTHONPATH=/app             # Path de Python
PYTHONUNBUFFERED=1          # Output sin buffer
```

### **Dependencias Principales:**
- **FastAPI 0.104.1**: Framework web moderno
- **SQLAlchemy 2.0.23**: ORM para base de datos
- **Pandas 2.1.4**: Procesamiento de datos
- **NumPy/SciPy**: Cálculos científicos
- **Chart.js 4.4.1**: Visualizaciones web

---

## 🔧 **Funcionalidades Especiales**

### **🎯 Detección Automática de Cohortes:**
```python
# El sistema identifica automáticamente:
"Tasa de fecundidad en niñas de 10 a 14 años" → Cohorte 10-14
"Embarazo adolescente 15-19" → Cohorte 15-19
```

### **📈 Análisis Estadísticos:**
- Estadísticas descriptivas completas (media, mediana, percentiles)
- Correlaciones de Pearson y Spearman
- Índice de Theil para medir desigualdad territorial
- Análisis de tendencias temporales

### **🛡️ Validación Robusta:**
- Limpieza automática de valores nulos
- Normalización de texto
- Manejo de errores graceful
- Fallback a SQLite si PostgreSQL falla

### **🎨 Visualizaciones Avanzadas:**
- Gráficos de barras responsive
- Series temporales con tendencias
- Diagramas de dispersión interactivos
- **Gráfico Theil scrolleable** que muestra todas las unidades

---

## 🐛 **Solución de Problemas**

### **❓ El dashboard no muestra datos**
```bash
# 1. Verificar que se cargaron datos
curl http://tu-app.railway.app/health

# 2. Verificar metadatos
curl http://tu-app.railway.app/metadatos

# 3. Probar endpoint específico
curl "http://tu-app.railway.app/caracterizacion?indicador=tu_indicador"
```

### **❓ Error al cargar Excel**
- Verificar que el archivo tenga las columnas requeridas: `Indicador_Nombre`, `Valor`, `Unidad_Medida`
- Revisar que los valores numéricos estén en formato correcto
- El sistema limpia automáticamente datos malformados

### **❓ Los filtros UPZ no funcionan**
```javascript
// v4.3.1 corrigió esto completamente
// Los filtros ahora funcionan así:
// Localidad = "Kennedy" → UPZ se filtra a UPZ de Kennedy
// Localidad = "Todas" → UPZ muestra todas las UPZ
```

### **❓ Error de despliegue en Railway**
```bash
# Verificar archivos principales:
ls -la main.py Procfile requirements.txt

# El Procfile debe contener EXACTAMENTE:
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## 📈 **Casos de Uso Típicos**

### **🔍 Caso 1: Análisis de Desigualdad Territorial**
1. Cargar datos de fecundidad por UPZ
2. Ir a pestaña "Desigualdad" 
3. Seleccionar indicador (ej: "Tasa específica de fecundidad 10-14")
4. Calcular índice de Theil
5. **Ver TODAS las UPZ** en el gráfico scrolleable
6. Identificar UPZ con mayor desigualdad

### **🔍 Caso 2: Evolución Temporal por Territorio**
1. Ir a pestaña "Series Temporales"
2. Seleccionar indicador y territorio específico
3. Ver evolución año a año
4. Analizar tendencias (creciente/decreciente)

### **🔍 Caso 3: Correlaciones entre Determinantes**
1. Ir a pestaña "Asociación"
2. Seleccionar dos indicadores (ej: educación vs fecundidad)
3. Ver correlación de Pearson y Spearman
4. Interpretar significancia estadística

---

## 🤝 **Contribuir al Proyecto**

### **🔧 Desarrollo Local:**
```bash
# Setup
git clone <repo>
cd fecundidad-temprana
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Desarrollo
python main.py  # Inicia en modo desarrollo
python test.py  # Ejecuta todos los tests

# Ver documentación
# http://localhost:8000/docs
```

### **📝 Estructura de Commits:**
```
feat: nueva funcionalidad de análisis
fix: corregir filtros UPZ por localidad  
docs: actualizar documentación README
test: agregar tests para endpoint X
```

### **🧪 Agregar Nuevos Tests:**
```python
def test_nueva_funcionalidad():
    """Test para nueva funcionalidad"""
    # Tu código de test aquí
    return True
```

---

## 📜 **Licencia y Uso**

Este proyecto está desarrollado para análisis de políticas públicas en Bogotá D.C. 

**Uso permitido:**
- ✅ Análisis de políticas públicas
- ✅ Investigación académica  
- ✅ Reportes institucionales
- ✅ Adaptación para otras ciudades

---

## 🆘 **Soporte y Contacto**

### **🐛 Reportar Issues:**
Usa el sistema de issues de GitHub para reportar bugs o solicitar funcionalidades.

### **📞 Issues Comunes Solucionados:**
- ✅ **#001**: Filtros Localidad/UPZ no funcionaban → **SOLUCIONADO v4.3.1**
- ✅ **#002**: Dashboard no cargaba datos → **SOLUCIONADO v4.3.1**  
- ✅ **#003**: Índice Theil solo top 10 → **SOLUCIONADO v4.3.1**
- ✅ **#004**: No responsive en móviles → **SOLUCIONADO v4.3.1**

### **📚 Recursos Adicionales:**
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Railway Deploy Guide](https://docs.railway.app/)
- [Chart.js Documentation](https://www.chartjs.org/docs/)

---

## 📊 **Changelog v4.3.1**

### **✨ Nuevas Funcionalidades:**
- **Lógica inteligente de datos**: Fallback automático UPZ → Localidad
- **Dashboard completamente funcional**: Carga y muestra datos reales
- **Filtros corregidos**: Localidad → UPZ funciona perfectamente
- **Índice Theil completo**: Gráfico scrolleable con todas las unidades
- **Nuevo endpoint**: `/brechas/cohortes` para análisis por grupos etarios
- **Sistema responsive**: Funciona en todos los dispositivos

### **🐛 Bugs Solucionados:**
- **Filtros UPZ**: Lógica Localidad → UPZ completamente corregida
- **Dashboard sin datos**: Ahora carga y muestra información correctamente
- **Gráfico Theil limitado**: Ahora muestra todas las unidades (no solo top 10)
- **API endpoints**: Mejor manejo de casos sin datos
- **Validación Excel**: Más robusta para diferentes formatos de datos

### **🚀 Mejoras de Performance:**
- **Consultas optimizadas**: Mejor rendimiento en base de datos
- **Carga asíncrona**: Dashboard más responsivo
- **Gráficos optimizados**: Mejor rendimiento con muchos datos
- **Manejo de memoria**: Más eficiente para datasets grandes

### **📚 Documentación:**
- **README completo**: Documentación actualizada paso a paso
- **Tests exhaustivos**: Suite completa de testing
- **Ejemplos de uso**: Casos de uso típicos documentados
- **API documentation**: Swagger UI mejorada

---

## 🎉 **Resumen Final**

**El sistema v4.3.1 está TOTALMENTE FUNCIONAL** y listo para uso en producción:

✅ **Carga datos reales** desde archivos Excel  
✅ **Muestra visualizaciones** con Chart.js  
✅ **Filtros funcionan** correctamente (Localidad → UPZ)  
✅ **Lógica inteligente** encuentra datos donde estén disponibles  
✅ **Dashboard responsive** funciona en cualquier dispositivo  
✅ **Índice Theil completo** con todas las unidades territoriales  
✅ **Testing exhaustivo** con suite de tests automatizados  
✅ **Deploy simplificado** en Railway con un click  

---

> **🚀 Versión 4.3.1** - Sistema completamente funcional con todas las mejoras implementadas y probadas. Listo para análisis de fecundidad temprana en Bogotá D.C.
