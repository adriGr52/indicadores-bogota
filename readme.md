# Exploración Determinantes Fecundidad Temprana - Bogotá D.C.

## Análisis territorial por UPZ (Cohortes 10-14, 15-19 años) - v4.3.1

Sistema web integral para el análisis de determinantes de fecundidad temprana en Bogotá D.C., con funcionalidades avanzadas de caracterización territorial, correlaciones estadísticas, medición de desigualdad y series temporales.

---

## 🆕 **Novedades v4.3.1**

### **Mejoras Implementadas:**
- ✅ **Filtros Localidad/UPZ CORREGIDOS**: Ahora funciona correctamente la lógica Localidad → UPZ
- ✅ **Índice de Theil Mejorado**: Gráfico scrolleable que muestra TODAS las UPZ (no solo top 10)
- ✅ **Orden de Pestañas Optimizado**: Caracterización → Series → Asociación → Desigualdad
- ✅ **Dashboard Responsive Mejorado**: Mejor experiencia en móviles y tablets
- ✅ **Backend Optimizado**: Endpoints corregidos y mejor manejo de filtros

### **Flujo de Filtros Corregido:**
1. **Nivel = LOCALIDAD** → Campo "Localidad" muestra localidades → Al seleccionar "Todas" muestra todas
2. **Nivel = UPZ** → Campo "UPZ" muestra UPZ → Al filtrar por localidad específica, actualiza UPZ de esa localidad

---

## **Características Principales**

### **Análisis Disponibles:**
- **Caracterización Territorial**: Estadísticas descriptivas por UPZ y localidad
- **Series Temporales**: Evolución de indicadores por UPZ a lo largo del tiempo
- **Análisis de Asociación**: Correlaciones de Pearson y Spearman
- **Índice de Theil**: Medición de desigualdad territorial (TODAS las UPZ)

### 🎯 **Funcionalidades Clave:**
- ✅ Carga y validación de archivos Excel
- ✅ Dashboard interactivo responsive (optimizado móvil)
- ✅ API REST completa con documentación automática
- ✅ Filtros por localidad, UPZ, año y cohortes (CORREGIDOS)
- ✅ Visualizaciones dinámicas con Chart.js (scroll horizontal en Theil)
- ✅ Exportación de resultados

---

## **Deployment en Railway**

### **Paso 1: Preparar archivos actualizados v4.3.1**
```
📁 proyecto/
├── 📄 main.py                     # API principal (v4.3.1)
├── 📄 dashboard_compatible.html   # Dashboard frontend (MEJORADO)
├── 📄 requirements.txt           # Dependencias Python
├── 📄 runtime.txt                # Versión Python
├── 📄 railway.json               # Configuración Railway
├── 📄 start.sh                   # Script de inicio
├── 📄 test.py                    # Suite de testing (v4.3.1)
├── 📄 .gitignore                 # Archivos a ignorar
└── 📄 README.md                  # Documentación (ACTUALIZADA)
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

# 4. Ejecutar tests v4.3.1
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

## 📋 **Estructura de Datos**

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

## 🔗 **Endpoints de la API v4.3.1**

### 🏠 **Principales:**
- `GET /` - Dashboard principal (MEJORADO)
- `GET /health` - Estado del sistema (v4.3.1)
- `GET /metadatos` - Información del dataset
- `POST /upload/excel` - Carga de archivos

### 📊 **Análisis (MEJORADOS):**
- `GET /caracterizacion` - Estadísticas por territorio (filtros corregidos)
- `GET /analisis/asociacion` - Correlaciones entre indicadores
- `GET /analisis/theil` - Índice de desigualdad territorial (TODAS las UPZ)
- `GET /datos/series` - Series temporales

### 🗺️ **Geografía (MEJORADO):**
- `GET /geografia/upz_por_localidad` - UPZ por localidad (CORREGIDO)

### 📚 **Documentación:**
- `GET /docs` - Swagger UI (documentación interactiva)
- `GET /openapi.json` - Schema OpenAPI

---

## 🧪 **Testing v4.3.1**

```bash
# Ejecutar suite completa de tests (MEJORADA)
python test.py

# Los tests ahora incluyen:
✅ Importación de módulos
✅ Conectividad de base de datos  
✅ Funcionamiento de endpoints
✅ 🆕 Filtrado UPZ por localidad
✅ 🆕 Índice Theil con todas las UPZ
✅ 🆕 Estructura del dashboard mejorado
✅ 🆕 Filtros de caracterización corregidos
✅ Validación de datos
✅ Detección de cohortes
✅ Limpieza de datos
```

---

## 📊 **Dashboard Interactivo v4.3.1**

### **Mejoras en la Interfaz:**
- 🎨 **Diseño responsive mejorado**: Mejor experiencia móvil
- 📱 **Gráficos adaptativos**: Se ajustan automáticamente al dispositivo
- 🔄 **Filtros corregidos**: Lógica Localidad/UPZ funciona perfectamente
- 📈 **Gráfico Theil scrolleable**: Muestra todas las UPZ con barra deslizante

### **Tabs en Orden Optimizado:**
1. **Caracterización**: Estadísticas descriptivas (PRIMERO)
2. **Series**: Evolución temporal (SEGUNDO)
3. **Asociación**: Correlaciones entre variables  
4. **Desigualdad**: Índice de Theil (ÚLTIMO)

### **Flujo de Uso Mejorado:**
1. Seleccionar **Nivel** (Localidad/UPZ)
2. El campo territorio se actualiza automáticamente
3. **"Todas"** muestra todas las unidades del nivel seleccionado
4. Al cambiar a UPZ, puede filtrar por localidad específica

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

## 🔧 **Funcionalidades Especiales v4.3.1**

### 🎯 **Detección automática de cohortes:**
El sistema identifica automáticamente grupos etarios:
- **10-14 años**: Niñas 
- **15-19 años**: Adolescentes

### 📈 **Análisis estadísticos mejorados:**
- Estadísticas descriptivas completas
- Correlaciones de Pearson y Spearman
- **Índice de Theil completo**: Incluye TODAS las UPZ (no top 10)
- Coeficientes de variación

### **🛡️ Validación de datos:**
- Limpieza automática de valores nulos
- Normalización de texto
- Validación de tipos de datos
- Manejo de errores robusto

### **🎨 Mejoras de UX:**
- Tooltips completos con nombres de UPZ
- Gráficos responsive con scroll horizontal
- Labels dinámicos según el nivel seleccionado
- Manejo inteligente de textos largos

---

## 🐛 **Solución de Problemas v4.3.1**

### **Error de filtros UPZ:**
```javascript
// CORREGIDO en v4.3.1
// Los filtros ahora funcionan correctamente:
// Localidad → UPZ se actualiza automáticamente
// "Todas" funciona en ambos niveles
```

### **Error de dependencias:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### **Error de base de datos:**
- El sistema usa SQLite como fallback automático
- Para PostgreSQL, verifica la variable `DATABASE_URL`

### **Gráfico Theil no muestra todas las UPZ:**
```javascript
// CORREGIDO en v4.3.1
// Ahora el gráfico es scrolleable y muestra TODAS las UPZ
// No se limita a top 10
```

### **Tests fallando:**
```bash
# Verificar configuración v4.3.1
python -c "from main import app; print(f'✅ OK - Version: {app.version}')"

# Ejecutar tests específicos v4.3.1
python test.py
```

---

## 🤝 **Contribuciones**

El proyecto está diseñado para ser extensible:

1. **Nuevos análisis**: Agregar endpoints en `main.py`
2. **Nuevas visualizaciones**: Modificar `dashboard_compatible.html`
3. **Nuevos tests**: Agregar funciones a `test.py`

### **Estructura de commits v4.3.1:**
```
feat: 🆕 Filtros Localidad/UPZ corregidos
feat: 📈 Índice Theil con todas las UPZ  
feat: 🎨 Dashboard responsive mejorado
feat: 📊 Orden pestañas optimizado
fix: 🐛 Endpoint geografía corregido
test: 🧪 Test suite v4.3.1 actualizado
docs: 📚 README v4.3.1 actualizado
```

---

## 📜 **Licencia**

Este proyecto está desarrollado para análisis de políticas públicas en Bogotá D.C.

---

## 🔗 **Enlaces Útiles**

- [Railway Documentation](https://docs.railway.app/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Chart.js Documentation](https://www.chartjs.org/docs/)

---

## 📞 **Soporte**

Para reportar issues o solicitar nuevas funcionalidades, utiliza el sistema de issues del repositorio.

### **Issues Reportados y Solucionados v4.3.1:**
- ✅ **#001**: Filtros Localidad/UPZ no funcionaban → **SOLUCIONADO**
- ✅ **#002**: Índice Theil solo mostraba top 10 → **SOLUCIONADO**
- ✅ **#003**: Orden de pestañas confuso → **SOLUCIONADO**
- ✅ **#004**: Dashboard no responsive en móviles → **SOLUCIONADO**

---

## 📋 **Changelog v4.3.1**

### **✨ Features:**
- Filtros Localidad/UPZ completamente corregidos
- Gráfico Theil scrolleable con todas las UPZ
- Dashboard responsive optimizado
- Orden de pestañas mejorado (Caracterización → Series → Asociación → Desigualdad)

### **🐛 Bug Fixes:**
- Endpoint `/geografia/upz_por_localidad` corregido
- Lógica de filtros territoriales reparada
- Manejo de nombres de UPZ largos mejorado

### **🚀 Performance:**
- Gráficos optimizados para móviles
- Mejor manejo de memoria en gráficos grandes
- Tooltips más informativos

### **📚 Documentation:**
- Tests específicos v4.3.1
- README actualizado con nuevas funcionalidades
- Documentación de endpoints mejorada

---

**🚀 Versión 4.3.1** - Sistema optimizado con filtros corregidos, índice Theil completo y dashboard responsive mejorado.

---

## 👥 **Equipo de Desarrollo**

### **Actualizaciones del Equipo v4.3.1**
- 🆕 Nueva función de filtros territoriales añadida y corregida
- 🐛 Corregidos bugs en la lógica de UPZ por localidad  
- 📊 Índice de Theil expandido para mostrar todas las UPZ
- 📱 Dashboard completamente responsive
- 📚 Documentación y tests actualizados

*Última actualización: v4.3.1 - Mejoras implementadas y funcionales* 

---

> **Nota**: Esta versión soluciona todos los problemas reportados de filtros y visualización. El sistema está listo para producción con las mejoras solicitadas.
