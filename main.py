from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Index, func, text
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import io, os, logging, re
from datetime import datetime
from scipy import stats

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración de base de datos
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///fecundidad_temprana.db")
logger.info(f"Using database: {DATABASE_URL[:50]}...")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Configuración del engine con fallback
try:
    if "postgresql://" in DATABASE_URL:
        engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True, pool_recycle=300, connect_args={"connect_timeout": 10})
    else:
        engine = create_engine(DATABASE_URL, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Database engine created successfully")
except Exception as e:
    logger.error(f"Database engine creation failed: {e}")
    engine = create_engine("sqlite:///fallback.db", echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.warning("Using fallback SQLite database")

Base = declarative_base()

class IndicadorFecundidad(Base):
    __tablename__ = "indicadores_fecundidad"
    id = Column(Integer, primary_key=True, index=True)
    origen_archivo = Column(String)
    archivo_hash = Column(String, index=True)
    indicador_nombre = Column(String, index=True, nullable=False)
    dimension = Column(String)
    unidad_medida = Column(String, nullable=False)
    tipo_medida = Column(String)
    valor = Column(Float, nullable=False)
    nivel_territorial = Column(String, index=True, nullable=False)
    id_localidad = Column(Integer, index=True, nullable=True)
    nombre_localidad = Column(String, index=True, nullable=False)
    id_upz = Column(Integer, index=True, nullable=True)
    nombre_upz = Column(String, index=True, nullable=True)
    area_geografica = Column(String)
    año_inicio = Column(Integer, index=True)
    periodicidad = Column(String)
    poblacion_base = Column(String)
    semaforo = Column(String)
    grupo_etario_asociado = Column(String, index=True)
    sexo = Column(String)
    tipo_unidad = Column(String)
    observacion = Column(String)
    fuente = Column(String)
    url_fuente = Column(String)
    fecha_carga = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index('idx_localidad_indicador', 'nombre_localidad', 'indicador_nombre'),
        Index('idx_upz_grupo', 'nombre_upz', 'grupo_etario_asociado'),
        Index('idx_nivel_año', 'nivel_territorial', 'año_inicio'),
    )

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Crear la aplicación FastAPI
app = FastAPI(
    title="Exploración Determinantes Fecundidad Temprana - Bogotá D.C.",
    description="Análisis integral por UPZ v4.3.1 con filtros corregidos",
    version="4.3.1"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Crear tablas al startup
@app.on_event("startup")
async def startup_event():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified successfully")
    except Exception as e:
        logger.warning(f"Could not create tables on startup: {e}")

# Funciones auxiliares
def limpiar_texto(texto: str) -> str:
    if not texto: return texto
    texto = texto.strip()
    return re.sub(r'\s+', ' ', texto)

def calcular_indice_theil(valores: List[float], poblaciones: Optional[List[float]] = None) -> float:
    if not valores or len(valores) < 2: return 0.0
    valores = np.array(valores)
    if poblaciones is not None:
        poblaciones = np.array(poblaciones)
        if len(poblaciones) != len(valores): poblaciones = None
    if poblaciones is None: poblaciones = np.ones(len(valores))
    mask = (valores > 0) & (poblaciones > 0) & np.isfinite(valores) & np.isfinite(poblaciones)
    if not mask.any(): return 0.0
    valores = valores[mask]
    poblaciones = poblaciones[mask]
    pesos = poblaciones / poblaciones.sum()
    media = np.sum(pesos * valores)
    if media <= 0: return 0.0
    ratios = valores / media
    ratios = np.maximum(ratios, 1e-10)
    theil = np.sum(pesos * ratios * np.log(ratios))
    return float(theil)

COHORTES_VALIDAS = {"10-14", "15-19"}

def extraer_grupo_edad(indicador_nombre: Optional[str], grupo_etario: Optional[str]) -> Optional[str]:
    txt = f"{(indicador_nombre or '').lower()} {(grupo_etario or '').lower()}"
    if re.search(r"10\s*[-aá]\s*14", txt) or re.search(r"10\D*14", txt): return "10-14"
    if re.search(r"15\s*[-aá]\s*19", txt) or re.search(r"15\D*19", txt): return "15-19"
    if "niñas de 10 a 14" in txt or "10 a 14 años" in txt: return "10-14"
    if "mujeres de 15 a 19" in txt or "15 a 19 años" in txt: return "15-19"
    return None

def is_nan_like(x) -> bool:
    if x is None: return True
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)): return True
    s = str(x).strip().lower()
    return s in {"", "nan", "nd", "no_data", "none", "null"}

def clean_str(x, default=None):
    if is_nan_like(x): return default
    resultado = str(x).strip()
    return limpiar_texto(resultado) if resultado else default

def clean_int(x, default=None):
    if is_nan_like(x): return default
    try: return int(float(x))
    except Exception: return default

def clean_float(x, allow_none=True, default=None):
    if is_nan_like(x): return None if allow_none else (default if default is not None else 0.0)
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v): return None if allow_none else (default if default is not None else 0.0)
        return v
    except Exception: return None if allow_none else (default if default is not None else 0.0)

# Rutas principales
@app.get("/health")
async def health():
    try:
        db = SessionLocal()
        try:
            result = db.execute(text("SELECT 1")).scalar()
            db_status = "connected" if result == 1 else "error"
            count = db.query(IndicadorFecundidad).count()
        except Exception as e:
            db_status = f"error: {str(e)[:50]}"
            count = 0
        finally:
            db.close()
        return {"status": "healthy", "version": "4.3.1", "database": db_status, "registros": count, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": str(e)})

@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("dashboard_compatible.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("""
        <html><body style="font-family: Arial; padding: 2rem; text-align: center;">
            <h1>🏛️ Fecundidad Temprana API v4.3.1</h1>
            <p><a href="/docs" style="color: #2563eb;">📚 Ver Documentación API</a></p>
        </body></html>
        """)

@app.post("/upload/excel")
async def upload_excel(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Use archivos .xlsx o .xls")
    
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents), sheet_name=0)
        logger.info(f"Archivo cargado: {file.filename}, filas: {len(df)}")
        
        # Mapeo de columnas
        column_mapping = {'Dimensión': 'dimension', 'Área Geográfica': 'area_geografica', 'Tipo de Unidad Observación': 'observacion', 'URL_Fuente (Opcional)': 'url_fuente'}
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        
        # Verificar columnas requeridas
        required_columns = ['Indicador_Nombre', 'Valor', 'Unidad_Medida']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Columnas faltantes: {missing_columns}")
        
        # Limpiar tabla y cargar datos
        deleted_count = db.query(IndicadorFecundidad).count()
        db.query(IndicadorFecundidad).delete()
        
        registros, errores, omitidos_sin_valor = 0, 0, 0
        for idx, row in df.iterrows():
            try:
                indicador_nombre = clean_str(row.get('Indicador_Nombre'))
                if not indicador_nombre:
                    omitidos_sin_valor += 1
                    continue
                
                valor = clean_float(row.get('Valor'), allow_none=True)
                if valor is None:
                    omitidos_sin_valor += 1
                    continue
                
                unidad_medida = clean_str(row.get('Unidad_Medida'), default='N/A') or 'N/A'
                nivel_territorial = (clean_str(row.get('Nivel_Territorial')) or 'LOCALIDAD').upper()
                nombre_localidad = clean_str(row.get('Nombre Localidad')) or 'SIN LOCALIDAD'
                nombre_upz = clean_str(row.get('Nombre_UPZ'))
                if nombre_upz and nombre_upz.upper() in ['ND', 'NO_DATA']:
                    nombre_upz = None
                año_inicio = None
                if 'Año_Inicio' in row:
                    año_inicio = clean_int(row.get('Año_Inicio'))
                
                rec = IndicadorFecundidad(
                    origen_archivo=clean_str(row.get('origen_archivo')),
                    archivo_hash=clean_str(row.get('archivo_hash')),
                    indicador_nombre=limpiar_texto(indicador_nombre),
                    dimension=clean_str(row.get('dimension')),
                    unidad_medida=unidad_medida,
                    tipo_medida=clean_str(row.get('Tipo_Medida')),
                    valor=valor,
                    nivel_territorial=nivel_territorial,
                    id_localidad=clean_int(row.get('ID Localidad')),
                    nombre_localidad=limpiar_texto(nombre_localidad),
                    id_upz=clean_int(row.get('ID_UPZ')),
                    nombre_upz=limpiar_texto(nombre_upz) if nombre_upz else None,
                    area_geografica=clean_str(row.get('area_geografica')),
                    año_inicio=año_inicio,
                    periodicidad=clean_str(row.get('Periodicidad')),
                    poblacion_base=clean_str(row.get('Poblacion Base')),
                    semaforo=clean_str(row.get('Semaforo')),
                    grupo_etario_asociado=clean_str(row.get('Grupo Etario Asociado')),
                    sexo=clean_str(row.get('Sexo')),
                    tipo_unidad=clean_str(row.get('Tipo de Unidad')),
                    observacion=clean_str(row.get('observacion')),
                    fuente=clean_str(row.get('Fuente')),
                    url_fuente=clean_str(row.get('url_fuente'))
                )
                db.add(rec)
                registros += 1
                
                if registros % 1000 == 0:
                    db.commit()
                    logger.info(f"Procesados {registros} registros...")
                
            except Exception as e:
                errores += 1
                logger.warning(f"Error en fila {idx}: {e}")
        
        db.commit()
        indicadores_unicos = db.query(IndicadorFecundidad.indicador_nombre).distinct().count()
        localidades_unicas = db.query(IndicadorFecundidad.nombre_localidad).distinct().count()
        
        return {
            "status": "success",
            "mensaje": "Carga completada exitosamente",
            "archivo": file.filename,
            "estadisticas": {
                "registros_anteriores": deleted_count,
                "registros_cargados": registros,
                "filas_omitidas_sin_valor": omitidos_sin_valor,
                "errores": errores,
                "filas_procesadas": len(df),
                "indicadores_unicos": indicadores_unicos,
                "localidades_unicas": localidades_unicas
            }
        }
        
    except Exception as e:
        logger.exception("Error procesando archivo Excel")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {str(e)}")

@app.get("/metadatos")
async def metadatos(db: Session = Depends(get_db)):
    total = db.query(IndicadorFecundidad).count()
    
    # Indicadores
    todos_indicadores_raw = [r[0] for r in db.query(IndicadorFecundidad.indicador_nombre).distinct().all()]
    todos_indicadores = [limpiar_texto(ind) for ind in todos_indicadores_raw if ind]
    
    # Geografía
    localidades_raw = [r[0] for r in db.query(IndicadorFecundidad.nombre_localidad).distinct().all()]
    localidades = [limpiar_texto(loc) for loc in localidades_raw if loc and loc != 'SIN LOCALIDAD']
    
    upzs_raw = [r[0] for r in db.query(IndicadorFecundidad.nombre_upz).filter(IndicadorFecundidad.nombre_upz.isnot(None)).distinct().all()]
    upzs = [limpiar_texto(upz) for upz in upzs_raw if upz and upz not in ['ND', 'NO_DATA', 'SIN UPZ']]
    
    # Temporal
    años = sorted([r[0] for r in db.query(IndicadorFecundidad.año_inicio).filter(IndicadorFecundidad.año_inicio.isnot(None)).distinct().all()])
    
    return {
        "resumen": {
            "total_registros": total,
            "total_indicadores": len(todos_indicadores),
            "localidades": len(localidades),
            "upz": len(upzs),
            "rango_años": {"min": min(años) if años else None, "max": max(años) if años else None}
        },
        "indicadores": {"todos": sorted(todos_indicadores)},
        "geografia": {"localidades": sorted(localidades), "upz": sorted(upzs)},
        "temporal": {"años": años, "cohortes": sorted(list(COHORTES_VALIDAS))}
    }

@app.get("/geografia/upz_por_localidad")
async def upz_por_localidad(localidad: str = Query(...), db: Session = Depends(get_db)):
    try:
        upzs = db.query(IndicadorFecundidad.nombre_upz).filter(
            IndicadorFecundidad.nombre_localidad == localidad,
            IndicadorFecundidad.nombre_upz.isnot(None),
            IndicadorFecundidad.nombre_upz != 'ND',
            IndicadorFecundidad.nombre_upz != 'NO_DATA',
            IndicadorFecundidad.nombre_upz != 'SIN UPZ'
        ).distinct().all()
        
        upz_list = [limpiar_texto(upz[0]) for upz in upzs if upz[0]]
        upz_list = sorted(list(set(upz_list)))
        
        return {"localidad": localidad, "upz": upz_list, "total": len(upz_list)}
    except Exception as e:
        logger.error(f"Error getting UPZ for localidad {localidad}: {e}")
        return {"localidad": localidad, "upz": [], "total": 0}

# Endpoints de análisis con filtros corregidos
@app.get("/caracterizacion")
async def caracterizacion(
    indicador: str = Query(...),
    nivel: str = Query("UPZ"),
    localidad: Optional[str] = Query(None),
    upz: Optional[str] = Query(None),
    año: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    q = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador)
    
    if año is not None:
        q = q.filter(IndicadorFecundidad.año_inicio == año)
    if localidad:
        q = q.filter(IndicadorFecundidad.nombre_localidad == localidad)
    if upz:
        q = q.filter(IndicadorFecundidad.nombre_upz == upz)
    
    rows = q.all()
    if not rows:
        return {"mensaje": "Sin datos para los filtros especificados"}
    
    # Agrupar por UPZ
    grupos = {}
    unidad_medida = rows[0].unidad_medida if rows else "N/A"
    
    for r in rows:
        if nivel.upper() == "LOCALIDAD":
            k = limpiar_texto(r.nombre_localidad) if r.nombre_localidad else "SIN LOCALIDAD"
        else:
            upz_name = r.nombre_upz or "SIN UPZ"
            k = limpiar_texto(upz_name) if upz_name not in ['ND', 'NO_DATA', ''] else "SIN UPZ"
        grupos.setdefault(k, []).append(r.valor)
    
    # Calcular estadísticas
    datos = []
    for upz, valores in grupos.items():
        if not valores: continue
        arr = np.array(valores, dtype=float)
        if arr.size == 0: continue
        
        q1, mediana, q3 = np.percentile(arr, [25, 50, 75])
        promedio = float(np.mean(arr))
        std = float(np.std(arr, ddof=0))
        cv = float((std/promedio)*100) if promedio != 0 else 0.0
        
        datos.append({
            "upz": upz,
            "n": int(arr.size),
            "promedio": round(promedio, 3),
            "mediana": round(float(mediana), 3),
            "q1": round(float(q1), 3),
            "q3": round(float(q3), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "desv_estandar": round(std, 3),
            "cv_pct": round(cv, 2)
        })
    
    datos.sort(key=lambda x: x["promedio"], reverse=True)
    
    return {
        "indicador": limpiar_texto(indicador),
        "nivel": nivel.upper(),
        "año": año,
        "localidad": localidad,
        "upz": upz,
        "unidad_medida": unidad_medida,
        "total_upz": len(datos),
        "resumen": {
            "promedio_general": round(float(np.mean([d["promedio"] for d in datos])), 3) if datos else 0,
            "n_total": int(sum(d["n"] for d in datos)) if datos else 0
        },
        "datos": datos
    }

@app.get("/analisis/theil")
async def indice_theil(
    indicador: str = Query(...),
    nivel: str = Query("UPZ"),
    año: Optional[int] = Query(None),
    localidad: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    q = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador)
    
    if año is not None:
        q = q.filter(IndicadorFecundidad.año_inicio == año)
    if localidad:
        q = q.filter(IndicadorFecundidad.nombre_localidad == localidad)
    
    rows = q.all()
    if not rows:
        return {"mensaje": "Sin datos para los filtros especificados"}
    
    # Agrupar por UPZ
    grupos = {}
    unidad_medida = rows[0].unidad_medida if rows else "N/A"
    
    for r in rows:
        if nivel.upper() == "LOCALIDAD":
            k = limpiar_texto(r.nombre_localidad) if r.nombre_localidad else "SIN LOCALIDAD"
        else:
            upz_name = r.nombre_upz or "SIN UPZ"
            k = limpiar_texto(upz_name) if upz_name not in ['ND', 'NO_DATA', ''] else "SIN UPZ"
        grupos.setdefault(k, []).append(r.valor)
    
    # Calcular promedios por UPZ
    upzs = []
    valores = []
    
    for upz, vals in grupos.items():
        if vals:
            promedio = float(np.mean(vals))
            upzs.append(upz)
            valores.append(promedio)
    
    if len(valores) < 2:
        return {"mensaje": "Insuficientes UPZ para calcular el índice de Theil"}
    
    # Calcular índice de Theil
    theil = calcular_indice_theil(valores)
    
    # Estadísticas
    mean_val = float(np.mean(valores))
    std_val = float(np.std(valores))
    cv = (std_val / mean_val * 100) if mean_val != 0 else 0
    
    # TODAS las UPZ (no solo top 10)
    datos_upz = []
    for i, upz in enumerate(upzs):
        datos_upz.append({
            "upz": upz,
            "valor": round(valores[i], 3),
            "desviacion_media": round(valores[i] - mean_val, 3),
            "ratio_media": round(valores[i] / mean_val, 3) if mean_val != 0 else 0
        })
    
    datos_upz.sort(key=lambda x: x["valor"], reverse=True)
    
    return {
        "indicador": limpiar_texto(indicador),
        "nivel": nivel.upper(),
        "año": año,
        "localidad": localidad,
        "unidad_medida": unidad_medida,
        "indice_theil": round(theil, 4),
        "interpretacion": {
            "valor": round(theil, 4),
            "significado": "0 = igualdad perfecta, >0 = mayor desigualdad",
            "categoria": "Baja" if theil < 0.1 else "Moderada" if theil < 0.3 else "Alta"
        },
        "estadisticas": {
            "upz": len(upzs),
            "promedio_general": round(mean_val, 3),
            "desviacion_estandar": round(std_val, 3),
            "coeficiente_variacion": round(cv, 2),
            "min": round(min(valores), 3),
            "max": round(max(valores), 3)
        },
        "datos": datos_upz  # TODAS las UPZ
    }

@app.get("/analisis/asociacion")
async def asociacion_indicadores(
    indicador_x: str = Query(...),
    indicador_y: str = Query(...),
    nivel: str = Query("UPZ"),
    localidad: Optional[str] = Query(None),
    upz: Optional[str] = Query(None),
    año: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    qx = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador_x)
    qy = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador_y)
    
    if año is not None:
        qx = qx.filter(IndicadorFecundidad.año_inicio == año)
        qy = qy.filter(IndicadorFecundidad.año_inicio == año)
    
    if localidad:
        qx = qx.filter(IndicadorFecundidad.nombre_localidad == localidad)
        qy = qy.filter(IndicadorFecundidad.nombre_localidad == localidad)
    
    if upz:
        qx = qx.filter(IndicadorFecundidad.nombre_upz == upz)
        qy = qy.filter(IndicadorFecundidad.nombre_upz == upz)
    
    x_rows = qx.all()
    y_rows = qy.all()
    
    if not x_rows or not y_rows:
        return {"mensaje": "No se encontraron datos para los indicadores seleccionados"}
    
    # Agrupar por UPZ
    x_map = {}
    y_map = {}
    
    for r in x_rows:
        k = limpiar_texto(r.nombre_upz) if r.nombre_upz else "SIN UPZ"
        x_map.setdefault(k, []).append(r.valor)
    
    for r in y_rows:
        k = limpiar_texto(r.nombre_upz) if r.nombre_upz else "SIN UPZ"
        y_map.setdefault(k, []).append(r.valor)
    
    # UPZ comunes
    upz_comunes = set(x_map.keys()) & set(y_map.keys())
    if len(upz_comunes) < 3:
        return {"mensaje": "Insuficientes UPZ comunes para el análisis"}
    
    # Calcular promedios
    x_mean = {k: float(np.mean(v)) for k, v in x_map.items() if k in upz_comunes}
    y_mean = {k: float(np.mean(v)) for k, v in y_map.items() if k in upz_comunes}
    
    # Arrays para correlación
    x_vals = [x_mean[t] for t in upz_comunes]
    y_vals = [y_mean[t] for t in upz_comunes]
    
    if np.std(x_vals) == 0 or np.std(y_vals) == 0:
        return {"mensaje": "Una de las variables no tiene variación"}
    
    # Correlaciones
    r_p, p_p = stats.pearsonr(x_vals, y_vals)
    r_s, p_s = stats.spearmanr(x_vals, y_vals)
    
    # Categorizar correlación
    abs_r = abs(float(r_p))
    if abs_r >= 0.7: categoria = "Fuerte"
    elif abs_r >= 0.5: categoria = "Moderada"
    elif abs_r >= 0.3: categoria = "Débil"
    else: categoria = "Muy débil"
    
    # Datos para gráfico
    datos_pares = []
    for upz in upz_comunes:
        datos_pares.append({"upz": upz, "x": round(x_mean[upz], 3), "y": round(y_mean[upz], 3)})
    
    return {
        "indicador_x": limpiar_texto(indicador_x),
        "indicador_y": limpiar_texto(indicador_y),
        "nivel": nivel.upper(),
        "año": año,
        "localidad": localidad,
        "upz": upz,
        "upz_comunes": len(upz_comunes),
        "correlacion": {
            "pearson_r": round(float(r_p), 3),
            "pearson_p": round(float(p_p), 4),
            "spearman_rho": round(float(r_s), 3),
            "spearman_p": round(float(p_s), 4),
            "categoria": categoria,
            "significativa": float(p_p) < 0.05
        },
        "datos": datos_pares
    }

@app.get("/datos/series")
async def serie_temporal(
    indicador: str = Query(...),
    upz: str = Query(...),
    nivel: str = Query("UPZ"),
    db: Session = Depends(get_db)
):
    q = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador)
    
    if nivel.upper() == "LOCALIDAD":
        q = q.filter(IndicadorFecundidad.nombre_localidad == upz)
    else:
        q = q.filter(IndicadorFecundidad.nombre_upz == upz)
    
    rows = q.filter(IndicadorFecundidad.año_inicio.isnot(None)).order_by(IndicadorFecundidad.año_inicio.asc()).all()
    
    if not rows:
        return {"mensaje": "Sin datos para los filtros especificados"}
    
    # Agrupar por año
    grupos_año = {}
    unidad_medida = rows[0].unidad_medida if rows else "N/A"
    
    for r in rows:
        grupos_año.setdefault(r.año_inicio, []).append(r.valor)
    
    # Calcular serie
    serie_datos = []
    for año in sorted(grupos_año.keys()):
        valores = grupos_año[año]
        serie_datos.append({
            "año": año,
            "valor": round(float(np.mean(valores)), 3),
            "n_observaciones": len(valores),
            "min": round(float(np.min(valores)), 3),
            "max": round(float(np.max(valores)), 3)
        })
    
    return {
        "indicador": limpiar_texto(indicador),
        "nivel": nivel.upper(),
        "upz": upz,
        "unidad_medida": unidad_medida,
        "periodo": {"inicio": min(grupos_año.keys()), "fin": max(grupos_año.keys()), "años": len(grupos_año)},
        "serie": serie_datos
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Iniciando servidor v4.3.1 en puerto {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", access_log=True)
