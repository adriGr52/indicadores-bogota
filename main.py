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

# Configurar logging MÁS DETALLADO
logging.basicConfig(
    level=logging.DEBUG,  # Cambiado a DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Configuración de base de datos
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///fecundidad_temprana.db")
logger.info(f"Using database: {DATABASE_URL[:50]}...")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Configuración del engine con fallback
try:
    if "postgresql://" in DATABASE_URL:
        engine = create_engine(DATABASE_URL, echo=True, pool_pre_ping=True, pool_recycle=300, connect_args={"connect_timeout": 10})
    else:
        engine = create_engine(DATABASE_URL, echo=True)  # Echo=True para debugging
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Database engine created successfully")
except Exception as e:
    logger.error(f"Database engine creation failed: {e}")
    engine = create_engine("sqlite:///fallback.db", echo=True)
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
    title="Exploración Determinantes Fecundidad Temprana - DEBUG v4.3.1",
    description="Análisis integral con debugging completo",
    version="4.3.1",
    debug=True  # Modo debug habilitado
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Crear tablas al startup
@app.on_event("startup")
async def startup_event():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified successfully")
        
        # Log initial data count
        db = SessionLocal()
        try:
            count = db.query(IndicadorFecundidad).count()
            logger.info(f"Current records in database: {count}")
        finally:
            db.close()
            
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
            
            # Información adicional de debugging
            indicadores_count = db.query(IndicadorFecundidad.indicador_nombre).distinct().count()
            localidades_count = db.query(IndicadorFecundidad.nombre_localidad).distinct().count()
            
            logger.info(f"Health check: {count} records, {indicadores_count} indicators, {localidades_count} localidades")
            
        except Exception as e:
            db_status = f"error: {str(e)[:50]}"
            count = 0
            indicadores_count = 0
            localidades_count = 0
            logger.error(f"Health check database error: {e}")
        finally:
            db.close()
        
        return {
            "status": "healthy", 
            "version": "4.3.1", 
            "database": db_status, 
            "registros": count,
            "indicadores": indicadores_count,
            "localidades": localidades_count,
            "timestamp": datetime.now().isoformat(),
            "debug_mode": True
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": str(e)})

@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("dashboard_compatible.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        logger.warning("Dashboard HTML file not found")
        return HTMLResponse("""
        <html><body style="font-family: Arial; padding: 2rem; text-align: center;">
            <h1>🛠️ Fecundidad Temprana API v4.3.1 - DEBUG MODE</h1>
            <p><a href="/docs" style="color: #2563eb;">📚 Ver Documentación API</a></p>
            <p><a href="/health" style="color: #2563eb;">💚 Health Check</a></p>
            <p><a href="/debug/database" style="color: #f59e0b;">🔍 Debug Database</a></p>
        </body></html>
        """)

# NUEVOS ENDPOINTS DE DEBUGGING
@app.get("/debug/database")
async def debug_database(db: Session = Depends(get_db)):
    """Endpoint de debugging para inspeccionar la base de datos"""
    try:
        logger.info("Debug database endpoint called")
        
        # Información básica
        total_records = db.query(IndicadorFecundidad).count()
        
        # Muestras de datos
        sample_records = db.query(IndicadorFecundidad).limit(5).all()
        
        # Estadísticas por columna
        distinct_indicators = db.query(IndicadorFecundidad.indicador_nombre).distinct().count()
        distinct_localidades = db.query(IndicadorFecundidad.nombre_localidad).distinct().count()
        distinct_upz = db.query(IndicadorFecundidad.nombre_upz).filter(
            IndicadorFecundidad.nombre_upz.isnot(None)
        ).distinct().count()
        
        # Primeros 10 indicadores
        sample_indicators = [r[0] for r in db.query(IndicadorFecundidad.indicador_nombre).distinct().limit(10).all()]
        
        # Primeras 10 localidades
        sample_localidades = [r[0] for r in db.query(IndicadorFecundidad.nombre_localidad).distinct().limit(10).all()]
        
        return {
            "database_status": "accessible",
            "total_records": total_records,
            "statistics": {
                "distinct_indicators": distinct_indicators,
                "distinct_localidades": distinct_localidades,
                "distinct_upz": distinct_upz
            },
            "sample_data": [
                {
                    "id": r.id,
                    "indicador_nombre": r.indicador_nombre,
                    "valor": r.valor,
                    "unidad_medida": r.unidad_medida,
                    "nombre_localidad": r.nombre_localidad,
                    "nombre_upz": r.nombre_upz,
                    "año_inicio": r.año_inicio
                } for r in sample_records
            ],
            "sample_indicators": sample_indicators,
            "sample_localidades": sample_localidades
        }
    except Exception as e:
        logger.error(f"Debug database error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e), "database_accessible": False})

@app.get("/debug/raw_data")
async def debug_raw_data(limit: int = Query(50, ge=1, le=1000), db: Session = Depends(get_db)):
    """Endpoint para obtener datos raw de la base de datos"""
    try:
        logger.info(f"Debug raw data endpoint called with limit {limit}")
        
        records = db.query(IndicadorFecundidad).limit(limit).all()
        
        return {
            "total_requested": limit,
            "total_returned": len(records),
            "data": [
                {
                    "id": r.id,
                    "indicador_nombre": r.indicador_nombre,
                    "valor": r.valor,
                    "unidad_medida": r.unidad_medida,
                    "nivel_territorial": r.nivel_territorial,
                    "nombre_localidad": r.nombre_localidad,
                    "nombre_upz": r.nombre_upz,
                    "año_inicio": r.año_inicio,
                    "grupo_etario_asociado": r.grupo_etario_asociado,
                    "fecha_carga": r.fecha_carga.isoformat() if r.fecha_carga else None
                } for r in records
            ]
        }
    except Exception as e:
        logger.error(f"Debug raw data error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/upload/excel")
async def upload_excel(file: UploadFile = File(...), db: Session = Depends(get_db)):
    logger.info(f"Upload started for file: {file.filename}")
    
    if not file.filename.endswith(('.xlsx', '.xls')):
        logger.error(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Use archivos .xlsx o .xls")
    
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents), sheet_name=0)
        logger.info(f"Excel file loaded: {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Columns found: {list(df.columns)}")
        
        # Verificar columnas requeridas
        required_columns = ['Indicador_Nombre', 'Valor', 'Unidad_Medida']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise HTTPException(status_code=400, detail=f"Columnas faltantes: {missing_columns}")
        
        # Log muestra de datos
        logger.info("Sample data from Excel:")
        for idx, row in df.head(3).iterrows():
            logger.info(f"Row {idx}: Indicador='{row.get('Indicador_Nombre')}', Valor='{row.get('Valor')}', Localidad='{row.get('Nombre Localidad')}'")
        
        # Limpiar tabla y cargar datos
        deleted_count = db.query(IndicadorFecundidad).count()
        logger.info(f"Deleting {deleted_count} existing records")
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
                año_inicio = clean_int(row.get('Año_Inicio')) if 'Año_Inicio' in row else None
                
                rec = IndicadorFecundidad(
                    origen_archivo=clean_str(row.get('origen_archivo')),
                    archivo_hash=clean_str(row.get('archivo_hash')),
                    indicador_nombre=limpiar_texto(indicador_nombre),
                    dimension=clean_str(row.get('Dimensión')),
                    unidad_medida=unidad_medida,
                    tipo_medida=clean_str(row.get('Tipo_Medida')),
                    valor=valor,
                    nivel_territorial=nivel_territorial,
                    id_localidad=clean_int(row.get('ID Localidad')),
                    nombre_localidad=limpiar_texto(nombre_localidad),
                    id_upz=clean_int(row.get('ID_UPZ')),
                    nombre_upz=limpiar_texto(nombre_upz) if nombre_upz else None,
                    area_geografica=clean_str(row.get('Área Geográfica')),
                    año_inicio=año_inicio,
                    periodicidad=clean_str(row.get('Periodicidad')),
                    poblacion_base=clean_str(row.get('Poblacion Base')),
                    semaforo=clean_str(row.get('Semaforo')),
                    grupo_etario_asociado=clean_str(row.get('Grupo Etario Asociado')),
                    sexo=clean_str(row.get('Sexo')),
                    tipo_unidad=clean_str(row.get('Tipo de Unidad')),
                    observacion=clean_str(row.get('Tipo de Unidad Observación')),
                    fuente=clean_str(row.get('Fuente')),
                    url_fuente=clean_str(row.get('URL_Fuente (Opcional)'))
                )
                db.add(rec)
                registros += 1
                
                if registros % 500 == 0:
                    db.commit()
                    logger.info(f"Committed {registros} records...")
                
            except Exception as e:
                errores += 1
                if errores <= 5:  # Log only first 5 errors
                    logger.warning(f"Error in row {idx}: {e}")
        
        db.commit()
        
        # Verificar datos cargados
        final_count = db.query(IndicadorFecundidad).count()
        indicadores_unicos = db.query(IndicadorFecundidad.indicador_nombre).distinct().count()
        localidades_unicas = db.query(IndicadorFecundidad.nombre_localidad).distinct().count()
        upz_unicas = db.query(IndicadorFecundidad.nombre_upz).filter(
            IndicadorFecundidad.nombre_upz.isnot(None)
        ).distinct().count()
        
        logger.info(f"Upload completed: {registros} records loaded, {final_count} total in DB")
        logger.info(f"Unique indicators: {indicadores_unicos}, localidades: {localidades_unicas}, UPZ: {upz_unicas}")
        
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
                "localidades_unicas": localidades_unicas,
                "upz_unicas": upz_unicas,
                "verificacion_final": final_count
            },
            "debug_info": {
                "columns_found": list(df.columns),
                "sample_indicators": df['Indicador_Nombre'].dropna().head(5).tolist(),
                "sample_localidades": df['Nombre Localidad'].dropna().head(5).tolist() if 'Nombre Localidad' in df.columns else []
            }
        }
        
    except Exception as e:
        logger.exception("Error procesando archivo Excel")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {str(e)}")

@app.get("/metadatos")
async def metadatos(db: Session = Depends(get_db)):
    logger.info("Metadatos endpoint called")
    
    try:
        total = db.query(IndicadorFecundidad).count()
        logger.info(f"Total records: {total}")
        
        if total == 0:
            logger.warning("No records found in database")
            return {
                "resumen": {"total_registros": 0, "total_indicadores": 0, "localidades": 0, "upz": 0},
                "indicadores": {"todos": []},
                "geografia": {"localidades": [], "upz": []},
                "temporal": {"años": [], "cohortes": sorted(list(COHORTES_VALIDAS))},
                "debug_info": {"message": "No data in database", "total_records": 0}
            }
        
        # Indicadores
        todos_indicadores_raw = [r[0] for r in db.query(IndicadorFecundidad.indicador_nombre).distinct().all()]
        todos_indicadores = [limpiar_texto(ind) for ind in todos_indicadores_raw if ind]
        logger.info(f"Found {len(todos_indicadores)} unique indicators")
        
        # Geografía
        localidades_raw = [r[0] for r in db.query(IndicadorFecundidad.nombre_localidad).distinct().all()]
        localidades = [limpiar_texto(loc) for loc in localidades_raw if loc and loc != 'SIN LOCALIDAD']
        logger.info(f"Found {len(localidades)} localidades")
        
        upzs_raw = [r[0] for r in db.query(IndicadorFecundidad.nombre_upz).filter(
            IndicadorFecundidad.nombre_upz.isnot(None)
        ).distinct().all()]
        upzs = [limpiar_texto(upz) for upz in upzs_raw if upz and upz not in ['ND', 'NO_DATA', 'SIN UPZ']]
        logger.info(f"Found {len(upzs)} UPZ")
        
        # Temporal
        años = sorted([r[0] for r in db.query(IndicadorFecundidad.año_inicio).filter(
            IndicadorFecundidad.año_inicio.isnot(None)
        ).distinct().all()])
        logger.info(f"Found years: {años}")
        
        result = {
            "resumen": {
                "total_registros": total,
                "total_indicadores": len(todos_indicadores),
                "localidades": len(localidades),
                "upz": len(upzs),
                "rango_años": {"min": min(años) if años else None, "max": max(años) if años else None}
            },
            "indicadores": {"todos": sorted(todos_indicadores)},
            "geografia": {"localidades": sorted(localidades), "upz": sorted(upzs)},
            "temporal": {"años": años, "cohortes": sorted(list(COHORTES_VALIDAS))},
            "debug_info": {
                "raw_indicators_count": len(todos_indicadores_raw),
                "raw_localidades_count": len(localidades_raw),
                "raw_upz_count": len(upzs_raw),
                "sample_indicators": todos_indicadores[:5],
                "sample_localidades": localidades[:5]
            }
        }
        
        logger.info("Metadatos response prepared successfully")
        return result
        
    except Exception as e:
        logger.exception("Error in metadatos endpoint")
        raise HTTPException(status_code=500, detail=f"Error getting metadata: {str(e)}")

@app.get("/geografia/upz_por_localidad")
async def upz_por_localidad(localidad: str = Query(...), db: Session = Depends(get_db)):
    logger.info(f"UPZ por localidad called for: {localidad}")
    
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
        
        logger.info(f"Found {len(upz_list)} UPZ for localidad {localidad}")
        
        return {"localidad": localidad, "upz": upz_list, "total": len(upz_list)}
    except Exception as e:
        logger.error(f"Error getting UPZ for localidad {localidad}: {e}")
        return {"localidad": localidad, "upz": [], "total": 0, "error": str(e)}

@app.get("/debug/columns")
async def debug_columns():
    """Endpoint de debug mejorado"""
    return {
        "columnas_requeridas": ["Indicador_Nombre", "Valor", "Unidad_Medida"],
        "columnas_opcionales": [
            "origen_archivo", "archivo_hash", "Dimensión", "Tipo_Medida",
            "Nivel_Territorial", "ID Localidad", "Nombre Localidad", 
            "ID_UPZ", "Nombre_UPZ", "Área Geográfica", "Año_Inicio",
            "Periodicidad", "Poblacion Base", "Semaforo", "Grupo Etario Asociado",
            "Sexo", "Tipo de Unidad", "Tipo de Unidad Observación", 
            "Fuente", "URL_Fuente (Opcional)"
        ],
        "ejemplo_estructura": {
            "Indicador_Nombre": "Tasa Específica de Fecundidad en niñas de 10 a 14 años",
            "Valor": 1.2,
            "Unidad_Medida": "Por cada 1000 niñas",
            "Nombre Localidad": "Usaquén",
            "Año_Inicio": 2020
        },
        "mejoras_v431": [
            "Dashboard con debugging completo",
            "Logging detallado de todas las operaciones",
            "Endpoints de debugging adicionales",
            "Verificación paso a paso de datos"
        ]
    }

# Endpoints de análisis simplificados para debugging
@app.get("/caracterizacion")
async def caracterizacion(
    indicador: str = Query(...),
    nivel: str = Query("UPZ"),
    localidad: Optional[str] = Query(None),
    upz: Optional[str] = Query(None),
    año: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    logger.info(f"Caracterizacion called: indicador='{indicador}', nivel='{nivel}', localidad='{localidad}', upz='{upz}'")
    
    try:
        # Verificar que el indicador existe
        indicator_exists = db.query(IndicadorFecundidad).filter(
            IndicadorFecundidad.indicador_nombre == indicador
        ).first()
        
        if not indicator_exists:
            logger.warning(f"Indicator '{indicador}' not found in database")
            return {"mensaje": f"Indicador '{indicador}' no encontrado en la base de datos"}
        
        q = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador)
        
        if año is not None:
            q = q.filter(IndicadorFecundidad.año_inicio == año)
        if localidad:
            q = q.filter(IndicadorFecundidad.nombre_localidad == localidad)
        if upz:
            q = q.filter(IndicadorFecundidad.nombre_upz == upz)
        
        rows = q.all()
        logger.info(f"Query returned {len(rows)} rows")
        
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
        
        logger.info(f"Grouped data into {len(grupos)} groups: {list(grupos.keys())}")
        
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
        logger.info(f"Calculated statistics for {len(datos)} groups")
        
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
            "datos": datos,
            "debug_info": {
                "rows_found": len(rows),
                "groups_created": len(grupos),
                "filters_applied": {"año": año, "localidad": localidad, "upz": upz}
            }
        }
    except Exception as e:
        logger.exception(f"Error in caracterizacion: {e}")
        raise HTTPException(status_code=500, detail=f"Error en caracterización: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting DEBUG server v4.3.1 on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="debug", access_log=True, reload=True)
