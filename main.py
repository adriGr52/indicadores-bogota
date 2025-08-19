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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración de base de datos
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///fecundidad_temprana.db")
logger.info(f"Using database: {DATABASE_URL[:50]}...")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Configuración del engine con fallback
try:
    if "postgresql://" in DATABASE_URL:
        engine = create_engine(
            DATABASE_URL, 
            echo=False,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args={"connect_timeout": 10}
        )
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
    description="Análisis integral por territorio, periodo y cohortes para la exploración de determinantes de fecundidad temprana en Bogotá D.C.",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear tablas al startup
@app.on_event("startup")
async def startup_event():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified successfully")
    except Exception as e:
        logger.warning(f"Could not create tables on startup: {e}")

# Funciones auxiliares
def calcular_indice_theil(valores: List[float], poblaciones: Optional[List[float]] = None) -> float:
    """Calcula el índice de Theil para medir desigualdad territorial"""
    if not valores or len(valores) < 2:
        return 0.0
    
    valores = np.array(valores)
    if poblaciones is not None:
        poblaciones = np.array(poblaciones)
        if len(poblaciones) != len(valores):
            poblaciones = None
    
    if poblaciones is None:
        poblaciones = np.ones(len(valores))
    
    mask = (valores > 0) & (poblaciones > 0) & np.isfinite(valores) & np.isfinite(poblaciones)
    if not mask.any():
        return 0.0
        
    valores = valores[mask]
    poblaciones = poblaciones[mask]
    
    pesos = poblaciones / poblaciones.sum()
    media = np.sum(pesos * valores)
    
    if media <= 0:
        return 0.0
    
    ratios = valores / media
    ratios = np.maximum(ratios, 1e-10)
    theil = np.sum(pesos * ratios * np.log(ratios))
    
    return float(theil)

def get_indicadores_fecundidad(db: Session) -> List[str]:
    """Obtiene lista de indicadores relacionados con fecundidad"""
    indicadores = [r[0] for r in db.query(IndicadorFecundidad.indicador_nombre).distinct().all()]
    palabras_clave = ['fecund', 'natalidad', 'nacimiento', 'maternidad', 'embarazo']
    return [ind for ind in indicadores if any(palabra in ind.lower() for palabra in palabras_clave)]

def get_indicadores_no_fecundidad(db: Session) -> List[str]:
    """Obtiene lista de indicadores que NO son de fecundidad"""
    indicadores = [r[0] for r in db.query(IndicadorFecundidad.indicador_nombre).distinct().all()]
    palabras_clave = ['fecund', 'natalidad', 'nacimiento', 'maternidad', 'embarazo']
    return [ind for ind in indicadores if not any(palabra in ind.lower() for palabra in palabras_clave)]

COHORTES_VALIDAS = {"10-14", "15-19"}

def extraer_grupo_edad(indicador_nombre: Optional[str], grupo_etario: Optional[str]) -> Optional[str]:
    txt = f"{(indicador_nombre or '').lower()} {(grupo_etario or '').lower()}"
    if re.search(r"10\D*14", txt) or re.search(r"10\s*-\s*14", txt):
        return "10-14"
    if re.search(r"15\D*19", txt) or re.search(r"15\s*-\s*19", txt):
        return "15-19"
    return None

def is_nan_like(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return True
    s = str(x).strip().lower()
    return s in {"", "nan", "nd", "no_data", "none", "null"}

def clean_str(x, default=None):
    return None if is_nan_like(x) else str(x).strip()

def clean_int(x, default=None):
    if is_nan_like(x):
        return default
    try:
        return int(float(x))
    except Exception:
        return default

def clean_float(x, allow_none=True, default=None):
    if is_nan_like(x):
        return None if allow_none else (default if default is not None else 0.0)
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None if allow_none else (default if default is not None else 0.0)
        return v
    except Exception:
        return None if allow_none else (default if default is not None else 0.0)

def terr_key(rec, nivel: str) -> str:
    return rec.nombre_localidad if nivel.upper() == "LOCALIDAD" else (rec.nombre_upz or "SIN UPZ")

def filtrar_por_cohorte(rows: List[IndicadorFecundidad], cohorte: Optional[str]) -> List[IndicadorFecundidad]:
    if not cohorte:
        return rows
    if cohorte not in COHORTES_VALIDAS:
        return []
    out = []
    for r in rows:
        coh = extraer_grupo_edad(r.indicador_nombre, r.grupo_etario_asociado)
        if coh == cohorte:
            out.append(r)
    return out

# ---------------- Rutas principales ----------------

@app.get("/health")
async def health():
    """Health check optimizado para Railway"""
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
        
        return {
            "status": "healthy",
            "version": "4.0.0",
            "database": db_status,
            "registros": count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/", response_class=HTMLResponse)
async def home():
    """Página principal con dashboard integrado"""
    try:
        with open("dashboard_compatible.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        logger.info("Dashboard HTML not found, serving basic welcome page")
        return HTMLResponse("""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Exploración Determinantes Fecundidad Temprana - Bogotá</title>
            <style>
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; 
                    margin: 0; padding: 2rem; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    min-height: 100vh; color: #333; 
                }
                .container { 
                    max-width: 900px; margin: 0 auto; background: white; padding: 3rem; 
                    border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); 
                }
                h1 { 
                    color: #1e3a8a; margin-bottom: 1rem; font-size: 2.5rem; text-align: center; 
                }
                .subtitle { 
                    text-align: center; color: #64748b; margin-bottom: 2rem; font-size: 1.2rem; 
                }
                .status { 
                    background: #dcfce7; border: 2px solid #16a34a; padding: 1rem; 
                    border-radius: 12px; margin: 2rem 0; text-align: center; 
                }
                .grid { 
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                    gap: 1rem; margin: 2rem 0; 
                }
                .card { 
                    background: #f8fafc; border: 1px solid #e2e8f0; padding: 1.5rem; 
                    border-radius: 12px; text-align: center; 
                }
                .links { 
                    display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; margin: 2rem 0; 
                }
                .btn { 
                    display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.75rem 1.5rem; 
                    background: #2563eb; color: white; text-decoration: none; border-radius: 8px; 
                    transition: all 0.2s; font-weight: 500; 
                }
                .btn:hover { 
                    background: #1d4ed8; transform: translateY(-2px); 
                }
                .feature { 
                    margin: 1rem 0; padding: 1rem; background: #f1f5f9; border-radius: 8px; 
                }
                ul { 
                    text-align: left; margin: 1rem 0; 
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏛️ Exploración Determinantes Fecundidad Temprana</h1>
                <p class="subtitle">Bogotá D.C. - Análisis Territorial Integral</p>
                
                <div class="status">
                    <strong>✅ API Funcionando</strong> - Sistema listo para análisis
                </div>

                <div class="grid">
                    <div class="card">
                        <h3>📊 Caracterización</h3>
                        <p>Análisis territorial por indicadores y cohortes</p>
                    </div>
                    <div class="card">
                        <h3>🔗 Asociación</h3>
                        <p>Correlaciones entre variables</p>
                    </div>
                    <div class="card">
                        <h3>📏 Desigualdad</h3>
                        <p>Índice de Theil territorial</p>
                    </div>
                    <div class="card">
                        <h3>📈 Tendencias</h3>
                        <p>Series temporales por territorio</p>
                    </div>
                </div>

                <div class="links">
                    <a href="/docs" class="btn">📚 Documentación API</a>
                    <a href="/health" class="btn">💚 Estado Sistema</a>
                    <a href="/metadatos" class="btn">📋 Metadatos</a>
                </div>

                <div class="feature">
                    <h3>🎯 Funcionalidades Principales</h3>
                    <ul>
                        <li><strong>Carga de datos:</strong> Upload de archivos Excel con validación</li>
                        <li><strong>Análisis territorial:</strong> Localidades y UPZ con estadísticas descriptivas</li>
                        <li><strong>Cohortes específicas:</strong> Análisis para grupos 10-14 y 15-19 años</li>
                        <li><strong>Correlaciones:</strong> Asociación entre indicadores con tests estadísticos</li>
                        <li><strong>Desigualdad:</strong> Medición con índice de Theil</li>
                        <li><strong>Series temporales:</strong> Evolución de indicadores por territorio</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """)
    except Exception as e:
        logger.error(f"Error serving home page: {e}")
        return HTMLResponse(f"""
        <html>
        <body style="font-family: Arial; padding: 2rem; text-align: center;">
            <h1>⚠️ Error de Configuración</h1>
            <p>Error: {str(e)}</p>
            <p><a href="/docs" style="color: #2563eb;">📚 Ver Documentación API</a></p>
        </body>
        </html>
        """, status_code=500)

@app.post("/upload/excel")
async def upload_excel(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Carga datos desde archivo Excel con validación mejorada"""
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Formato no válido. Use archivos .xlsx o .xls")
    
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents), sheet_name=0)
        
        logger.info(f"Archivo cargado: {file.filename}, filas: {len(df)}, columnas: {len(df.columns)}")
        
        # Normalización de columnas
        column_mapping = {
            'Tipo de Unidad Observación': 'Observación',
            'Año_Inicio': 'año_inicio',
            'AÃ±o_Inicio': 'año_inicio'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Limpiar tabla existente
        deleted_count = db.query(IndicadorFecundidad).count()
        db.query(IndicadorFecundidad).delete()
        
        registros, errores, omitidos_sin_valor = 0, 0, 0
        
        for idx, row in df.iterrows():
            try:
                indicador_nombre = clean_str(row.get('Indicador_Nombre'))
                if not indicador_nombre:
                    continue
                
                valor = clean_float(row.get('Valor'), allow_none=True)
                if valor is None:
                    omitidos_sin_valor += 1
                    continue
                
                # Manejar diferentes formatos de año
                año_inicio = None
                for col in ['año_inicio', 'Año_Inicio', 'AÃ±o_Inicio']:
                    if col in row:
                        año_inicio = clean_int(row.get(col))
                        break
                
                rec = IndicadorFecundidad(
                    archivo_hash=clean_str(row.get('archivo_hash')),
                    indicador_nombre=indicador_nombre,
                    dimension=clean_str(row.get('Dimensión')),
                    unidad_medida=clean_str(row.get('Unidad_Medida'), default='N/A') or 'N/A',
                    tipo_medida=clean_str(row.get('Tipo_Medida')),
                    valor=valor,
                    nivel_territorial=(clean_str(row.get('Nivel_Territorial')) or 'LOCALIDAD').upper(),
                    id_localidad=clean_int(row.get('ID Localidad')),
                    nombre_localidad=clean_str(row.get('Nombre Localidad')) or 'SIN LOCALIDAD',
                    id_upz=clean_int(row.get('ID_UPZ')),
                    nombre_upz=clean_str(row.get('Nombre_UPZ')),
                    area_geografica=clean_str(row.get('Área Geográfica')),
                    año_inicio=año_inicio,
                    periodicidad=clean_str(row.get('Periodicidad')),
                    poblacion_base=clean_str(row.get('Poblacion Base')),
                    semaforo=clean_str(row.get('Semaforo')),
                    grupo_etario_asociado=clean_str(row.get('Grupo Etario Asociado')),
                    sexo=clean_str(row.get('Sexo')),
                    tipo_unidad=clean_str(row.get('Tipo de Unidad')),
                    observacion=clean_str(row.get('Observación')),
                    fuente=clean_str(row.get('Fuente')),
                    url_fuente=clean_str(row.get('URL_Fuente (Opcional)'))
                )
                db.add(rec)
                registros += 1
                
            except Exception as e:
                errores += 1
                logger.warning(f"Error en fila {idx}: {e}")
        
        db.commit()
        logger.info(f"Carga completada: {registros} registros, {errores} errores, {omitidos_sin_valor} omitidos")
        
        return {
            "status": "success",
            "mensaje": "Carga completada exitosamente",
            "archivo": file.filename,
            "estadisticas": {
                "registros_anteriores": deleted_count,
                "registros_cargados": registros,
                "filas_omitidas_sin_valor": omitidos_sin_valor,
                "errores": errores,
                "filas_procesadas": len(df)
            }
        }
        
    except Exception as e:
        logger.exception("Error procesando archivo Excel")
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {str(e)}")

@app.get("/metadatos")
async def metadatos(db: Session = Depends(get_db)):
    """Metadatos completos del sistema"""
    total = db.query(IndicadorFecundidad).count()
    
    # Indicadores categorizados
    todos_indicadores = [r[0] for r in db.query(IndicadorFecundidad.indicador_nombre).distinct().all()]
    indicadores_fecundidad = get_indicadores_fecundidad(db)
    indicadores_otros = get_indicadores_no_fecundidad(db)
    
    # Geografía
    localidades = [r[0] for r in db.query(IndicadorFecundidad.nombre_localidad).distinct().all()]
    upzs = [r[0] for r in db.query(IndicadorFecundidad.nombre_upz).filter(
        IndicadorFecundidad.nombre_upz.isnot(None)
    ).distinct().all()]
    
    # Temporal
    años = sorted([r[0] for r in db.query(IndicadorFecundidad.año_inicio).filter(
        IndicadorFecundidad.año_inicio.isnot(None)
    ).distinct().all()])
    
    # Unidades de medida
    unidades_medida = {}
    for indicador in todos_indicadores:
        unidad = db.query(IndicadorFecundidad.unidad_medida).filter(
            IndicadorFecundidad.indicador_nombre == indicador
        ).first()
        unidades_medida[indicador] = unidad[0] if unidad else "N/A"
    
    return {
        "resumen": {
            "total_registros": total,
            "total_indicadores": len(todos_indicadores),
            "indicadores_fecundidad": len(indicadores_fecundidad),
            "indicadores_otros": len(indicadores_otros),
            "localidades": len(localidades),
            "upz": len([u for u in upzs if u]),
            "rango_años": {"min": min(años) if años else None, "max": max(años) if años else None}
        },
        "indicadores": {
            "todos": sorted(todos_indicadores),
            "fecundidad": sorted(indicadores_fecundidad),
            "otros": sorted(indicadores_otros)
        },
        "geografia": {
            "localidades": sorted(localidades),
            "upz": sorted([u for u in upzs if u])
        },
        "temporal": {
            "años": años,
            "cohortes": sorted(list(COHORTES_VALIDAS))
        },
        "unidades_medida": unidades_medida
    }

@app.get("/estadisticas/generales")
async def estadisticas_generales(año: Optional[int] = Query(None), db: Session = Depends(get_db)):
    q = db.query(IndicadorFecundidad)
    if año is not None:
        q = q.filter(IndicadorFecundidad.año_inicio == año)
    filas = q.all()
    total = len(filas)
    if total == 0:
        return {"total_registros": 0}
    
    c10 = len([1 for f in filas if extraer_grupo_edad(f.indicador_nombre, f.grupo_etario_asociado) == "10-14"])
    c15 = len([1 for f in filas if extraer_grupo_edad(f.indicador_nombre, f.grupo_etario_asociado) == "15-19"])
    localidades = len({f.nombre_localidad for f in filas})
    upz = len({f.nombre_upz for f in filas if f.nombre_upz})
    ind = len({f.indicador_nombre for f in filas})
    
    return {
        "total_registros": total,
        "indicadores_unicos": ind,
        "localidades_unicas": localidades,
        "upzs_unicas": upz,
        "conteo_10_14": c10,
        "conteo_15_19": c15
    }

@app.get("/caracterizacion")
async def caracterizacion(
    indicador: str = Query(..., description="Nombre del indicador a caracterizar"),
    nivel: str = Query("LOCALIDAD", description="Nivel territorial: LOCALIDAD o UPZ"),
    año: Optional[int] = Query(None, description="Año específico (opcional)"),
    cohorte: Optional[str] = Query(None, description="Cohorte específica: 10-14 o 15-19"),
    localidad: Optional[str] = Query(None, description="Filtrar por localidad específica"),
    db: Session = Depends(get_db)
):
    """Caracterización estadística territorial"""
    if nivel.upper() not in {"LOCALIDAD", "UPZ"}:
        raise HTTPException(status_code=400, detail="nivel debe ser LOCALIDAD o UPZ")
    
    q = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador)
    if año is not None:
        q = q.filter(IndicadorFecundidad.año_inicio == año)
    if localidad:
        q = q.filter(IndicadorFecundidad.nombre_localidad == localidad)
    
    rows = filtrar_por_cohorte(q.all(), cohorte)
    if not rows:
        return {"mensaje": "Sin datos para los filtros especificados"}
    
    # Agrupar por territorio
    grupos: Dict[str, List[float]] = {}
    unidad_medida = rows[0].unidad_medida if rows else "N/A"
    
    for r in rows:
        k = terr_key(r, nivel)
        grupos.setdefault(k, []).append(r.valor)
    
    # Calcular estadísticas
    datos = []
    for territorio, valores in grupos.items():
        if not valores:
            continue
            
        arr = np.array(valores, dtype=float)
        if arr.size == 0:
            continue
            
        q1, mediana, q3 = np.percentile(arr, [25, 50, 75])
        promedio = float(np.mean(arr))
        std = float(np.std(arr, ddof=0))
        cv = float((std/promedio)*100) if promedio != 0 else 0.0
        
        datos.append({
            "territorio": territorio,
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
        "indicador": indicador,
        "nivel": nivel.upper(),
        "año": año,
        "cohorte": cohorte,
        "localidad": localidad,
        "unidad_medida": unidad_medida,
        "total_territorios": len(datos),
        "resumen": {
            "promedio_general": round(float(np.mean([d["promedio"] for d in datos])), 3) if datos else 0,
            "n_total": int(sum(d["n"] for d in datos)) if datos else 0
        },
        "datos": datos
    }

@app.get("/analisis/asociacion")
async def asociacion_indicadores(
    indicador_x: str = Query(..., description="Primer indicador"),
    indicador_y: str = Query(..., description="Segundo indicador"),
    nivel: str = Query("LOCALIDAD", description="Nivel territorial"),
    año: Optional[int] = Query(None, description="Año específico"),
    cohorte: Optional[str] = Query(None, description="Cohorte específica"),
    db: Session = Depends(get_db)
):
    """Análisis de asociación entre dos indicadores"""
    if nivel.upper() not in {"LOCALIDAD", "UPZ"}:
        raise HTTPException(status_code=400, detail="nivel debe ser LOCALIDAD o UPZ")
    
    # Obtener datos de ambos indicadores
    qx = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador_x)
    qy = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador_y)
    
    if año is not None:
        qx = qx.filter(IndicadorFecundidad.año_inicio == año)
        qy = qy.filter(IndicadorFecundidad.año_inicio == año)
    
    x_rows = filtrar_por_cohorte(qx.all(), cohorte)
    y_rows = filtrar_por_cohorte(qy.all(), cohorte)
    
    if not x_rows or not y_rows:
        return {"mensaje": "No se encontraron datos para los indicadores seleccionados"}
    
    # Agrupar por territorio
    x_map: Dict[str, List[float]] = {}
    y_map: Dict[str, List[float]] = {}
    
    for r in x_rows:
        x_map.setdefault(terr_key(r, nivel), []).append(r.valor)
    
    for r in y_rows:
        y_map.setdefault(terr_key(r, nivel), []).append(r.valor)
    
    # Territorios comunes
    territorios_comunes = set(x_map.keys()) & set(y_map.keys())
    if len(territorios_comunes) < 3:
        return {"mensaje": "Insuficientes territorios comunes para el análisis"}
    
    # Calcular promedios
    x_mean = {k: float(np.mean(v)) for k, v in x_map.items() if k in territorios_comunes}
    y_mean = {k: float(np.mean(v)) for k, v in y_map.items() if k in territorios_comunes}
    
    # Arrays para correlación
    x_vals = [x_mean[t] for t in territorios_comunes]
    y_vals = [y_mean[t] for t in territorios_comunes]
    
    if np.std(x_vals) == 0 or np.std(y_vals) == 0:
        return {"mensaje": "Una de las variables no tiene variación"}
    
    # Correlaciones
    r_p, p_p = stats.pearsonr(x_vals, y_vals)
    r_s, p_s = stats.spearmanr(x_vals, y_vals)
    
    # Categorizar correlación
    abs_r = abs(float(r_p))
    if abs_r >= 0.7:
        categoria = "Fuerte"
    elif abs_r >= 0.5:
        categoria = "Moderada"
    elif abs_r >= 0.3:
        categoria = "Débil"
    else:
        categoria = "Muy débil"
    
    # Datos para gráfico de dispersión
    datos_pares = []
    for territorio in territorios_comunes:
        datos_pares.append({
            "territorio": territorio,
            "x": round(x_mean[territorio], 3),
            "y": round(y_mean[territorio], 3)
        })
    
    return {
        "indicador_x": indicador_x,
        "indicador_y": indicador_y,
        "nivel": nivel.upper(),
        "año": año,
        "cohorte": cohorte,
        "territorios_comunes": len(territorios_comunes),
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

@app.get("/analisis/theil")
async def indice_theil(
    indicador: str = Query(..., description="Indicador para análisis de desigualdad"),
    nivel: str = Query("LOCALIDAD", description="Nivel territorial"),
    año: Optional[int] = Query(None, description="Año específico"),
    cohorte: Optional[str] = Query(None, description="Cohorte específica"),
    db: Session = Depends(get_db)
):
    """Calcula el índice de Theil para medir desigualdad territorial"""
    if nivel.upper() not in {"LOCALIDAD", "UPZ"}:
        raise HTTPException(status_code=400, detail="nivel debe ser LOCALIDAD o UPZ")
    
    q = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador)
    if año is not None:
        q = q.filter(IndicadorFecundidad.año_inicio == año)
    
    rows = filtrar_por_cohorte(q.all(), cohorte)
    if not rows:
        return {"mensaje": "Sin datos para los filtros especificados"}
    
    # Agrupar por territorio
    grupos: Dict[str, List[float]] = {}
    unidad_medida = rows[0].unidad_medida if rows else "N/A"
    
    for r in rows:
        k = terr_key(r, nivel)
        grupos.setdefault(k, []).append(r.valor)
    
    # Calcular promedios por territorio
    territorios = []
    valores = []
    
    for territorio, vals in grupos.items():
        if vals:
            promedio = float(np.mean(vals))
            territorios.append(territorio)
            valores.append(promedio)
    
    if len(valores) < 2:
        return {"mensaje": "Insuficientes territorios para calcular el índice de Theil"}
    
    # Calcular índice de Theil
    theil = calcular_indice_theil(valores)
    
    # Estadísticas adicionales
    mean_val = float(np.mean(valores))
    std_val = float(np.std(valores))
    cv = (std_val / mean_val * 100) if mean_val != 0 else 0
    
    # Datos por territorio
    datos_territorios = []
    for i, territorio in enumerate(territorios):
        datos_territorios.append({
            "territorio": territorio,
            "valor": round(valores[i], 3),
            "desviacion_media": round(valores[i] - mean_val, 3),
            "ratio_media": round(valores[i] / mean_val, 3) if mean_val != 0 else 0
        })
    
    datos_territorios.sort(key=lambda x: x["valor"], reverse=True)
    
    return {
        "indicador": indicador,
        "nivel": nivel.upper(),
        "año": año,
        "cohorte": cohorte,
        "unidad_medida": unidad_medida,
        "indice_theil": round(theil, 4),
        "interpretacion": {
            "valor": round(theil, 4),
            "significado": "0 = igualdad perfecta, >0 = mayor desigualdad",
            "categoria": "Baja" if theil < 0.1 else "Moderada" if theil < 0.3 else "Alta"
        },
        "estadisticas": {
            "territorios": len(territorios),
            "promedio_general": round(mean_val, 3),
            "desviacion_estandar": round(std_val, 3),
            "coeficiente_variacion": round(cv, 2),
            "min": round(min(valores), 3),
            "max": round(max(valores), 3)
        },
        "datos": datos_territorios
    }

@app.get("/datos/series")
async def serie_temporal(
    indicador: str = Query(..., description="Indicador para análisis temporal"),
    territorio: str = Query(..., description="Territorio específico"),
    nivel: str = Query("LOCALIDAD", description="Nivel territorial"),
    cohorte: Optional[str] = Query(None, description="Cohorte específica"),
    db: Session = Depends(get_db)
):
    """Serie temporal de un indicador en un territorio específico"""
    q = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador)
    
    if nivel.upper() == "LOCALIDAD":
        q = q.filter(IndicadorFecundidad.nombre_localidad == territorio)
    else:
        q = q.filter(IndicadorFecundidad.nombre_upz == territorio)
    
    rows = q.filter(IndicadorFecundidad.año_inicio.isnot(None)).order_by(
        IndicadorFecundidad.año_inicio.asc()
    ).all()
    
    rows = filtrar_por_cohorte(rows, cohorte)
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
        "indicador": indicador,
        "nivel": nivel.upper(),
        "territorio": territorio,
        "cohorte": cohorte,
        "unidad_medida": unidad_medida,
        "periodo": {
            "inicio": min(grupos_año.keys()),
            "fin": max(grupos_año.keys()),
            "años": len(grupos_año)
        },
        "serie": serie_datos
    }

@app.get("/brechas/cohortes")
async def brechas_cohortes(
    indicador: str = Query(...),
    nivel: str = Query("LOCALIDAD"),
    año: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    """Calcula brecha 15-19 menos 10-14 por territorio"""
    if nivel.upper() not in {"LOCALIDAD", "UPZ"}:
        raise HTTPException(status_code=400, detail="nivel debe ser LOCALIDAD o UPZ")
    
    q = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador)
    if año is not None:
        q = q.filter(IndicadorFecundidad.año_inicio == año)
    
    rows = q.all()
    if not rows:
        return {"mensaje": "Sin datos para esos filtros"}
    
    m10: Dict[str, List[float]] = {}
    m15: Dict[str, List[float]] = {}
    unidad_medida = rows[0].unidad_medida if rows else "N/A"
    
    for r in rows:
        coh = extraer_grupo_edad(r.indicador_nombre, r.grupo_etario_asociado)
        if coh not in COHORTES_VALIDAS:
            continue
        k = terr_key(r, nivel)
        if coh == "10-14":
            m10.setdefault(k, []).append(r.valor)
        else:
            m15.setdefault(k, []).append(r.valor)
    
    territorios = sorted(set(m10.keys()) | set(m15.keys()))
    datos = []
    
    for t in territorios:
        v10 = float(np.mean(m10.get(t, []))) if m10.get(t) else None
        v15 = float(np.mean(m15.get(t, []))) if m15.get(t) else None
        
        if v10 is None or v15 is None:
            continue
            
        delta = v15 - v10
        ratio = (v15 / v10) if v10 not in (0, None) else None
        pct_change = ((v15 - v10) / v10 * 100) if v10 not in (0, None) else None
        
        datos.append({
            "territorio": t,
            "prom_10_14": round(v10, 3),
            "prom_15_19": round(v15, 3),
            "brecha_abs": round(delta, 3),
            "brecha_rel": round(ratio, 3) if ratio is not None else None,
            "cambio_porcentual": round(pct_change, 2) if pct_change is not None else None,
            "n_10_14": len(m10.get(t, [])),
            "n_15_19": len(m15.get(t, []))
        })
    
    datos.sort(key=lambda d: d["brecha_abs"], reverse=True)
    
    return {
        "indicador": indicador,
        "nivel": nivel.upper(),
        "año": año,
        "unidad_medida": unidad_medida,
        "total_territorios": len(datos),
        "resumen": {
            "brecha_promedio": round(float(np.mean([d["brecha_abs"] for d in datos])), 3) if datos else 0,
            "mayor_brecha": datos[0]["territorio"] if datos else None,
            "menor_brecha": datos[-1]["territorio"] if datos else None
        },
        "datos": datos
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Iniciando servidor en puerto {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
