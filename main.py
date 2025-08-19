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

# Configurar logging primero
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración de base de datos simplificada
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///fecundidad_temprana.db")
logger.info(f"Using database: {DATABASE_URL[:50]}...")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Configuración básica del engine
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
    # Fallback a SQLite simple
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
    valor = Column(Float, nullable=False)  # <- no nulls
    nivel_territorial = Column(String, index=True, nullable=False)
    id_localidad = Column(Integer, index=True, nullable=True)
    nombre_localidad = Column(String, index=True, nullable=False)
    id_upz = Column(Integer, index=True, nullable=True)
    nombre_upz = Column(String, index=True, nullable=True)
    area_geografica = Column(String)
    año_inicio = Column(Integer)
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
    title="API Fecundidad Temprana - Bogotá D.C.",
    description="Análisis integral de fecundidad temprana por territorio, periodo y cohortes (10-14 y 15-19 años).",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear tablas de forma segura al arrancar
@app.on_event("startup")
async def startup_event():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified successfully")
    except Exception as e:
        logger.warning(f"Could not create tables on startup: {e}")
        # No fallar si hay problemas con las tablas

# ---------------- Utilidades ----------------
COHORTES_VALIDAS = {"10-14","15-19"}

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
    if isinstance(x, float) and np.isnan(x):
        return True
    s = str(x).strip().lower()
    return s in {"", "nan", "nd", "no_data", "none"}

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

# ---------------- Rutas básicas ----------------
@app.get("/health")
async def health():
    """Endpoint de health check simplificado para Railway"""
    try:
        # Test básico de base de datos
        db = SessionLocal()
        try:
            result = db.execute(text("SELECT 1")).scalar()
            db_status = "connected" if result == 1 else "error"
        except Exception:
            db_status = "disconnected"
        finally:
            db.close()
        
        return {
            "status": "healthy",
            "version": "3.1.0",
            "database": db_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Página principal - dashboard o página de bienvenida"""
    # Intentar cargar el dashboard, pero no fallar si no existe
    try:
        with open("dashboard_compatible.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        logger.info("Dashboard HTML not found, serving basic page")
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>API Fecundidad Temprana - Bogotá D.C.</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f8fafc; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                h1 { color: #1e3a8a; margin-bottom: 20px; }
                .status { background: #dcfce7; border: 1px solid #16a34a; padding: 15px; border-radius: 8px; margin: 20px 0; }
                .links { margin: 30px 0; }
                .links a { display: inline-block; margin: 10px 15px 10px 0; padding: 10px 20px; background: #2563eb; color: white; text-decoration: none; border-radius: 6px; }
                .links a:hover { background: #1d4ed8; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 API Fecundidad Temprana - Bogotá D.C.</h1>
                <div class="status">
                    <strong>✅ Estado:</strong> API funcionando correctamente
                </div>
                <p>Análisis integral de fecundidad temprana por territorio, periodo y cohortes (10-14 y 15-19 años).</p>
                <div class="links">
                    <a href="/docs">📚 Documentación API</a>
                    <a href="/health">💚 Estado del Sistema</a>
                    <a href="/metadatos">📊 Metadatos</a>
                </div>
                <p><strong>Funcionalidades disponibles:</strong></p>
                <ul>
                    <li>Carga de datos desde archivos Excel</li>
                    <li>Caracterización territorial por indicadores</li>
                    <li>Análisis de asociación entre variables</li>
                    <li>Análisis de brechas entre cohortes</li>
                    <li>Análisis de tendencias temporales</li>
                </ul>
            </div>
        </body>
        </html>
        """)
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        return HTMLResponse(f"""
        <html>
        <head><title>API Error</title></head>
        <body style="font-family: Arial, sans-serif; padding: 2rem; text-align: center;">
            <h1>⚠️ Error de Configuración</h1>
            <p>Error: {str(e)}</p>
            <p><a href="/docs" style="color: #2563eb;">📚 Ver Documentación API</a></p>
        </body>
        </html>
        """)

@app.post("/upload/excel")
async def upload_excel(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(('.xlsx','.xls')):
        raise HTTPException(status_code=400, detail="Formato no válido (.xlsx/.xls)")
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents), sheet_name=0)

        # Normalización de columnas conocidas
        if 'Tipo de Unidad Observación' in df.columns and 'Observación' not in df.columns:
            df['Observación'] = df['Tipo de Unidad Observación']

        # Limpiar tabla antes de insertar
        db.query(IndicadorFecundidad).delete()

        registros, errores, omitidos_sin_valor = 0, 0, 0

        for _, row in df.iterrows():
            try:
                indicador_nombre = clean_str(row.get('Indicador_Nombre'))
                if not indicador_nombre:
                    continue  # no insertamos filas sin nombre de indicador

                # --- valor: si no es numérico o es NaN -> OMITIR FILA (para respetar NOT NULL) ---
                valor = clean_float(row.get('Valor'), allow_none=True)
                if valor is None:
                    omitidos_sin_valor += 1
                    continue

                rec = IndicadorFecundidad(
                    archivo_hash = clean_str(row.get('archivo_hash')),
                    indicador_nombre = indicador_nombre,
                    dimension = clean_str(row.get('Dimensión')),
                    unidad_medida = clean_str(row.get('Unidad_Medida'), default='N/A') or 'N/A',
                    tipo_medida = clean_str(row.get('Tipo_Medida')),
                    valor = valor,
                    nivel_territorial = (clean_str(row.get('Nivel_Territorial')) or 'LOCALIDAD').upper(),
                    id_localidad = clean_int(row.get('ID Localidad')),
                    nombre_localidad = clean_str(row.get('Nombre Localidad')) or 'SIN LOCALIDAD',
                    id_upz = clean_int(row.get('ID_UPZ')),
                    nombre_upz = clean_str(row.get('Nombre_UPZ')),
                    area_geografica = clean_str(row.get('Área Geográfica')),
                    año_inicio = clean_int(row.get('Año_Inicio')),
                    periodicidad = clean_str(row.get('Periodicidad')),
                    poblacion_base = clean_str(row.get('Poblacion Base')),
                    semaforo = clean_str(row.get('Semaforo')),
                    grupo_etario_asociado = clean_str(row.get('Grupo Etario Asociado')),
                    sexo = clean_str(row.get('Sexo')),
                    tipo_unidad = clean_str(row.get('Tipo de Unidad')),
                    observacion = clean_str(row.get('Observación')),
                    fuente = clean_str(row.get('Fuente')),
                    url_fuente = clean_str(row.get('URL_Fuente (Opcional)'))
                )
                db.add(rec)
                registros += 1

            except Exception as e:
                errores += 1
                logger.warning(f"Fila con error: {e}")

        db.commit()

        return {
            "status": "success",
            "mensaje": "Carga completada",
            "registros_cargados": registros,
            "filas_omitidas_sin_valor": omitidos_sin_valor,
            "errores": errores
        }
    except Exception as e:
        logger.exception("Error procesando archivo")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metadatos")
async def metadatos(db: Session = Depends(get_db)):
    total = db.query(IndicadorFecundidad).count()
    indicadores = [r[0] for r in db.query(IndicadorFecundidad.indicador_nombre).distinct().all()]
    localidades = [r[0] for r in db.query(IndicadorFecundidad.nombre_localidad).distinct().all()]
    upzs = [r[0] for r in db.query(IndicadorFecundidad.nombre_upz).filter(IndicadorFecundidad.nombre_upz.isnot(None)).distinct().all()]
    años = sorted([r[0] for r in db.query(IndicadorFecundidad.año_inicio).filter(IndicadorFecundidad.año_inicio.isnot(None)).distinct().all()])
    
    # Obtener unidades de medida por indicador
    unidades_medida = {}
    for indicador in indicadores:
        unidad = db.query(IndicadorFecundidad.unidad_medida).filter(
            IndicadorFecundidad.indicador_nombre == indicador
        ).first()
        unidades_medida[indicador] = unidad[0] if unidad else "N/A"
    
    return {
        "total_registros": total, 
        "indicadores": sorted(indicadores), 
        "localidades": sorted(localidades), 
        "upz": sorted([u for u in upzs if u]), 
        "años": años, 
        "cohortes": sorted(list(COHORTES_VALIDAS)),
        "unidades_medida": unidades_medida
    }

@app.get("/localidades/{nombre_localidad}/upz")
async def get_upz_por_localidad(nombre_localidad: str, db: Session = Depends(get_db)):
    """Obtiene las UPZ de una localidad específica"""
    upzs = [r[0] for r in db.query(IndicadorFecundidad.nombre_upz).filter(
        IndicadorFecundidad.nombre_localidad == nombre_localidad,
        IndicadorFecundidad.nombre_upz.isnot(None)
    ).distinct().all()]
    return {"localidad": nombre_localidad, "upz": sorted(upzs)}

@app.get("/estadisticas/generales")
async def estadisticas_generales(año: Optional[int] = Query(None), db: Session = Depends(get_db)):
    q = db.query(IndicadorFecundidad)
    if año is not None:
        q = q.filter(IndicadorFecundidad.año_inicio == año)
    filas = q.all()
    total = len(filas)
    if total == 0:
        return {"total_registros":0}
    c10 = len([1 for f in filas if extraer_grupo_edad(f.indicador_nombre, f.grupo_etario_asociado) == "10-14"])
    c15 = len([1 for f in filas if extraer_grupo_edad(f.indicador_nombre, f.grupo_etario_asociado) == "15-19"])
    localidades = len({f.nombre_localidad for f in filas})
    upz = len({f.nombre_upz for f in filas if f.nombre_upz})
    ind = len({f.indicador_nombre for f in filas})
    return {"total_registros": total, "indicadores_unicos": ind, "localidades_unicas": localidades, "upzs_unicas": upz, "conteo_10_14": c10, "conteo_15_19": c15}

@app.get("/indicadores")
async def listar_indicadores(buscar: Optional[str] = Query(None), db: Session = Depends(get_db)):
    q = db.query(IndicadorFecundidad.indicador_nombre, IndicadorFecundidad.unidad_medida, func.count().label("n")).group_by(IndicadorFecundidad.indicador_nombre, IndicadorFecundidad.unidad_medida)
    if buscar:
        q = q.filter(IndicadorFecundidad.indicador_nombre.ilike(f"%{buscar}%"))
    filas = q.order_by(func.count().desc()).all()
    return [{"indicador": f[0], "unidad": f[1], "registros": int(f[2])} for f in filas]

def _caracterizacion_desde_rows(rows: List[IndicadorFecundidad], nivel: str) -> List[Dict]:
    grupos: Dict[str, List[float]] = {}
    unidad_medida = None
    for r in rows:
        if unidad_medida is None:
            unidad_medida = r.unidad_medida
        k = terr_key(r, nivel)
        grupos.setdefault(k, []).append(r.valor)
    out = []
    for terr, vals in grupos.items():
        arr = np.array(vals, dtype=float)
        if arr.size == 0: continue
        q1, med, q3 = np.percentile(arr, [25,50,75])
        mean = float(np.mean(arr)); std = float(np.std(arr, ddof=0))
        cv = float((std/mean)*100) if mean != 0 else 0.0
        out.append({
            "territorio": terr, "n": int(arr.size),
            "promedio": round(mean,3), "mediana": round(float(med),3),
            "q1": round(float(q1),3), "q3": round(float(q3),3),
            "min": round(float(np.min(arr)),3), "max": round(float(np.max(arr)),3),
            "desv_estandar": round(std,3), "cv_pct": round(cv,2)
        })
    out.sort(key=lambda x: x["promedio"], reverse=True)
    return out, unidad_medida

@app.get("/caracterizacion")
async def caracterizacion(
    indicador: str = Query(...),
    nivel: str = Query("LOCALIDAD"),
    año: Optional[int] = Query(None),
    cohorte: Optional[str] = Query(None, description="10-14 o 15-19"),
    localidad: Optional[str] = Query(None, description="Filtrar por localidad específica"),
    db: Session = Depends(get_db)
):
    if nivel.upper() not in {"LOCALIDAD","UPZ"}:
        raise HTTPException(status_code=400, detail="nivel debe ser LOCALIDAD o UPZ")
    
    q = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador)
    if año is not None:
        q = q.filter(IndicadorFecundidad.año_inicio == año)
    if localidad:
        q = q.filter(IndicadorFecundidad.nombre_localidad == localidad)
    
    rows = q.all()
    rows = filtrar_por_cohorte(rows, cohorte)
    if not rows:
        return {"mensaje":"Sin datos para los filtros."}
    
    datos, unidad_medida = _caracterizacion_desde_rows(rows, nivel)
    
    # Si estamos filtrando por localidad y nivel es UPZ, también traer el total de la localidad
    resumen_localidad = None
    if localidad and nivel.upper() == "UPZ":
        q_loc = db.query(IndicadorFecundidad).filter(
            IndicadorFecundidad.indicador_nombre == indicador,
            IndicadorFecundidad.nombre_localidad == localidad
        )
        if año is not None:
            q_loc = q_loc.filter(IndicadorFecundidad.año_inicio == año)
        rows_loc = filtrar_por_cohorte(q_loc.all(), cohorte)
        if rows_loc:
            valores_loc = [r.valor for r in rows_loc]
            arr_loc = np.array(valores_loc, dtype=float)
            resumen_localidad = {
                "territorio": localidad,
                "n": int(arr_loc.size),
                "promedio": round(float(np.mean(arr_loc)), 3),
                "mediana": round(float(np.percentile(arr_loc, 50)), 3),
                "min": round(float(np.min(arr_loc)), 3),
                "max": round(float(np.max(arr_loc)), 3),
                "desv_estandar": round(float(np.std(arr_loc, ddof=0)), 3)
            }
    
    return {
        "indicador": indicador, 
        "nivel": nivel.upper(), 
        "año": año, 
        "cohorte": cohorte,
        "localidad": localidad,
        "unidad_medida": unidad_medida,
        "total_territorios": len(datos),
        "resumen": {
            "promedio_general": round(float(np.mean([d["promedio"] for d in datos])),3), 
            "n_total": int(sum(d["n"] for d in datos))
        },
        "resumen_localidad": resumen_localidad,
        "datos": datos
    }

@app.get("/caracterizacion/comparativa")
async def caracterizacion_comparativa(
    indicador: str = Query(...),
    localidad: str = Query(...),
    año: Optional[int] = Query(None),
    cohorte: Optional[str] = Query(None, description="10-14 o 15-19"),
    db: Session = Depends(get_db)
):
    """Compara una localidad específica con el total de Bogotá y sus UPZ"""
    
    # Datos de la localidad
    q_loc = db.query(IndicadorFecundidad).filter(
        IndicadorFecundidad.indicador_nombre == indicador,
        IndicadorFecundidad.nombre_localidad == localidad
    )
    if año is not None:
        q_loc = q_loc.filter(IndicadorFecundidad.año_inicio == año)
    
    rows_loc = filtrar_por_cohorte(q_loc.all(), cohorte)
    
    # Datos de todas las localidades (total Bogotá)
    q_total = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador)
    if año is not None:
        q_total = q_total.filter(IndicadorFecundidad.año_inicio == año)
    
    rows_total = filtrar_por_cohorte(q_total.all(), cohorte)
    
    # UPZ de la localidad específica
    upz_rows = [r for r in rows_loc if r.nombre_upz]
    
    if not rows_loc:
        return {"mensaje": "Sin datos para la localidad especificada"}
    
    # Calcular estadísticas
    vals_loc = [r.valor for r in rows_loc]
    vals_total = [r.valor for r in rows_total]
    
    unidad_medida = rows_loc[0].unidad_medida if rows_loc else "N/A"
    
    result = {
        "indicador": indicador,
        "localidad": localidad,
        "año": año,
        "cohorte": cohorte,
        "unidad_medida": unidad_medida,
        "localidad_stats": {
            "n": len(vals_loc),
            "promedio": round(float(np.mean(vals_loc)), 3),
            "mediana": round(float(np.percentile(vals_loc, 50)), 3),
            "min": round(float(np.min(vals_loc)), 3),
            "max": round(float(np.max(vals_loc)), 3)
        },
        "bogota_stats": {
            "n": len(vals_total),
            "promedio": round(float(np.mean(vals_total)), 3),
            "mediana": round(float(np.percentile(vals_total, 50)), 3),
            "min": round(float(np.min(vals_total)), 3),
            "max": round(float(np.max(vals_total)), 3)
        }
    }
    
    # UPZ de la localidad
    if upz_rows:
        upz_grupos = {}
        for r in upz_rows:
            upz_grupos.setdefault(r.nombre_upz, []).append(r.valor)
        
        upz_stats = []
        for upz, vals in upz_grupos.items():
            upz_stats.append({
                "upz": upz,
                "n": len(vals),
                "promedio": round(float(np.mean(vals)), 3),
                "mediana": round(float(np.percentile(vals, 50)), 3)
            })
        result["upz_stats"] = sorted(upz_stats, key=lambda x: x["promedio"], reverse=True)
    
    return result

@app.get("/analisis/asociacion")
async def asociacion(
    indicador_objetivo: str = Query(...),
    nivel: str = Query("LOCALIDAD"),
    año: Optional[int] = Query(None),
    cohorte: Optional[str] = Query(None, description="10-14 o 15-19"),
    top: int = Query(10, ge=3, le=50),
    db: Session = Depends(get_db)
):
    if nivel.upper() not in {"LOCALIDAD","UPZ"}:
        raise HTTPException(status_code=400, detail="nivel debe ser LOCALIDAD o UPZ")

    # Objetivo (Y)
    qy = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador_objetivo)
    if año is not None:
        qy = qy.filter(IndicadorFecundidad.año_inicio == año)
    y_rows = filtrar_por_cohorte(qy.all(), cohorte)
    if not y_rows:
        return {"mensaje":"No se encontraron datos para el indicador objetivo."}
    y_map: Dict[str, List[float]] = {}
    for r in y_rows:
        y_map.setdefault(terr_key(r, nivel), []).append(r.valor)
    y_mean = {k: float(np.mean(v)) for k,v in y_map.items()}

    # Otros indicadores (X)
    otros = [r[0] for r in db.query(IndicadorFecundidad.indicador_nombre).filter(IndicadorFecundidad.indicador_nombre != indicador_objetivo).distinct().all()]
    resultados = []
    for ind in otros:
        qx = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == ind)
        if año is not None:
            qx = qx.filter(IndicadorFecundidad.año_inicio == año)
        x_rows = filtrar_por_cohorte(qx.all(), cohorte)
        if not x_rows: continue
        x_map: Dict[str, List[float]] = {}
        for r in x_rows:
            x_map.setdefault(terr_key(r, nivel), []).append(r.valor)
        comunes = set(x_map.keys()) & set(y_mean.keys())
        if len(comunes) < 3: continue
        x = [float(np.mean(x_map[t])) for t in comunes]
        y = [y_mean[t] for t in comunes]
        if np.std(x)==0 or np.std(y)==0: continue
        r_p, p_p = stats.pearsonr(x, y)
        r_s, p_s = stats.spearmanr(x, y)
        
        # Categorizar la correlación
        abs_r = abs(float(r_p))
        if abs_r >= 0.7:
            categoria = "Fuerte"
        elif abs_r >= 0.5:
            categoria = "Moderada"
        elif abs_r >= 0.3:
            categoria = "Débil"
        else:
            categoria = "Muy débil"
            
        resultados.append({
            "indicador_comparado": ind, 
            "territorios": len(comunes),
            "pearson_r": round(float(r_p),3), 
            "pearson_p": round(float(p_p),4),
            "spearman_rho": round(float(r_s),3), 
            "spearman_p": round(float(p_s),4),
            "categoria_correlacion": categoria,
            "significativa": float(p_p) < 0.05
        })
    resultados.sort(key=lambda d: abs(d["pearson_r"]), reverse=True)
    return {
        "indicador_objetivo": indicador_objetivo, 
        "nivel": nivel.upper(), 
        "año": año, 
        "cohorte": cohorte, 
        "comparaciones": resultados[:top]
    }

@app.get("/analisis/asociacion/detalle")
async def asociacion_detalle(
    indicador_objetivo: str = Query(...),
    indicador_comparado: str = Query(...),
    nivel: str = Query("LOCALIDAD"),
    año: Optional[int] = Query(None),
    cohorte: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Obtiene los datos detallados para la dispersión entre dos indicadores"""
    
    # Datos del indicador objetivo
    qy = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador_objetivo)
    if año is not None:
        qy = qy.filter(IndicadorFecundidad.año_inicio == año)
    y_rows = filtrar_por_cohorte(qy.all(), cohorte)
    
    # Datos del indicador comparado
    qx = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador_comparado)
    if año is not None:
        qx = qx.filter(IndicadorFecundidad.año_inicio == año)
    x_rows = filtrar_por_cohorte(qx.all(), cohorte)
    
    # Agrupar por territorio
    y_map = {}
    x_map = {}
    
    for r in y_rows:
        territorio = terr_key(r, nivel)
        y_map.setdefault(territorio, []).append(r.valor)
    
    for r in x_rows:
        territorio = terr_key(r, nivel)
        x_map.setdefault(territorio, []).append(r.valor)
    
    # Crear pares de datos para territorios comunes
    territorios_comunes = set(y_map.keys()) & set(x_map.keys())
    datos_pares = []
    
    for territorio in territorios_comunes:
        x_mean = float(np.mean(x_map[territorio]))
        y_mean = float(np.mean(y_map[territorio]))
        datos_pares.append({
            "territorio": territorio,
            "x": round(x_mean, 3),
            "y": round(y_mean, 3)
        })
    
    return {
        "indicador_objetivo": indicador_objetivo,
        "indicador_comparado": indicador_comparado,
        "nivel": nivel,
        "año": año,
        "cohorte": cohorte,
        "territorios_comunes": len(territorios_comunes),
        "datos": datos_pares
    }

@app.get("/datos/series")
async def serie_temporal(
    indicador: str = Query(...),
    territorio: str = Query(...),
    nivel: str = Query("LOCALIDAD"),
    cohorte: Optional[str] = Query(None, description="10-14 o 15-19"),
    db: Session = Depends(get_db)
):
    q = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador)
    if nivel.upper() == "LOCALIDAD":
        q = q.filter(IndicadorFecundidad.nombre_localidad == territorio)
    else:
        q = q.filter(IndicadorFecundidad.nombre_upz == territorio)
    rows = q.filter(IndicadorFecundidad.año_inicio.isnot(None)).order_by(IndicadorFecundidad.año_inicio.asc()).all()
    rows = filtrar_por_cohorte(rows, cohorte)
    if not rows:
        return {"mensaje":"Sin datos para esos filtros."}
    
    # Agrupar por año y calcular promedios
    grupos_año = {}
    unidad_medida = rows[0].unidad_medida if rows else "N/A"
    
    for r in rows:
        grupos_año.setdefault(r.año_inicio, []).append(r.valor)
    
    serie_datos = []
    for año in sorted(grupos_año.keys()):
        valores = grupos_año[año]
        serie_datos.append({
            "año": año,
            "valor": round(float(np.mean(valores)), 3),
            "n_observaciones": len(valores)
        })
    
    return {
        "indicador": indicador, 
        "nivel": nivel.upper(), 
        "territorio": territorio, 
        "cohorte": cohorte,
        "unidad_medida": unidad_medida,
        "serie": serie_datos
    }

@app.get("/brechas/cohortes")
async def brechas_cohortes(
    indicador: str = Query(...),
    nivel: str = Query("LOCALIDAD"),
    año: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    """Calcula brecha 15-19 menos 10-14 por territorio (promedio por cohorte)."""
    if nivel.upper() not in {"LOCALIDAD","UPZ"}:
        raise HTTPException(status_code=400, detail="nivel debe ser LOCALIDAD o UPZ")
    q = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador)
    if año is not None:
        q = q.filter(IndicadorFecundidad.año_inicio == año)
    rows = q.all()
    if not rows:
        return {"mensaje":"Sin datos para esos filtros."}
    
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
            "prom_10_14": round(v10,3), 
            "prom_15_19": round(v15,3), 
            "brecha_abs": round(delta,3), 
            "brecha_rel": round(ratio,3) if ratio is not None else None,
            "cambio_porcentual": round(pct_change,2) if pct_change is not None else None,
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

@app.get("/analisis/tendencias")
async def analisis_tendencias(
    indicador: str = Query(...),
    nivel: str = Query("LOCALIDAD"),
    cohorte: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Analiza tendencias temporales agregadas por territorio"""
    
    q = db.query(IndicadorFecundidad).filter(
        IndicadorFecundidad.indicador_nombre == indicador,
        IndicadorFecundidad.año_inicio.isnot(None)
    )
    rows = filtrar_por_cohorte(q.all(), cohorte)
    
    if not rows:
        return {"mensaje": "Sin datos para los filtros especificados"}
    
    # Agrupar por territorio y año
    datos_territorio = {}
    unidad_medida = rows[0].unidad_medida if rows else "N/A"
    
    for r in rows:
        territorio = terr_key(r, nivel)
        if territorio not in datos_territorio:
            datos_territorio[territorio] = {}
        año = r.año_inicio
        if año not in datos_territorio[territorio]:
            datos_territorio[territorio][año] = []
        datos_territorio[territorio][año].append(r.valor)
    
    # Calcular tendencias por territorio
    tendencias = []
    for territorio, años_datos in datos_territorio.items():
        if len(años_datos) < 2:  # Necesita al menos 2 puntos para calcular tendencia
            continue
            
        años = sorted(años_datos.keys())
        valores = [np.mean(años_datos[año]) for año in años]
        
        # Calcular tendencia lineal
        if len(años) >= 2:
            pendiente, intercepto, r_value, p_value, std_err = stats.linregress(años, valores)
            
            # Clasificar tendencia
            if abs(pendiente) < 0.1:
                clasificacion = "Estable"
            elif pendiente > 0:
                clasificacion = "Creciente" if pendiente > 0.5 else "Levemente creciente"
            else:
                clasificacion = "Decreciente" if pendiente < -0.5 else "Levemente decreciente"
            
            tendencias.append({
                "territorio": territorio,
                "años_disponibles": len(años),
                "periodo": f"{años[0]}-{años[-1]}",
                "valor_inicial": round(valores[0], 3),
                "valor_final": round(valores[-1], 3),
                "cambio_absoluto": round(valores[-1] - valores[0], 3),
                "cambio_porcentual": round(((valores[-1] - valores[0]) / valores[0]) * 100, 2) if valores[0] != 0 else None,
                "pendiente": round(pendiente, 4),
                "r_cuadrado": round(r_value**2, 3),
                "p_valor": round(p_value, 4),
                "clasificacion": clasificacion,
                "significativa": p_value < 0.05
            })
    
    # Ordenar por magnitud del cambio
    tendencias.sort(key=lambda x: abs(x["cambio_absoluto"]), reverse=True)
    
    return {
        "indicador": indicador,
        "nivel": nivel,
        "cohorte": cohorte,
        "unidad_medida": unidad_medida,
        "total_territorios": len(tendencias),
        "tendencias": tendencias
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on 0.0.0.0:{port}")
    uvicorn.run(
        "main:app",  # Usar string en lugar de objeto app directamente
        host="0.0.0.0", 
        port=port,
        log_level="info",
        access_log=True
    )
