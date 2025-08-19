from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Index, func
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import io, os, logging, re, hashlib
from datetime import datetime
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de base de datos
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///fecundidad_temprana.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

try:
    if "postgresql://" in DATABASE_URL:
        engine = create_engine(
            DATABASE_URL, 
            echo=False,
            pool_pre_ping=True,
            pool_recycle=300
        )
    else:
        engine = create_engine(DATABASE_URL, echo=False)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Database engine created successfully")
except Exception as e:
    logger.error(f"Database error: {e}")
    engine = create_engine("sqlite:///fallback.db", echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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
    ano_inicio = Column(Integer, index=True)
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
        Index('idx_nivel_ano', 'nivel_territorial', 'ano_inicio'),
    )

# Crear tablas
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Tables created successfully")
except Exception as e:
    logger.error(f"Error creating tables: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(
    title="API Fecundidad Temprana - Bogotá D.C.",
    description="Análisis integral por territorio, periodo y cohortes",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utilidades
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
    return s in {"", "nan", "nd", "no_data", "none", "null"}

def clean_str(x, default=None):
    if is_nan_like(x):
        return default
    return str(x).strip()

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

def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas comunes"""
    # Mapeo de nombres comunes
    column_mapping = {
        'Indicador_Nombre': 'indicador_nombre',
        'Indicador Nombre': 'indicador_nombre', 
        'Indicador': 'indicador_nombre',
        'Dimensión': 'dimension',
        'Dimension': 'dimension',
        'Unidad_Medida': 'unidad_medida',
        'Unidad Medida': 'unidad_medida',
        'Unidad': 'unidad_medida',
        'Tipo_Medida': 'tipo_medida',
        'Tipo Medida': 'tipo_medida',
        'Valor': 'valor',
        'Nivel_Territorial': 'nivel_territorial',
        'Nivel Territorial': 'nivel_territorial',
        'ID Localidad': 'id_localidad',
        'ID_Localidad': 'id_localidad',
        'Nombre Localidad': 'nombre_localidad',
        'Nombre_Localidad': 'nombre_localidad',
        'ID_UPZ': 'id_upz',
        'ID UPZ': 'id_upz',
        'Nombre_UPZ': 'nombre_upz',
        'Nombre UPZ': 'nombre_upz',
        'Área Geográfica': 'area_geografica',
        'Area Geografica': 'area_geografica',
        'Año_Inicio': 'ano_inicio',
        'Año Inicio': 'ano_inicio',
        'Ano_Inicio': 'ano_inicio',
        'Año': 'ano_inicio',
        'Ano': 'ano_inicio',
        'Periodicidad': 'periodicidad',
        'Poblacion Base': 'poblacion_base',
        'Población Base': 'poblacion_base',
        'Semaforo': 'semaforo',
        'Semáforo': 'semaforo',
        'Grupo Etario Asociado': 'grupo_etario_asociado',
        'Grupo_Etario_Asociado': 'grupo_etario_asociado',
        'Sexo': 'sexo',
        'Tipo de Unidad': 'tipo_unidad',
        'Tipo_de_Unidad': 'tipo_unidad',
        'Observación': 'observacion',
        'Observacion': 'observacion',
        'Fuente': 'fuente',
        'URL_Fuente (Opcional)': 'url_fuente',
        'URL Fuente': 'url_fuente'
    }
    
    # Aplicar mapeo
    df_renamed = df.rename(columns=column_mapping)
    return df_renamed

# Rutas
@app.get("/", response_class=HTMLResponse)
async def home():
    """Página principal con dashboard"""
    try:
        with open("dashboard_compatible.html","r",encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>API Fecundidad Temprana - Bogotá</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f8fafc; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; }
                h1 { color: #1e3a8a; text-align: center; }
                .alert { background: #dcfce7; border: 1px solid #16a34a; padding: 15px; border-radius: 8px; margin: 20px 0; }
                .links { text-align: center; margin: 30px 0; }
                .links a { display: inline-block; margin: 10px; padding: 10px 20px; background: #2563eb; color: white; text-decoration: none; border-radius: 6px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 API Fecundidad Temprana - Bogotá D.C.</h1>
                <div class="alert">
                    <strong>✅ Estado:</strong> API funcionando correctamente
                </div>
                <p>Análisis integral por territorio, periodo y cohortes para fecundidad temprana en Bogotá.</p>
                <div class="links">
                    <a href="/docs">📚 Documentación API</a>
                    <a href="/health">💚 Estado</a>
                    <a href="/metadatos">📊 Metadatos</a>
                </div>
                <p><strong>Para empezar:</strong></p>
                <ol>
                    <li>Ve a <a href="/docs">/docs</a> para ver toda la documentación</li>
                    <li>Usa el endpoint <strong>POST /upload/excel</strong> para cargar datos</li>
                    <li>Consulta <strong>GET /metadatos</strong> para ver qué datos tienes</li>
                </ol>
            </div>
        </body>
        </html>
        """)

@app.get("/health")
async def health():
    """Health check"""
    try:
        db = SessionLocal()
        try:
            total = db.query(IndicadorFecundidad).count()
            db_status = "connected"
        except Exception:
            total = 0
            db_status = "error"
        finally:
            db.close()
        
        return {
            "status": "healthy",
            "version": "4.0.0",
            "database": db_status,
            "total_registros": total
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/excel")
async def upload_excel(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Cargar datos desde archivo Excel"""
    if not file.filename.endswith(('.xlsx','.xls')):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .xlsx o .xls")
    
    try:
        contents = await file.read()
        logger.info(f"Processing file: {file.filename}")
        
        # Leer Excel
        df = pd.read_excel(io.BytesIO(contents), sheet_name=0)
        logger.info(f"Excel loaded with {len(df)} rows and columns: {list(df.columns)}")
        
        # Normalizar nombres de columnas
        df = normalizar_columnas(df)
        logger.info(f"Columns after normalization: {list(df.columns)}")
        
        # Verificar columnas esenciales
        required_columns = ['indicador_nombre', 'valor']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            available_cols = [col for col in df.columns if 'indicador' in col.lower() or 'valor' in col.lower()]
            raise HTTPException(
                status_code=400, 
                detail=f"Faltan columnas requeridas: {missing_columns}. Columnas disponibles relacionadas: {available_cols}"
            )
        
        # Limpiar tabla existente
        deleted_count = db.query(IndicadorFecundidad).count()
        db.query(IndicadorFecundidad).delete()
        logger.info(f"Deleted {deleted_count} existing records")
        
        # Procesar filas
        registros_exitosos = 0
        errores = 0
        omitidos_sin_valor = 0
        
        # Hash del archivo para tracking
        archivo_hash = hashlib.md5(contents).hexdigest()[:16]
        
        for idx, row in df.iterrows():
            try:
                # Verificar datos mínimos
                indicador_nombre = clean_str(row.get('indicador_nombre'))
                if not indicador_nombre:
                    logger.warning(f"Row {idx}: No indicator name, skipping")
                    continue
                
                valor = clean_float(row.get('valor'), allow_none=True)
                if valor is None:
                    omitidos_sin_valor += 1
                    continue
                
                # Crear registro
                record = IndicadorFecundidad(
                    archivo_hash=archivo_hash,
                    indicador_nombre=indicador_nombre,
                    dimension=clean_str(row.get('dimension')),
                    unidad_medida=clean_str(row.get('unidad_medida'), default='N/A') or 'N/A',
                    tipo_medida=clean_str(row.get('tipo_medida')),
                    valor=valor,
                    nivel_territorial=(clean_str(row.get('nivel_territorial')) or 'LOCALIDAD').upper(),
                    id_localidad=clean_int(row.get('id_localidad')),
                    nombre_localidad=clean_str(row.get('nombre_localidad'), default='SIN LOCALIDAD') or 'SIN LOCALIDAD',
                    id_upz=clean_int(row.get('id_upz')),
                    nombre_upz=clean_str(row.get('nombre_upz')),
                    area_geografica=clean_str(row.get('area_geografica')),
                    ano_inicio=clean_int(row.get('ano_inicio')),
                    periodicidad=clean_str(row.get('periodicidad')),
                    poblacion_base=clean_str(row.get('poblacion_base')),
                    semaforo=clean_str(row.get('semaforo')),
                    grupo_etario_asociado=clean_str(row.get('grupo_etario_asociado')),
                    sexo=clean_str(row.get('sexo')),
                    tipo_unidad=clean_str(row.get('tipo_unidad')),
                    observacion=clean_str(row.get('observacion')),
                    fuente=clean_str(row.get('fuente')),
                    url_fuente=clean_str(row.get('url_fuente'))
                )
                
                db.add(record)
                registros_exitosos += 1
                
                # Commit cada 100 registros para evitar problemas de memoria
                if registros_exitosos % 100 == 0:
                    db.commit()
                    logger.info(f"Committed {registros_exitosos} records")
                    
            except Exception as e:
                errores += 1
                logger.warning(f"Error in row {idx}: {e}")
        
        # Commit final
        db.commit()
        logger.info(f"Upload completed: {registros_exitosos} successful, {errores} errors, {omitidos_sin_valor} skipped")
        
        return {
            "status": "success",
            "mensaje": "Archivo cargado exitosamente",
            "archivo": file.filename,
            "registros_cargados": registros_exitosos,
            "filas_omitidas_sin_valor": omitidos_sin_valor,
            "errores": errores,
            "archivo_hash": archivo_hash
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing Excel file")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {str(e)}")

@app.get("/metadatos")
async def metadatos(db: Session = Depends(get_db)):
    """Obtener metadatos del sistema"""
    try:
        total = db.query(IndicadorFecundidad).count()
        
        if total == 0:
            return {
                "total_registros": 0,
                "mensaje": "No hay datos cargados. Use POST /upload/excel para cargar datos.",
                "indicadores": [],
                "localidades": [],
                "upz": [],
                "anos": [],
                "cohortes": list(COHORTES_VALIDAS)
            }
        
        indicadores = [r[0] for r in db.query(IndicadorFecundidad.indicador_nombre).distinct().all()]
        localidades = [r[0] for r in db.query(IndicadorFecundidad.nombre_localidad).distinct().all()]
        upzs = [r[0] for r in db.query(IndicadorFecundidad.nombre_upz).filter(
            IndicadorFecundidad.nombre_upz.isnot(None)
        ).distinct().all()]
        anos = sorted([r[0] for r in db.query(IndicadorFecundidad.ano_inicio).filter(
            IndicadorFecundidad.ano_inicio.isnot(None)
        ).distinct().all()])
        
        return {
            "total_registros": total,
            "indicadores": sorted(indicadores),
            "localidades": sorted(localidades),
            "upz": sorted([u for u in upzs if u]),
            "anos": anos,
            "cohortes": sorted(list(COHORTES_VALIDAS))
        }
    except Exception as e:
        logger.exception("Error getting metadata")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/estadisticas/generales")
async def estadisticas_generales(ano: Optional[int] = Query(None), db: Session = Depends(get_db)):
    """Estadísticas generales del sistema"""
    try:
        q = db.query(IndicadorFecundidad)
        if ano is not None:
            q = q.filter(IndicadorFecundidad.ano_inicio == ano)
        
        filas = q.all()
        total = len(filas)
        
        if total == 0:
            return {"total_registros": 0, "mensaje": "No hay datos para los filtros especificados"}
        
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
    except Exception as e:
        logger.exception("Error getting general statistics")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/indicadores")
async def listar_indicadores(buscar: Optional[str] = Query(None), db: Session = Depends(get_db)):
    """Listar indicadores disponibles"""
    try:
        q = db.query(
            IndicadorFecundidad.indicador_nombre,
            IndicadorFecundidad.unidad_medida,
            func.count().label("n")
        ).group_by(
            IndicadorFecundidad.indicador_nombre,
            IndicadorFecundidad.unidad_medida
        )
        
        if buscar:
            q = q.filter(IndicadorFecundidad.indicador_nombre.ilike(f"%{buscar}%"))
        
        filas = q.order_by(func.count().desc()).all()
        
        return [
            {
                "indicador": f[0],
                "unidad": f[1],
                "registros": int(f[2])
            } for f in filas
        ]
    except Exception as e:
        logger.exception("Error listing indicators")
        raise HTTPException(status_code=500, detail=str(e))

def _caracterizacion_desde_rows(rows: List[IndicadorFecundidad], nivel: str) -> List[Dict]:
    """Procesar caracterización desde filas de datos"""
    grupos: Dict[str, List[float]] = {}
    for r in rows:
        k = terr_key(r, nivel)
        grupos.setdefault(k, []).append(r.valor)
    
    out = []
    for terr, vals in grupos.items():
        arr = np.array(vals, dtype=float)
        if arr.size == 0:
            continue
        
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=0))
        cv = float((std/mean)*100) if mean != 0 else 0.0
        
        out.append({
            "territorio": terr,
            "n": int(arr.size),
            "promedio": round(mean, 3),
            "mediana": round(float(med), 3),
            "q1": round(float(q1), 3),
            "q3": round(float(q3), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "desv_estandar": round(std, 3),
            "cv_pct": round(cv, 2)
        })
    
    out.sort(key=lambda x: x["promedio"], reverse=True)
    return out

@app.get("/caracterizacion")
async def caracterizacion(
    indicador: str = Query(...),
    nivel: str = Query("LOCALIDAD"),
    ano: Optional[int] = Query(None),
    cohorte: Optional[str] = Query(None, description="10-14 o 15-19"),
    db: Session = Depends(get_db)
):
    """Caracterización territorial de indicadores"""
    if nivel.upper() not in {"LOCALIDAD", "UPZ"}:
        raise HTTPException(status_code=400, detail="nivel debe ser LOCALIDAD o UPZ")
    
    try:
        q = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador)
        if ano is not None:
            q = q.filter(IndicadorFecundidad.ano_inicio == ano)
        
        rows = q.all()
        rows = filtrar_por_cohorte(rows, cohorte)
        
        if not rows:
            return {"mensaje": "Sin datos para los filtros especificados."}
        
        datos = _caracterizacion_desde_rows(rows, nivel)
        
        return {
            "indicador": indicador,
            "nivel": nivel.upper(),
            "ano": ano,
            "cohorte": cohorte,
            "total_territorios": len(datos),
            "resumen": {
                "promedio_general": round(float(np.mean([d["promedio"] for d in datos])), 3),
                "n_total": int(sum(d["n"] for d in datos))
            },
            "datos": datos
        }
    except Exception as e:
        logger.exception("Error in caracterizacion")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analisis/asociacion")
async def asociacion(
    indicador_objetivo: str = Query(...),
    nivel: str = Query("LOCALIDAD"),
    ano: Optional[int] = Query(None),
    cohorte: Optional[str] = Query(None, description="10-14 o 15-19"),
    top: int = Query(10, ge=3, le=50),
    db: Session = Depends(get_db)
):
    """Análisis de asociación entre indicadores"""
    if nivel.upper() not in {"LOCALIDAD", "UPZ"}:
        raise HTTPException(status_code=400, detail="nivel debe ser LOCALIDAD o UPZ")

    try:
        # Datos del indicador objetivo (Y)
        qy = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador_objetivo)
        if ano is not None:
            qy = qy.filter(IndicadorFecundidad.ano_inicio == ano)
        
        y_rows = filtrar_por_cohorte(qy.all(), cohorte)
        if not y_rows:
            return {"mensaje": "No se encontraron datos para el indicador objetivo."}
        
        # Agrupar Y por territorio
        y_map: Dict[str, List[float]] = {}
        for r in y_rows:
            y_map.setdefault(terr_key(r, nivel), []).append(r.valor)
        
        y_mean = {k: float(np.mean(v)) for k, v in y_map.items()}
        
        # Otros indicadores (X)
        otros = [r[0] for r in db.query(IndicadorFecundidad.indicador_nombre).filter(
            IndicadorFecundidad.indicador_nombre != indicador_objetivo
        ).distinct().all()]
        
        resultados = []
        
        for ind in otros:
            qx = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == ind)
            if ano is not None:
                qx = qx.filter(IndicadorFecundidad.ano_inicio == ano)
            
            x_rows = filtrar_por_cohorte(qx.all(), cohorte)
            if not x_rows:
                continue
            
            # Agrupar X por territorio
            x_map: Dict[str, List[float]] = {}
            for r in x_rows:
                x_map.setdefault(terr_key(r, nivel), []).append(r.valor)
            
            # Territorios comunes
            comunes = set(x_map.keys()) & set(y_mean.keys())
            if len(comunes) < 3:
                continue
            
            x = [float(np.mean(x_map[t])) for t in comunes]
            y = [y_mean[t] for t in comunes]
            
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            
            r_p, p_p = stats.pearsonr(x, y)
            r_s, p_s = stats.spearmanr(x, y)
            
            resultados.append({
                "indicador_comparado": ind,
                "territorios": len(comunes),
                "pearson_r": round(float(r_p), 3),
                "pearson_p": round(float(p_p), 4),
                "spearman_rho": round(float(r_s), 3),
                "spearman_p": round(float(p_s), 4)
            })
        
        resultados.sort(key=lambda d: abs(d["pearson_r"]), reverse=True)
        
        return {
            "indicador_objetivo": indicador_objetivo,
            "nivel": nivel.upper(),
            "ano": ano,
            "cohorte": cohorte,
            "comparaciones": resultados[:top]
        }
    except Exception as e:
        logger.exception("Error in asociacion analysis")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/datos/series")
async def serie_temporal(
    indicador: str = Query(...),
    territorio: str = Query(...),
    nivel: str = Query("LOCALIDAD"),
    cohorte: Optional[str] = Query(None, description="10-14 o 15-19"),
    db: Session = Depends(get_db)
):
    """Serie temporal de un indicador en un territorio"""
    try:
        q = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador)
        
        if nivel.upper() == "LOCALIDAD":
            q = q.filter(IndicadorFecundidad.nombre_localidad == territorio)
        else:
            q = q.filter(IndicadorFecundidad.nombre_upz == territorio)
        
        rows = q.filter(IndicadorFecundidad.ano_inicio.isnot(None)).order_by(
            IndicadorFecundidad.ano_inicio.asc()
        ).all()
        
        rows = filtrar_por_cohorte(rows, cohorte)
        
        if not rows:
            return {"mensaje": "Sin datos para esos filtros."}
        
        return {
            "indicador": indicador,
            "nivel": nivel.upper(),
            "territorio": territorio,
            "cohorte": cohorte,
            "serie": [
                {"ano": r.ano_inicio, "valor": r.valor}
                for r in rows if r.ano_inicio is not None
            ]
        }
    except Exception as e:
        logger.exception("Error getting time series")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/brechas/cohortes")
async def brechas_cohortes(
    indicador: str = Query(...),
    nivel: str = Query("LOCALIDAD"),
    ano: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    """Calcula brecha 15-19 menos 10-14 por territorio"""
    if nivel.upper() not in {"LOCALIDAD", "UPZ"}:
        raise HTTPException(status_code=400, detail="nivel debe ser LOCALIDAD o UPZ")
    
    try:
        q = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador)
        if ano is not None:
            q = q.filter(IndicadorFecundidad.ano_inicio == ano)
        
        rows = q.all()
        if not rows:
            return {"mensaje": "Sin datos para esos filtros."}
        
        m10: Dict[str, List[float]] = {}
        m15: Dict[str, List[float]] = {}
        
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
            
            datos.append({
                "territorio": t,
                "prom_10_14": round(v10, 3),
                "prom_15_19": round(v15, 3),
                "brecha_abs": round(delta, 3),
                "brecha_rel": round(ratio, 3) if ratio is not None else None
            })
        
        datos.sort(key=lambda d: d["brecha_abs"], reverse=True)
        
        return {
            "indicador": indicador,
            "nivel": nivel.upper(),
            "ano": ano,
            "datos": datos
        }
    except Exception as e:
        logger.exception("Error calculating cohort gaps")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
