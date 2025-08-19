
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Index, func
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import io, os, json, logging, re
from datetime import datetime
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///fecundidad_temprana.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, echo=False, future=True)
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

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(
    title="API Determinantes de Fecundidad Temprana - Bogotá D.C.",
    description="Objetivo 1: Caracterizar los indicadores determinantes y de fecundidad por territorio y periodo.",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Utilidades ----
def extraer_grupo_edad(indicador_nombre: str, grupo_etario: Optional[str]) -> Optional[str]:
    txt = (indicador_nombre or "").lower() + " " + (grupo_etario or "").lower()
    if re.search(r"10\D*14", txt) or re.search(r"10\s*-\s*14", txt):
        return "10-14"
    if re.search(r"15\D*19", txt) or re.search(r"15\s*-\s*19", txt):
        return "15-19"
    return None

def calcular_nivel_riesgo(valor: float, grupo_edad: Optional[str]) -> str:
    if grupo_edad == "10-14":
        if valor >= 5.0: return "CRÍTICO"
        elif valor >= 2.0: return "ALTO"
        elif valor >= 0.5: return "MODERADO"
        else: return "BAJO"
    elif grupo_edad == "15-19":
        if valor >= 80.0: return "CRÍTICO"
        elif valor >= 60.0: return "ALTO"
        elif valor >= 40.0: return "MODERADO"
        else: return "BAJO"
    return "N/A"

def to_int_safe(x):
    try:
        return int(x)
    except Exception:
        return None

# ---- Rutas ----
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    try:
        with open("dashboard_compatible.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>API Fecundidad - Ver /docs</h1>")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "API Fecundidad Temprana"}

@app.post("/upload/excel")
async def upload_excel(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Formato no válido. Use .xlsx o .xls")

    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents), sheet_name=0)
        logger.info(f"Archivo: {file.filename} -> {len(df)} filas, {len(df.columns)} columnas")

        # Normalizar nombres de columnas esperadas
        cols = {c: c for c in df.columns}
        # Si existe una columna combinada 'Tipo de Unidad Observación', la mapeamos a 'Observación'
        if 'Tipo de Unidad Observación' in df.columns and 'Observación' not in df.columns:
            df['Observación'] = df['Tipo de Unidad Observación']
            cols['Tipo de Unidad Observación'] = 'Observación'
        df = df.rename(columns=cols)

        # Limpiar tabla
        db.query(IndicadorFecundidad).delete()

        registros = 0
        errores = 0
        for _, row in df.iterrows():
            try:
                indicador_nombre = str(row.get('Indicador_Nombre', '')).strip()
                if not indicador_nombre:
                    continue

                registro = IndicadorFecundidad(
                    archivo_hash=str(row.get('archivo_hash', '')),
                    indicador_nombre=indicador_nombre,
                    dimension=str(row.get('Dimensión', '')),
                    unidad_medida=str(row.get('Unidad_Medida', '')) or 'N/A',
                    tipo_medida=str(row.get('Tipo_Medida', '')),
                    valor=float(row.get('Valor', 0)) if pd.notna(row.get('Valor')) else 0.0,
                    nivel_territorial=str(row.get('Nivel_Territorial', '')).upper() or 'LOCALIDAD',
                    id_localidad=to_int_safe(row.get('ID Localidad', None)),
                    nombre_localidad=str(row.get('Nombre Localidad', '')).strip() or 'SIN LOCALIDAD',
                    id_upz=to_int_safe(row.get('ID_UPZ', None)),
                    nombre_upz=str(row.get('Nombre_UPZ', '')).strip() if pd.notna(row.get('Nombre_UPZ')) else None,
                    area_geografica=str(row.get('Área Geográfica', '')),
                    año_inicio=to_int_safe(row.get('Año_Inicio', None)),
                    periodicidad=str(row.get('Periodicidad', '')),
                    poblacion_base=str(row.get('Poblacion Base', '')),
                    semaforo=str(row.get('Semaforo', '')),
                    grupo_etario_asociado=str(row.get('Grupo Etario Asociado', '')),
                    sexo=str(row.get('Sexo', '')),
                    tipo_unidad=str(row.get('Tipo de Unidad', '')) if 'Tipo de Unidad' in df.columns else None,
                    observacion=str(row.get('Observación', '')) if 'Observación' in df.columns else None,
                    fuente=str(row.get('Fuente', '')),
                    url_fuente=str(row.get('URL_Fuente (Opcional)', ''))
                )
                db.add(registro)
                registros += 1
            except Exception as e:
                errores += 1
                logger.warning(f"Fila con error: {e}")
        db.commit()

        return {
            "status": "success",
            "mensaje": "Carga completada",
            "registros_cargados": registros,
            "total_filas": int(len(df)),
            "errores": errores
        }
    except Exception as e:
        logger.exception("Error procesando archivo")
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {str(e)}")

@app.get("/metadatos")
async def metadatos(db: Session = Depends(get_db)):
    total = db.query(IndicadorFecundidad).count()
    if total == 0:
        return {"mensaje": "No hay datos cargados."}
    indicadores = [r[0] for r in db.query(IndicadorFecundidad.indicador_nombre).distinct().all()]
    localidades = [r[0] for r in db.query(IndicadorFecundidad.nombre_localidad).distinct().all()]
    upzs = [r[0] for r in db.query(IndicadorFecundidad.nombre_upz).filter(IndicadorFecundidad.nombre_upz.isnot(None)).distinct().all()]
    años = sorted([r[0] for r in db.query(IndicadorFecundidad.año_inicio).filter(IndicadorFecundidad.año_inicio.isnot(None)).distinct().all()])
    return {
        "total_registros": total,
        "indicadores": sorted(indicadores),
        "localidades": sorted(localidades),
        "upz": sorted([u for u in upzs if u]),
        "años": años
    }

@app.get("/indicadores")
async def listar_indicadores(buscar: Optional[str] = Query(None), db: Session = Depends(get_db)):
    q = db.query(IndicadorFecundidad.indicador_nombre, IndicadorFecundidad.unidad_medida, func.count().label("n"))\
        .group_by(IndicadorFecundidad.indicador_nombre, IndicadorFecundidad.unidad_medida)
    if buscar:
        q = q.filter(IndicadorFecundidad.indicador_nombre.ilike(f"%{buscar}%"))
    filas = q.order_by(func.count().desc()).all()
    return [{"indicador": f[0], "unidad": f[1], "registros": int(f[2])} for f in filas]

@app.get("/caracterizacion")
async def caracterizacion(
    indicador: str = Query(..., description="Nombre exacto del indicador"),
    nivel: str = Query("LOCALIDAD", description="LOCALIDAD o UPZ"),
    año: Optional[int] = Query(None, description="Año específico"),
    db: Session = Depends(get_db)
):
    if nivel.upper() not in {"LOCALIDAD", "UPZ"}:
        raise HTTPException(status_code=400, detail="nivel debe ser LOCALIDAD o UPZ")
    base = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador)
    if año is not None:
        base = base.filter(IndicadorFecundidad.año_inicio == año)
    data = base.all()
    if not data:
        return {"mensaje": "Sin datos para los filtros."}

    grupos: Dict[str, list] = {}
    for r in data:
        key = r.nombre_localidad if nivel.upper() == "LOCALIDAD" else (r.nombre_upz or "SIN UPZ")
        grupos.setdefault(key, []).append(r.valor)

    out = []
    for terr, vals in grupos.items():
        arr = np.array(vals, dtype=float)
        if arr.size == 0:
            continue
        q1 = float(np.percentile(arr, 25))
        med = float(np.percentile(arr, 50))
        q3 = float(np.percentile(arr, 75))
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=0))
        cv = float((std/mean)*100) if mean != 0 else 0.0
        out.append({
            "territorio": terr,
            "n": int(arr.size),
            "promedio": round(mean, 3),
            "mediana": round(med, 3),
            "q1": round(q1, 3),
            "q3": round(q3, 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "desv_estandar": round(std, 3),
            "cv_pct": round(cv, 2)
        })

    out.sort(key=lambda x: x["promedio"], reverse=True)
    return {
        "indicador": indicador,
        "nivel": nivel.upper(),
        "año": año,
        "total_territorios": len(out),
        "resumen": {
            "promedio_general": round(float(np.mean([d["promedio"] for d in out])), 3),
            "n_total": int(sum(d["n"] for d in out))
        },
        "datos": out
    }

@app.get("/analisis/asociacion")
async def asociacion(
    indicador_objetivo: str = Query(..., description="Indicador de fecundidad objetivo"),
    nivel: str = Query("LOCALIDAD", description="LOCALIDAD o UPZ"),
    año: Optional[int] = Query(None, description="Año específico"),
    top: int = Query(10, ge=3, le=50),
    db: Session = Depends(get_db)
):
    if nivel.upper() not in {"LOCALIDAD", "UPZ"}:
        raise HTTPException(status_code=400, detail="nivel debe ser LOCALIDAD o UPZ")

    # Construir series territoriales del indicador objetivo
    base = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador_objetivo)
    if año is not None:
        base = base.filter(IndicadorFecundidad.año_inicio == año)
    filas_obj = base.all()
    if not filas_obj:
        return {"mensaje": "No se encontraron datos para el indicador objetivo con esos filtros."}

    def terr_key(r): return r.nombre_localidad if nivel.upper() == "LOCALIDAD" else (r.nombre_upz or "SIN UPZ")
    serie_obj: Dict[str, list] = {}
    for r in filas_obj:
        serie_obj.setdefault(terr_key(r), []).append(r.valor)
    # Usar promedio por territorio como valor representativo
    y_map = {k: float(np.mean(v)) for k,v in serie_obj.items()}

    # Listar todos los demás indicadores presentes (excepto el objetivo)
    otros = [r[0] for r in db.query(IndicadorFecundidad.indicador_nombre).filter(IndicadorFecundidad.indicador_nombre != indicador_objetivo).distinct().all()]

    resultados = []
    for ind in otros:
        q = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == ind)
        if año is not None:
            q = q.filter(IndicadorFecundidad.año_inicio == año)
        filas = q.all()
        if not filas:
            continue
        x_map: Dict[str, list] = {}
        for r in filas:
            x_map.setdefault(terr_key(r), []).append(r.valor)
        # Territorios comunes
        comunes = set(x_map.keys()) & set(y_map.keys())
        if len(comunes) < 3:
            continue
        x = [float(np.mean(x_map[t])) for t in comunes]
        y = [y_map[t] for t in comunes]
        # Si todas las x o y son constantes, saltar
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

    # Ordenar por |r| de Pearson descendente
    resultados.sort(key=lambda d: abs(d["pearson_r"]), reverse=True)
    return {
        "indicador_objetivo": indicador_objetivo,
        "nivel": nivel.upper(),
        "año": año,
        "comparaciones": resultados[:top]
    }

@app.get("/datos/series")
async def serie_temporal(
    indicador: str = Query(...),
    territorio: str = Query(..., description="Nombre de localidad o UPZ exacto"),
    nivel: str = Query("LOCALIDAD", description="LOCALIDAD o UPZ"),
    db: Session = Depends(get_db)
):
    q = db.query(IndicadorFecundidad).filter(IndicadorFecundidad.indicador_nombre == indicador)
    if nivel.upper() == "LOCALIDAD":
        q = q.filter(IndicadorFecundidad.nombre_localidad == territorio)
    else:
        q = q.filter(IndicadorFecundidad.nombre_upz == territorio)
    filas = q.filter(IndicadorFecundidad.año_inicio.isnot(None)).order_by(IndicadorFecundidad.año_inicio.asc()).all()
    if not filas:
        return {"mensaje": "Sin datos para esos filtros."}
    return {
        "indicador": indicador,
        "nivel": nivel.upper(),
        "territorio": territorio,
        "serie": [{"año": f.año_inicio, "valor": f.valor} for f in filas]
    }

@app.get("/estadisticas/generales")
async def estadisticas_generales(db: Session = Depends(get_db)):
    try:
        total = db.query(IndicadorFecundidad).count()
        if total == 0:
            return {"total_registros": 0, "mensaje": "No hay datos cargados."}
        localidades = db.query(IndicadorFecundidad.nombre_localidad).distinct().count()
        upzs = db.query(IndicadorFecundidad.nombre_upz).filter(IndicadorFecundidad.nombre_upz.isnot(None)).distinct().count()
        indicadores = db.query(IndicadorFecundidad.indicador_nombre).distinct().count()
        años = [a[0] for a in db.query(IndicadorFecundidad.año_inicio).filter(IndicadorFecundidad.año_inicio.isnot(None)).distinct().all()]
        return {
            "total_registros": total,
            "localidades_unicas": localidades,
            "upzs_unicas": upzs,
            "indicadores_unicos": indicadores,
            "años_disponibles": sorted(años)
        }
    except Exception as e:
        logger.exception("Error en estadísticas")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
