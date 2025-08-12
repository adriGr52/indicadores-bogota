from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime
import io
import json
import os
from enum import Enum

# =====================================================
# CONFIGURACIÓN DE BASE DE DATOS
# =====================================================

# Configuración para funcionar tanto en local como en producción
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./indicadores_bogota.db")

# Railway/Heroku a veces usa postgres:// pero SQLAlchemy necesita postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

SQLALCHEMY_DATABASE_URL = DATABASE_URL

# Configurar engine según el tipo de base de datos
if "sqlite" in SQLALCHEMY_DATABASE_URL:
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
else:
    # Para PostgreSQL en producción
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# =====================================================
# MODELOS DE BASE DE DATOS
# =====================================================

class IndicadorDB(Base):
    __tablename__ = "indicadores"
    
    id = Column(Integer, primary_key=True, index=True)
    indicador_nombre = Column(String, index=True)
    dimension = Column(String, index=True)
    sub_dimension = Column(String, nullable=True)
    unidad_medida = Column(String)
    tipo_medida = Column(String)
    valor = Column(Float)
    nivel_territorial = Column(String)
    id_localidad = Column(Integer, nullable=True)
    nombre_localidad = Column(String, nullable=True)
    id_upz = Column(String, nullable=True)
    nombre_upz = Column(String, nullable=True)
    area_geografica = Column(String, nullable=True)
    año_inicio = Column(Integer, nullable=True)
    año_fin = Column(Integer, nullable=True)
    año_referencia = Column(String, nullable=True)
    periodicidad = Column(String, nullable=True)
    poblacion_base = Column(String, nullable=True)
    pertenencia_etnica = Column(String, nullable=True)
    semaforo = Column(String, nullable=True)
    grupo_etario = Column(String, nullable=True)
    sexo = Column(String, nullable=True)
    tipo_unidad_observacion = Column(String, nullable=True)
    fuente = Column(String, nullable=True)
    url_fuente = Column(String, nullable=True)
    observaciones = Column(Text, nullable=True)
    archivo_origen = Column(String, nullable=True)
    fecha_carga = Column(DateTime, default=datetime.utcnow)

class EstadisticasArchivo(Base):
    __tablename__ = "estadisticas_archivo"
    
    id = Column(Integer, primary_key=True, index=True)
    archivo = Column(String)
    filas = Column(Integer)
    columnas = Column(Integer)
    indicador = Column(String)
    fecha_procesamiento = Column(DateTime, default=datetime.utcnow)

# =====================================================
# MODELOS PYDANTIC
# =====================================================

class DimensionEnum(str, Enum):
    demografia = "Demografía"
    seguridad = "Seguridad"
    educacion = "Educación"
    salud = "Salud"
    economia = "Economía"

class IndicadorResponse(BaseModel):
    id: int
    indicador_nombre: str
    dimension: str
    valor: float
    nombre_localidad: Optional[str]
    año_inicio: Optional[int]
    fecha_carga: datetime
    
    class Config:
        from_attributes = True

class CaracterizacionResponse(BaseModel):
    total_registros: int
    indicadores_unicos: int
    dimensiones: List[str]
    localidades: List[str]
    rango_años: Dict[str, Any]
    estadisticas_por_indicador: List[Dict[str, Any]]
    estadisticas_por_localidad: List[Dict[str, Any]]
    estadisticas_por_dimension: List[Dict[str, Any]]

class FiltroIndicadores(BaseModel):
    indicador_nombre: Optional[str] = None
    dimension: Optional[str] = None
    localidad: Optional[str] = None
    año_desde: Optional[int] = None
    año_hasta: Optional[int] = None
    valor_min: Optional[float] = None
    valor_max: Optional[float] = None

# =====================================================
# CONFIGURACIÓN DE FASTAPI
# =====================================================

app = FastAPI(
    title="API Indicadores Sociales Bogotá",
    description="API para carga y caracterización de indicadores sociales de Bogotá D.C.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear tablas
Base.metadata.create_all(bind=engine)

# Configurar archivos estáticos y rutas del dashboard
try:
    app.mount("/static", StaticFiles(directory="."), name="static")
except:
    pass  # En caso de que el directorio no exista

@app.get("/dashboard")
async def get_dashboard():
    """Servir el dashboard principal"""
    return FileResponse('dashboard_compatible.html')

@app.get("/dashboard/")
async def get_dashboard_slash():
    """Servir el dashboard principal con slash"""
    return FileResponse('dashboard_compartible.html')

# =====================================================
# DEPENDENCIAS
# =====================================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =====================================================
# UTILIDADES
# =====================================================

def procesar_excel_consolidado(file_content: bytes) -> Dict[str, pd.DataFrame]:
    """Procesa el archivo Excel con múltiples hojas"""
    try:
        excel_data = pd.read_excel(io.BytesIO(file_content), sheet_name=None)
        return excel_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando archivo Excel: {str(e)}")

def limpiar_datos_indicador(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y valida los datos del indicador"""
    # Limpiar nombres de columnas
    df.columns = df.columns.str.strip()
    
    # Convertir valores numéricos
    if 'Valor' in df.columns:
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce')
    
    # Limpiar años
    for col in ['Año_Inicio', 'año referencia']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Eliminar filas completamente vacías
    df = df.dropna(how='all')
    
    return df

# =====================================================
# ENDPOINTS PRINCIPALES
# =====================================================

@app.post("/upload/consolidado")
async def cargar_archivo_consolidado(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Carga el archivo consolidado de indicadores sociales"""
    
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos Excel (.xlsx, .xls)")
    
    try:
        # Leer archivo
        content = await file.read()
        excel_data = procesar_excel_consolidado(content)
        
        # Limpiar datos existentes para evitar duplicaciones
        db.query(IndicadorDB).delete()
        db.query(EstadisticasArchivo).delete()
        db.commit()  # Asegurar que se borren los datos
        
        total_registros = 0
        
        # Procesar hoja principal de datos
        if 'Datos_Consolidados' in excel_data:
            df_principal = limpiar_datos_indicador(excel_data['Datos_Consolidados'])
            
            for _, row in df_principal.iterrows():
                indicador = IndicadorDB(
                    indicador_nombre=row.get('Indicador_Nombre'),
                    dimension=row.get('Dimensión'),
                    sub_dimension=row.get('Sub_Dimensión (Opcional)'),
                    unidad_medida=row.get('Unidad_Medida'),
                    tipo_medida=row.get('Tipo_Medida'),
                    valor=row.get('Valor'),
                    nivel_territorial=row.get('Nivel_Territorial'),
                    id_localidad=row.get('ID Localidad'),
                    nombre_localidad=row.get('Nombre Localidad'),
                    id_upz=row.get('ID_UPZ'),
                    nombre_upz=row.get('Nombre_UPZ'),
                    area_geografica=row.get('Área Geográfica'),
                    año_inicio=row.get('Año_Inicio'),
                    año_fin=row.get('Año_Fin (Si es Rango)'),
                    año_referencia=row.get('año referencia'),
                    periodicidad=row.get('Periodicidad'),
                    poblacion_base=row.get('Poblacion Base'),
                    pertenencia_etnica=row.get('Pertenencia Étnica'),
                    semaforo=row.get('Semaforo'),
                    grupo_etario=row.get('Grupo Etario Asociado'),
                    sexo=row.get('Sexo'),
                    tipo_unidad_observacion=row.get('Tipo de Unidad Observación'),
                    fuente=row.get('Fuente'),
                    url_fuente=row.get('URL_Fuente (Opcional)'),
                    observaciones=row.get('Observaciones_Adicionales'),
                    archivo_origen=row.get('Archivo_Origen')
                )
                db.add(indicador)
                total_registros += 1
        
        # Procesar estadísticas por archivo
        if 'Stats_por_Archivo' in excel_data:
            df_stats = excel_data['Stats_por_Archivo']
            for _, row in df_stats.iterrows():
                if pd.notna(row.iloc[0]):  # Verificar que no sea la fila de headers
                    stats = EstadisticasArchivo(
                        archivo=row.iloc[0],
                        filas=row.iloc[1] if pd.notna(row.iloc[1]) else 0,
                        columnas=row.iloc[2] if pd.notna(row.iloc[2]) else 0,
                        indicador=row.iloc[3] if pd.notna(row.iloc[3]) else ""
                    )
                    db.add(stats)
        
        db.commit()
        
        return {
            "mensaje": "Archivo cargado exitosamente",
            "registros_procesados": total_registros,
            "hojas_procesadas": list(excel_data.keys()),
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {str(e)}")

@app.get("/caracterizacion/completa", response_model=CaracterizacionResponse)
async def obtener_caracterizacion_completa(db: Session = Depends(get_db)):
    """Obtiene una caracterización completa de todos los indicadores"""
    
    # Estadísticas básicas
    total_registros = db.query(IndicadorDB).count()
    indicadores_unicos = db.query(IndicadorDB.indicador_nombre).distinct().count()
    
    # Listas únicas
    dimensiones = [d[0] for d in db.query(IndicadorDB.dimension).distinct().all() if d[0]]
    localidades = [l[0] for l in db.query(IndicadorDB.nombre_localidad).distinct().all() if l[0]]
    
    # Rango de años
    año_min = db.query(func.min(IndicadorDB.año_inicio)).scalar()
    año_max = db.query(func.max(IndicadorDB.año_inicio)).scalar()
    
    # Estadísticas por indicador
    stats_indicador = db.query(
        IndicadorDB.indicador_nombre,
        func.count(IndicadorDB.id).label('registros'),
        func.avg(IndicadorDB.valor).label('valor_promedio'),
        func.min(IndicadorDB.valor).label('valor_min'),
        func.max(IndicadorDB.valor).label('valor_max')
    ).group_by(IndicadorDB.indicador_nombre).all()
    
    # Estadísticas por localidad
    stats_localidad = db.query(
        IndicadorDB.nombre_localidad,
        func.count(IndicadorDB.id).label('registros'),
        func.count(IndicadorDB.indicador_nombre.distinct()).label('indicadores_diferentes')
    ).filter(IndicadorDB.nombre_localidad.isnot(None)).group_by(IndicadorDB.nombre_localidad).all()
    
    # Estadísticas por dimensión
    stats_dimension = db.query(
        IndicadorDB.dimension,
        func.count(IndicadorDB.id).label('registros'),
        func.count(IndicadorDB.indicador_nombre.distinct()).label('indicadores_diferentes')
    ).filter(IndicadorDB.dimension.isnot(None)).group_by(IndicadorDB.dimension).all()
    
    return CaracterizacionResponse(
        total_registros=total_registros,
        indicadores_unicos=indicadores_unicos,
        dimensiones=dimensiones,
        localidades=localidades,
        rango_años={"desde": año_min, "hasta": año_max},
        estadisticas_por_indicador=[
            {
                "indicador": s.indicador_nombre,
                "registros": s.registros,
                "valor_promedio": round(s.valor_promedio, 2) if s.valor_promedio else None,
                "valor_min": s.valor_min,
                "valor_max": s.valor_max
            } for s in stats_indicador
        ],
        estadisticas_por_localidad=[
            {
                "localidad": s.nombre_localidad,
                "registros": s.registros,
                "indicadores_diferentes": s.indicadores_diferentes
            } for s in stats_localidad
        ],
        estadisticas_por_dimension=[
            {
                "dimension": s.dimension,
                "registros": s.registros,
                "indicadores_diferentes": s.indicadores_diferentes
            } for s in stats_dimension
        ]
    )

@app.post("/indicadores/filtrar")
async def filtrar_indicadores(
    filtros: FiltroIndicadores,
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0),
    db: Session = Depends(get_db)
) -> List[IndicadorResponse]:
    """Filtra indicadores según criterios específicos"""
    
    query = db.query(IndicadorDB)
    
    if filtros.indicador_nombre:
        query = query.filter(IndicadorDB.indicador_nombre.ilike(f"%{filtros.indicador_nombre}%"))
    
    if filtros.dimension:
        query = query.filter(IndicadorDB.dimension.ilike(f"%{filtros.dimension}%"))
    
    if filtros.localidad:
        query = query.filter(IndicadorDB.nombre_localidad.ilike(f"%{filtros.localidad}%"))
    
    if filtros.año_desde:
        query = query.filter(IndicadorDB.año_inicio >= filtros.año_desde)
    
    if filtros.año_hasta:
        query = query.filter(IndicadorDB.año_inicio <= filtros.año_hasta)
    
    if filtros.valor_min is not None:
        query = query.filter(IndicadorDB.valor >= filtros.valor_min)
    
    if filtros.valor_max is not None:
        query = query.filter(IndicadorDB.valor <= filtros.valor_max)
    
    resultados = query.offset(offset).limit(limit).all()
    
    return [IndicadorResponse.from_orm(r) for r in resultados]

@app.get("/indicadores/{indicador_nombre}/analisis")
async def analizar_indicador_especifico(
    indicador_nombre: str,
    db: Session = Depends(get_db)
):
    """Análisis detallado de un indicador específico"""
    
    # Verificar que existe el indicador
    existe = db.query(IndicadorDB).filter(
        IndicadorDB.indicador_nombre.ilike(f"%{indicador_nombre}%")
    ).first()
    
    if not existe:
        raise HTTPException(status_code=404, detail="Indicador no encontrado")
    
    # Estadísticas básicas
    stats = db.query(
        func.count(IndicadorDB.id).label('total_registros'),
        func.avg(IndicadorDB.valor).label('valor_promedio'),
        func.min(IndicadorDB.valor).label('valor_min'),
        func.max(IndicadorDB.valor).label('valor_max'),
        func.min(IndicadorDB.año_inicio).label('año_min'),
        func.max(IndicadorDB.año_inicio).label('año_max')
    ).filter(IndicadorDB.indicador_nombre.ilike(f"%{indicador_nombre}%")).first()
    
    # Distribución por localidad
    por_localidad = db.query(
        IndicadorDB.nombre_localidad,
        func.avg(IndicadorDB.valor).label('valor_promedio'),
        func.count(IndicadorDB.id).label('registros')
    ).filter(
        IndicadorDB.indicador_nombre.ilike(f"%{indicador_nombre}%"),
        IndicadorDB.nombre_localidad.isnot(None)
    ).group_by(IndicadorDB.nombre_localidad).all()
    
    # Evolución temporal
    temporal = db.query(
        IndicadorDB.año_inicio,
        func.avg(IndicadorDB.valor).label('valor_promedio'),
        func.count(IndicadorDB.id).label('registros')
    ).filter(
        IndicadorDB.indicador_nombre.ilike(f"%{indicador_nombre}%"),
        IndicadorDB.año_inicio.isnot(None)
    ).group_by(IndicadorDB.año_inicio).order_by(IndicadorDB.año_inicio).all()
    
    return {
        "indicador": indicador_nombre,
        "estadisticas_generales": {
            "total_registros": stats.total_registros,
            "valor_promedio": round(stats.valor_promedio, 2) if stats.valor_promedio else None,
            "valor_min": stats.valor_min,
            "valor_max": stats.valor_max,
            "rango_años": f"{stats.año_min} - {stats.año_max}"
        },
        "distribucion_por_localidad": [
            {
                "localidad": loc.nombre_localidad,
                "valor_promedio": round(loc.valor_promedio, 2) if loc.valor_promedio else None,
                "registros": loc.registros
            } for loc in por_localidad
        ],
        "evolucion_temporal": [
            {
                "año": temp.año_inicio,
                "valor_promedio": round(temp.valor_promedio, 2) if temp.valor_promedio else None,
                "registros": temp.registros
            } for temp in temporal
        ]
    }

@app.get("/localidades/{localidad}/resumen")
async def resumen_localidad(
    localidad: str,
    db: Session = Depends(get_db)
):
    """Resumen de todos los indicadores para una localidad específica"""
    
    # Verificar que existe la localidad
    existe = db.query(IndicadorDB).filter(
        IndicadorDB.nombre_localidad.ilike(f"%{localidad}%")
    ).first()
    
    if not existe:
        raise HTTPException(status_code=404, detail="Localidad no encontrada")
    
    # Resumen por indicador
    por_indicador = db.query(
        IndicadorDB.indicador_nombre,
        IndicadorDB.dimension,
        func.avg(IndicadorDB.valor).label('valor_promedio'),
        func.count(IndicadorDB.id).label('registros'),
        func.max(IndicadorDB.año_inicio).label('año_mas_reciente')
    ).filter(
        IndicadorDB.nombre_localidad.ilike(f"%{localidad}%")
    ).group_by(IndicadorDB.indicador_nombre, IndicadorDB.dimension).all()
    
    # Estadísticas generales
    total_registros = db.query(IndicadorDB).filter(
        IndicadorDB.nombre_localidad.ilike(f"%{localidad}%")
    ).count()
    
    indicadores_unicos = db.query(IndicadorDB.indicador_nombre).filter(
        IndicadorDB.nombre_localidad.ilike(f"%{localidad}%")
    ).distinct().count()
    
    return {
        "localidad": localidad,
        "resumen_general": {
            "total_registros": total_registros,
            "indicadores_disponibles": indicadores_unicos
        },
        "indicadores_detalle": [
            {
                "indicador": ind.indicador_nombre,
                "dimension": ind.dimension,
                "valor_promedio": round(ind.valor_promedio, 2) if ind.valor_promedio else None,
                "registros": ind.registros,
                "año_mas_reciente": ind.año_mas_reciente
            } for ind in por_indicador
        ]
    }

@app.get("/dimensiones")
async def listar_dimensiones(db: Session = Depends(get_db)):
    """Lista todas las dimensiones disponibles con sus estadísticas"""
    
    dimensiones = db.query(
        IndicadorDB.dimension,
        func.count(IndicadorDB.id).label('total_registros'),
        func.count(IndicadorDB.indicador_nombre.distinct()).label('indicadores_unicos'),
        func.count(IndicadorDB.nombre_localidad.distinct()).label('localidades_cubiertas')
    ).filter(
        IndicadorDB.dimension.isnot(None)
    ).group_by(IndicadorDB.dimension).all()
    
    return [
        {
            "dimension": dim.dimension,
            "total_registros": dim.total_registros,
            "indicadores_unicos": dim.indicadores_unicos,
            "localidades_cubiertas": dim.localidades_cubiertas
        } for dim in dimensiones
    ]

@app.get("/estadisticas/archivos")
async def estadisticas_archivos(db: Session = Depends(get_db)):
    """Estadísticas de los archivos procesados"""
    
    archivos = db.query(EstadisticasArchivo).all()
    
    return [
        {
            "archivo": arch.archivo,
            "filas": arch.filas,
            "columnas": arch.columnas,
            "indicador": arch.indicador,
            "fecha_procesamiento": arch.fecha_procesamiento
        } for arch in archivos
    ]

# =====================================================
# ENDPOINTS DE SALUD Y INFORMACIÓN
# =====================================================

@app.get("/")
async def root():
    return {
        "mensaje": "API Indicadores Sociales Bogotá D.C.",
        "version": "1.0.0",
        "documentacion": "/docs"
    }

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    try:
        total_registros = db.query(IndicadorDB).count()
        return {
            "status": "healthy",
            "total_indicadores": total_registros,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)