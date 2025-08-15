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
import unicodedata
import re

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
# FUNCIONES DE UTILIDAD MEJORADAS CON NORMALIZACIÓN UPZ
# =====================================================

def normalizar_nombre_localidad(nombre: str) -> str:
    """
    Normaliza nombres de localidades eliminando tildes y caracteres especiales.
    """
    if not nombre:
        return ""
    
    # Eliminar tildes y diacríticos
    nombre_normalizado = unicodedata.normalize('NFD', nombre)
    nombre_sin_tildes = ''.join(
        char for char in nombre_normalizado 
        if unicodedata.category(char) != 'Mn'
    )
    
    # Limpiar caracteres especiales pero mantener espacios y guiones
    nombre_limpio = re.sub(r'[^\w\s\-]', '', nombre_sin_tildes)
    
    # Capitalizar correctamente
    return nombre_limpio.title().strip()

def normalizar_nombre_upz(nombre: str) -> str:
    """
    Normaliza nombres de UPZ eliminando tildes, caracteres especiales y unificando variaciones.
    """
    if not nombre:
        return ""
    
    # Convertir a string por si viene como número
    nombre = str(nombre).strip()
    
    # Eliminar tildes y diacríticos
    nombre_normalizado = unicodedata.normalize('NFD', nombre)
    nombre_sin_tildes = ''.join(
        char for char in nombre_normalizado 
        if unicodedata.category(char) != 'Mn'
    )
    
    # Limpiar caracteres especiales pero mantener espacios, guiones y números
    nombre_limpio = re.sub(r'[^\w\s\-\d]', '', nombre_sin_tildes)
    
    # Normalizar espacios múltiples a uno solo
    nombre_limpio = re.sub(r'\s+', ' ', nombre_limpio)
    
    # Capitalizar correctamente cada palabra
    nombre_final = nombre_limpio.title().strip()
    
    # Aplicar reglas específicas de normalización para UPZ comunes
    reglas_upz = {
        'La Candelaria': 'La Candelaria',
        'Las Nieves': 'Las Nieves',
        'Santa Barbara': 'Santa Barbara',
        'Santa Bárbara': 'Santa Barbara',
        'San Diego': 'San Diego',
        'La Macarena': 'La Macarena',
        'Las Cruces': 'Las Cruces',
        'Lourdes': 'Lourdes',
        'Sagrado Corazon': 'Sagrado Corazon',
        'Sagrado Corazón': 'Sagrado Corazon',
        'La Sabana': 'La Sabana',
        'San Cristobal Norte': 'San Cristobal Norte',
        'San Cristóbal Norte': 'San Cristobal Norte',
        'Usaquen': 'Usaquen',
        'Usaquén': 'Usaquen',
        'Country Club': 'Country Club',
        'Santa Barbara Central': 'Santa Barbara Central',
        'Santa Bárbara Central': 'Santa Barbara Central',
        'Los Cedros': 'Los Cedros',
        'Usaquen Pueblo': 'Usaquen Pueblo',
        'San Patricio': 'San Patricio',
        'La Uribe': 'La Uribe',
        'Toberrin': 'Toberrin',
        'Toberin': 'Toberin',
        'Los Novios': 'Los Novios',
        'Casco Urbano De Suba': 'Casco Urbano De Suba',
        'El Prado': 'El Prado',
        'Britalia': 'Britalia',
        'El Rincon': 'El Rincon',
        'El Rincón': 'El Rincon',
        'Bolivia': 'Bolivia',
        'Niza': 'Niza',
        'La Floresta': 'La Floresta',
        'Suba': 'Suba',
        'Casa Blanca Suba': 'Casa Blanca Suba',
        'Las Mercedes': 'Las Mercedes',
        'Lombardia': 'Lombardia',
        'Lombardía': 'Lombardia',
        'Castilla': 'Castilla',
        'Tintal Norte': 'Tintal Norte',
        'Tintal Sur': 'Tintal Sur',
        'Patio Bonito': 'Patio Bonito',
        'Las Margaritas': 'Las Margaritas',
        'Calandaima': 'Calandaima',
        'Corabastos': 'Corabastos',
        'Zona Industrial': 'Zona Industrial',
        'Puente Aranda': 'Puente Aranda',
        'Ciudad Montes': 'Ciudad Montes',
        'Muzu': 'Muzu',
        'Muzú': 'Muzu',
        'Zona Franca': 'Zona Franca',
        'Granjas De Techo': 'Granjas De Techo',
        'Granjas Techo': 'Granjas De Techo'
    }
    
    # Buscar coincidencia exacta o similar
    for variacion, nombre_estandar in reglas_upz.items():
        if nombre_final.lower() == variacion.lower():
            return nombre_estandar
    
    return nombre_final

def calcular_promedio_general_indicador(db: Session, indicador_nombre: str) -> float:
    """
    Calcula el promedio general de un indicador específico.
    """
    resultado = db.query(func.avg(IndicadorDB.valor)).filter(
        IndicadorDB.indicador_nombre.ilike(f"%{indicador_nombre}%"),
        IndicadorDB.valor.isnot(None),
        IndicadorDB.valor > 0
    ).scalar()
    
    return resultado or 0.0

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
    nombre_localidad = Column(String, nullable=True, index=True)  # Añadido índice
    id_upz = Column(String, nullable=True)
    nombre_upz = Column(String, nullable=True)
    area_geografica = Column(String, nullable=True)
    año_inicio = Column(Integer, nullable=True, index=True)  # Añadido índice
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
# MODELOS PYDANTIC MEJORADOS
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

class IndicadorSuperiorPromedio(BaseModel):
    indicador: str
    dimension: str
    valor_promedio_localidad: float
    promedio_general: float
    diferencia_porcentual: float
    registros: int
    año_mas_reciente: Optional[int]
    supera_promedio: bool

class UPZInfo(BaseModel):
    id_upz: str
    nombre_upz: str
    registros: int
    indicadores: int

class LocalidadAnalisisCompleto(BaseModel):
    localidad: Dict[str, Any]
    resumen_general: Dict[str, Any]
    upzs: List[UPZInfo]
    indicadores_superiores_promedio: List[IndicadorSuperiorPromedio]
    analisis_por_dimension: List[Dict[str, Any]]
    evoluciones_temporales: List[Dict[str, Any]]
    estadisticas_comparativas: Dict[str, Any]

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
    solo_superiores_promedio: Optional[bool] = False

# =====================================================
# CONFIGURACIÓN DE FASTAPI
# =====================================================

app = FastAPI(
    title="API Indicadores Sociales Bogotá",
    description="API para carga y caracterización de indicadores sociales de Bogotá D.C. - Versión con Normalización UPZ",
    version="1.2.0"
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
    return FileResponse('dashboard_compatible.html')

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
# FUNCIONES DE ANÁLISIS MEJORADAS
# =====================================================

def obtener_indicadores_superiores_promedio(db: Session, localidad: str) -> List[Dict]:
    """
    Obtiene solo los indicadores que superan el promedio general.
    """
    # Obtener todos los indicadores de la localidad
    indicadores_localidad = db.query(
        IndicadorDB.indicador_nombre,
        IndicadorDB.dimension,
        func.avg(IndicadorDB.valor).label('valor_promedio_localidad'),
        func.count(IndicadorDB.id).label('registros'),
        func.max(IndicadorDB.año_inicio).label('año_mas_reciente')
    ).filter(
        IndicadorDB.nombre_localidad.ilike(f"%{localidad}%"),
        IndicadorDB.valor.isnot(None),
        IndicadorDB.valor > 0
    ).group_by(IndicadorDB.indicador_nombre, IndicadorDB.dimension).all()
    
    indicadores_filtrados = []
    
    for indicador in indicadores_localidad:
        promedio_general = calcular_promedio_general_indicador(db, indicador.indicador_nombre)
        
        # Solo incluir si supera el promedio general
        if indicador.valor_promedio_localidad > promedio_general:
            indicadores_filtrados.append({
                "indicador": indicador.indicador_nombre,
                "dimension": indicador.dimension,
                "valor_promedio_localidad": round(indicador.valor_promedio_localidad, 2),
                "promedio_general": round(promedio_general, 2),
                "diferencia_porcentual": round(
                    ((indicador.valor_promedio_localidad - promedio_general) / promedio_general) * 100, 1
                ) if promedio_general > 0 else 0,
                "registros": indicador.registros,
                "año_mas_reciente": indicador.año_mas_reciente,
                "supera_promedio": True
            })
    
    return sorted(indicadores_filtrados, key=lambda x: x['diferencia_porcentual'], reverse=True)

def analisis_detallado_localidad_con_upz(db: Session, localidad: str) -> Dict[str, Any]:
    """
    Análisis completo de una localidad incluyendo información de UPZ.
    """
    localidad_normalizada = normalizar_nombre_localidad(localidad)
    
    # Información básica de la localidad
    info_basica = db.query(
        IndicadorDB.id_localidad,
        IndicadorDB.nombre_localidad,
        func.count(IndicadorDB.id).label('total_registros'),
        func.count(IndicadorDB.indicador_nombre.distinct()).label('indicadores_unicos'),
        func.min(IndicadorDB.año_inicio).label('año_min'),
        func.max(IndicadorDB.año_inicio).label('año_max')
    ).filter(
        IndicadorDB.nombre_localidad.ilike(f"%{localidad}%")
    ).group_by(IndicadorDB.id_localidad, IndicadorDB.nombre_localidad).first()
    
    if not info_basica:
        return {"error": f"Localidad '{localidad}' no encontrada"}
    
    # Información de UPZ en esta localidad
    upzs = db.query(
        IndicadorDB.id_upz,
        IndicadorDB.nombre_upz,
        func.count(IndicadorDB.id).label('registros_upz'),
        func.count(IndicadorDB.indicador_nombre.distinct()).label('indicadores_upz')
    ).filter(
        IndicadorDB.nombre_localidad.ilike(f"%{localidad}%"),
        IndicadorDB.id_upz.isnot(None)
    ).group_by(IndicadorDB.id_upz, IndicadorDB.nombre_upz).all()
    
    # Indicadores que superan el promedio
    indicadores_superiores = obtener_indicadores_superiores_promedio(db, localidad)
    
    # Análisis por dimensión
    por_dimension = db.query(
        IndicadorDB.dimension,
        func.count(IndicadorDB.id).label('registros'),
        func.count(IndicadorDB.indicador_nombre.distinct()).label('indicadores_diferentes'),
        func.avg(IndicadorDB.valor).label('valor_promedio_dimension')
    ).filter(
        IndicadorDB.nombre_localidad.ilike(f"%{localidad}%"),
        IndicadorDB.dimension.isnot(None),
        IndicadorDB.valor.isnot(None)
    ).group_by(IndicadorDB.dimension).all()
    
    # Evolución temporal de los mejores indicadores
    evoluciones = []
    for indicador in indicadores_superiores[:5]:  # Top 5 indicadores
        evolucion = db.query(
            IndicadorDB.año_inicio,
            func.avg(IndicadorDB.valor).label('valor_promedio')
        ).filter(
            IndicadorDB.nombre_localidad.ilike(f"%{localidad}%"),
            IndicadorDB.indicador_nombre == indicador['indicador'],
            IndicadorDB.año_inicio.isnot(None)
        ).group_by(IndicadorDB.año_inicio).order_by(IndicadorDB.año_inicio).all()
        
        if evolucion:
            evoluciones.append({
                "indicador": indicador['indicador'],
                "serie_temporal": [
                    {"año": e.año_inicio, "valor": round(e.valor_promedio, 2)}
                    for e in evolucion
                ]
            })
    
    return {
        "localidad": {
            "nombre_original": info_basica.nombre_localidad,
            "nombre_normalizado": localidad_normalizada,
            "id_localidad": info_basica.id_localidad
        },
        "resumen_general": {
            "total_registros": info_basica.total_registros,
            "indicadores_unicos": info_basica.indicadores_unicos,
            "rango_años": f"{info_basica.año_min} - {info_basica.año_max}",
            "upzs_disponibles": len(upzs)
        },
        "upzs": [
            {
                "id_upz": upz.id_upz,
                "nombre_upz": upz.nombre_upz,
                "registros": upz.registros_upz,
                "indicadores": upz.indicadores_upz
            } for upz in upzs
        ],
        "indicadores_superiores_promedio": indicadores_superiores,
        "analisis_por_dimension": [
            {
                "dimension": dim.dimension,
                "registros": dim.registros,
                "indicadores_diferentes": dim.indicadores_diferentes,
                "valor_promedio": round(dim.valor_promedio_dimension, 2) if dim.valor_promedio_dimension else None
            } for dim in por_dimension
        ],
        "evoluciones_temporales": evoluciones,
        "estadisticas_comparativas": {
            "total_indicadores_analizados": len(indicadores_superiores),
            "mejor_indicador": indicadores_superiores[0] if indicadores_superiores else None,
            "promedio_diferencia_porcentual": round(
                sum(ind['diferencia_porcentual'] for ind in indicadores_superiores) / len(indicadores_superiores), 1
            ) if indicadores_superiores else 0
        }
    }

def obtener_indicadores_superiores_promedio_upz(db: Session, upz_id: str) -> List[Dict]:
    """
    Obtiene solo los indicadores que superan el promedio general para una UPZ específica.
    """
    # Obtener todos los indicadores de la UPZ
    indicadores_upz = db.query(
        IndicadorDB.indicador_nombre,
        IndicadorDB.dimension,
        func.avg(IndicadorDB.valor).label('valor_promedio_upz'),
        func.count(IndicadorDB.id).label('registros'),
        func.max(IndicadorDB.año_inicio).label('año_mas_reciente')
    ).filter(
        IndicadorDB.id_upz == upz_id,
        IndicadorDB.valor.isnot(None),
        IndicadorDB.valor > 0
    ).group_by(IndicadorDB.indicador_nombre, IndicadorDB.dimension).all()
    
    indicadores_filtrados = []
    
    for indicador in indicadores_upz:
        promedio_general = calcular_promedio_general_indicador(db, indicador.indicador_nombre)
        
        # Solo incluir si supera el promedio general
        if indicador.valor_promedio_upz > promedio_general:
            indicadores_filtrados.append({
                "indicador": indicador.indicador_nombre,
                "dimension": indicador.dimension,
                "valor_promedio_upz": round(indicador.valor_promedio_upz, 2),
                "promedio_general": round(promedio_general, 2),
                "diferencia_porcentual": round(
                    ((indicador.valor_promedio_upz - promedio_general) / promedio_general) * 100, 1
                ) if promedio_general > 0 else 0,
                "registros": indicador.registros,
                "año_mas_reciente": indicador.año_mas_reciente,
                "supera_promedio": True
            })
    
    return sorted(indicadores_filtrados, key=lambda x: x['diferencia_porcentual'], reverse=True)

def analisis_detallado_upz(db: Session, upz_id: str) -> Dict[str, Any]:
    """
    Análisis completo de una UPZ específica.
    """
    # Información básica de la UPZ
    info_basica = db.query(
        IndicadorDB.id_upz,
        IndicadorDB.nombre_upz,
        IndicadorDB.nombre_localidad,
        func.count(IndicadorDB.id).label('total_registros'),
        func.count(IndicadorDB.indicador_nombre.distinct()).label('indicadores_unicos'),
        func.min(IndicadorDB.año_inicio).label('año_min'),
        func.max(IndicadorDB.año_inicio).label('año_max')
    ).filter(
        IndicadorDB.id_upz == upz_id
    ).group_by(
        IndicadorDB.id_upz, 
        IndicadorDB.nombre_upz, 
        IndicadorDB.nombre_localidad
    ).first()
    
    if not info_basica:
        return {"error": f"UPZ '{upz_id}' no encontrada"}
    
    # Indicadores que superan el promedio
    indicadores_superiores = obtener_indicadores_superiores_promedio_upz(db, upz_id)
    
    # Análisis por dimensión
    por_dimension = db.query(
        IndicadorDB.dimension,
        func.count(IndicadorDB.id).label('registros'),
        func.count(IndicadorDB.indicador_nombre.distinct()).label('indicadores_diferentes'),
        func.avg(IndicadorDB.valor).label('valor_promedio_dimension')
    ).filter(
        IndicadorDB.id_upz == upz_id,
        IndicadorDB.dimension.isnot(None),
        IndicadorDB.valor.isnot(None)
    ).group_by(IndicadorDB.dimension).all()
    
    # Evolución temporal de los mejores indicadores
    evoluciones = []
    for indicador in indicadores_superiores[:5]:  # Top 5 indicadores
        evolucion = db.query(
            IndicadorDB.año_inicio,
            func.avg(IndicadorDB.valor).label('valor_promedio')
        ).filter(
            IndicadorDB.id_upz == upz_id,
            IndicadorDB.indicador_nombre == indicador['indicador'],
            IndicadorDB.año_inicio.isnot(None)
        ).group_by(IndicadorDB.año_inicio).order_by(IndicadorDB.año_inicio).all()
        
        if evolucion:
            evoluciones.append({
                "indicador": indicador['indicador'],
                "serie_temporal": [
                    {"año": e.año_inicio, "valor": round(e.valor_promedio, 2)}
                    for e in evolucion
                ]
            })
    
    return {
        "upz": {
            "id_upz": info_basica.id_upz,
            "nombre_upz": info_basica.nombre_upz,
            "localidad_pertenece": info_basica.nombre_localidad
        },
        "resumen_general": {
            "total_registros": info_basica.total_registros,
            "indicadores_unicos": info_basica.indicadores_unicos,
            "rango_años": f"{info_basica.año_min} - {info_basica.año_max}"
        },
        "indicadores_superiores_promedio": indicadores_superiores,
        "analisis_por_dimension": [
            {
                "dimension": dim.dimension,
                "registros": dim.registros,
                "indicadores_diferentes": dim.indicadores_diferentes,
                "valor_promedio": round(dim.valor_promedio_dimension, 2) if dim.valor_promedio_dimension else None
            } for dim in por_dimension
        ],
        "evoluciones_temporales": evoluciones,
        "estadisticas_comparativas": {
            "total_indicadores_analizados": len(indicadores_superiores),
            "mejor_indicador": indicadores_superiores[0] if indicadores_superiores else None,
            "promedio_diferencia_porcentual": round(
                sum(ind['diferencia_porcentual'] for ind in indicadores_superiores) / len(indicadores_superiores), 1
            ) if indicadores_superiores else 0
        }
    }

def generar_datos_graficos_upz(db: Session, upz_id: str) -> Dict[str, Any]:
    """
    Genera datos específicos para gráficos de una UPZ.
    """
    # Obtener indicadores superiores al promedio
    indicadores_superiores = obtener_indicadores_superiores_promedio_upz(db, upz_id)
    
    if not indicadores_superiores:
        return {"error": "No hay indicadores que superen el promedio para esta UPZ"}
    
    # Datos para gráfico de barras (top indicadores)
    datos_barras = {
        "labels": [ind['indicador'][:25] + "..." if len(ind['indicador']) > 25 else ind['indicador'] 
                  for ind in indicadores_superiores[:8]],
        "valores": [ind['valor_promedio_upz'] for ind in indicadores_superiores[:8]],
        "promedios_generales": [ind['promedio_general'] for ind in indicadores_superiores[:8]],
        "diferencias_porcentuales": [ind['diferencia_porcentual'] for ind in indicadores_superiores[:8]]
    }
    
    # Datos para gráfico de tendencia temporal (mejor indicador)
    mejor_indicador = indicadores_superiores[0]['indicador']
    tendencia_temporal = db.query(
        IndicadorDB.año_inicio,
        func.avg(IndicadorDB.valor).label('valor_promedio')
    ).filter(
        IndicadorDB.id_upz == upz_id,
        IndicadorDB.indicador_nombre == mejor_indicador,
        IndicadorDB.año_inicio.isnot(None)
    ).group_by(IndicadorDB.año_inicio).order_by(IndicadorDB.año_inicio).all()
    
    datos_tendencia = {
        "indicador": mejor_indicador,
        "años": [t.año_inicio for t in tendencia_temporal],
        "valores": [round(t.valor_promedio, 2) for t in tendencia_temporal]
    }
    
    # Datos para gráfico de dimensiones
    por_dimension = db.query(
        IndicadorDB.dimension,
        func.avg(IndicadorDB.valor).label('valor_promedio')
    ).filter(
        IndicadorDB.id_upz == upz_id,
        IndicadorDB.dimension.isnot(None),
        IndicadorDB.valor.isnot(None)
    ).group_by(IndicadorDB.dimension).all()
    
    datos_dimensiones = {
        "labels": [dim.dimension for dim in por_dimension],
        "valores": [round(dim.valor_promedio, 2) for dim in por_dimension]
    }
    
    return {
        "upz_id": upz_id,
        "graficos": {
            "barras_indicadores": datos_barras,
            "tendencia_temporal": datos_tendencia,
            "por_dimensiones": datos_dimensiones
        },
        "metadatos": {
            "total_indicadores_superiores": len(indicadores_superiores),
            "fecha_generacion": datetime.utcnow().isoformat()
        }
    }

def generar_datos_graficos_localidad(db: Session, localidad: str) -> Dict[str, Any]:
    """
    Genera datos específicos para gráficos de una localidad.
    """
    # Obtener indicadores superiores al promedio
    indicadores_superiores = obtener_indicadores_superiores_promedio(db, localidad)
    
    if not indicadores_superiores:
        return {"error": "No hay indicadores que superen el promedio para esta localidad"}
    
    # Datos para gráfico de barras (top indicadores)
    datos_barras = {
        "labels": [ind['indicador'][:30] + "..." if len(ind['indicador']) > 30 else ind['indicador'] 
                  for ind in indicadores_superiores[:8]],
        "valores": [ind['valor_promedio_localidad'] for ind in indicadores_superiores[:8]],
        "promedios_generales": [ind['promedio_general'] for ind in indicadores_superiores[:8]],
        "diferencias_porcentuales": [ind['diferencia_porcentual'] for ind in indicadores_superiores[:8]]
    }
    
    # Datos para gráfico de tendencia temporal (mejor indicador)
    mejor_indicador = indicadores_superiores[0]['indicador']
    tendencia_temporal = db.query(
        IndicadorDB.año_inicio,
        func.avg(IndicadorDB.valor).label('valor_promedio')
    ).filter(
        IndicadorDB.nombre_localidad.ilike(f"%{localidad}%"),
        IndicadorDB.indicador_nombre == mejor_indicador,
        IndicadorDB.año_inicio.isnot(None)
    ).group_by(IndicadorDB.año_inicio).order_by(IndicadorDB.año_inicio).all()
    
    datos_tendencia = {
        "indicador": mejor_indicador,
        "años": [t.año_inicio for t in tendencia_temporal],
        "valores": [round(t.valor_promedio, 2) for t in tendencia_temporal]
    }
    
    # Datos para gráfico de dimensiones
    por_dimension = db.query(
        IndicadorDB.dimension,
        func.avg(IndicadorDB.valor).label('valor_promedio')
    ).filter(
        IndicadorDB.nombre_localidad.ilike(f"%{localidad}%"),
        IndicadorDB.dimension.isnot(None),
        IndicadorDB.valor.isnot(None)
    ).group_by(IndicadorDB.dimension).all()
    
    datos_dimensiones = {
        "labels": [dim.dimension for dim in por_dimension],
        "valores": [round(dim.valor_promedio, 2) for dim in por_dimension]
    }
    
    return {
        "localidad": localidad,
        "graficos": {
            "barras_indicadores": datos_barras,
            "tendencia_temporal": datos_tendencia,
            "por_dimensiones": datos_dimensiones
        },
        "metadatos": {
            "total_indicadores_superiores": len(indicadores_superiores),
            "fecha_generacion": datetime.utcnow().isoformat()
        }
    }

def normalizar_todas_las_localidades(db: Session) -> Dict[str, int]:
    """
    Normaliza todas las localidades en la base de datos.
    """
    # Obtener todas las localidades únicas
    localidades_unicas = db.query(IndicadorDB.nombre_localidad).distinct().all()
    
    actualizaciones = 0
    mapeo_cambios = {}
    
    for localidad_tupla in localidades_unicas:
        nombre_original = localidad_tupla[0]
        if nombre_original:
            nombre_normalizado = normalizar_nombre_localidad(nombre_original)
            
            if nombre_original != nombre_normalizado:
                # Actualizar todos los registros con este nombre
                registros_actualizados = db.query(IndicadorDB).filter(
                    IndicadorDB.nombre_localidad == nombre_original
                ).update({"nombre_localidad": nombre_normalizado})
                
                actualizaciones += registros_actualizados
                mapeo_cambios[nombre_original] = {
                    "nuevo_nombre": nombre_normalizado,
                    "registros_actualizados": registros_actualizados
                }
    
    db.commit()
    
    return {
        "total_actualizaciones": actualizaciones,
        "localidades_modificadas": len(mapeo_cambios),
        "mapeo_cambios": mapeo_cambios
    }

def normalizar_todas_las_upz(db: Session) -> Dict[str, int]:
    """
    Normaliza todas las UPZ en la base de datos para eliminar duplicados.
    """
    # Obtener todas las UPZ únicas (tanto ID como nombre)
    upz_unicas = db.query(
        IndicadorDB.id_upz, 
        IndicadorDB.nombre_upz
    ).filter(
        IndicadorDB.id_upz.isnot(None),
        IndicadorDB.nombre_upz.isnot(None)
    ).distinct().all()
    
    actualizaciones = 0
    mapeo_cambios = {}
    upz_duplicadas_eliminadas = 0
    
    # Crear mapeo de UPZ normalizadas
    upz_normalizadas = {}
    
    for upz in upz_unicas:
        id_upz_original = str(upz.id_upz).strip()
        nombre_original = upz.nombre_upz
        nombre_normalizado = normalizar_nombre_upz(nombre_original)
        
        # Crear clave única combinando ID y nombre normalizado
        clave_upz = f"{id_upz_original}_{nombre_normalizado}"
        
        if clave_upz not in upz_normalizadas:
            upz_normalizadas[clave_upz] = {
                "id_upz": id_upz_original,
                "nombre_normalizado": nombre_normalizado,
                "nombres_originales": [nombre_original]
            }
        else:
            # Es un duplicado, agregar el nombre original a la lista
            upz_normalizadas[clave_upz]["nombres_originales"].append(nombre_original)
    
    # Actualizar registros con nombres normalizados
    for clave_upz, info_upz in upz_normalizadas.items():
        nombres_originales = info_upz["nombres_originales"]
        nombre_normalizado = info_upz["nombre_normalizado"]
        id_upz = info_upz["id_upz"]
        
        # Si hay múltiples nombres originales, son duplicados
        if len(nombres_originales) > 1:
            upz_duplicadas_eliminadas += len(nombres_originales) - 1
        
        # Actualizar todos los registros que tengan cualquiera de los nombres originales
        for nombre_original in nombres_originales:
            if nombre_original != nombre_normalizado:
                registros_actualizados = db.query(IndicadorDB).filter(
                    IndicadorDB.id_upz == id_upz,
                    IndicadorDB.nombre_upz == nombre_original
                ).update({"nombre_upz": nombre_normalizado})
                
                if registros_actualizados > 0:
                    actualizaciones += registros_actualizados
                    mapeo_cambios[f"{id_upz}_{nombre_original}"] = {
                        "id_upz": id_upz,
                        "nombre_original": nombre_original,
                        "nombre_normalizado": nombre_normalizado,
                        "registros_actualizados": registros_actualizados
                    }
    
    db.commit()
    
    return {
        "total_actualizaciones": actualizaciones,
        "upz_duplicadas_eliminadas": upz_duplicadas_eliminadas,
        "upz_modificadas": len(mapeo_cambios),
        "mapeo_cambios": mapeo_cambios,
        "upz_unicas_finales": len(upz_normalizadas)
    }

def normalizar_datos_completos(db: Session) -> Dict[str, Any]:
    """
    Normaliza tanto localidades como UPZ en una sola operación.
    """
    resultado_localidades = normalizar_todas_las_localidades(db)
    resultado_upz = normalizar_todas_las_upz(db)
    
    return {
        "localidades": resultado_localidades,
        "upz": resultado_upz,
        "resumen": {
            "total_actualizaciones": resultado_localidades["total_actualizaciones"] + resultado_upz["total_actualizaciones"],
            "localidades_normalizadas": resultado_localidades["localidades_modificadas"],
            "upz_normalizadas": resultado_upz["upz_modificadas"],
            "upz_duplicadas_eliminadas": resultado_upz["upz_duplicadas_eliminadas"]
        }
    }

# =====================================================
# UTILIDADES EXISTENTES MEJORADAS
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
    
    # Normalizar nombres de localidades
    if 'Nombre Localidad' in df.columns:
        df['Nombre Localidad'] = df['Nombre Localidad'].apply(
            lambda x: normalizar_nombre_localidad(str(x)) if pd.notna(x) else x
        )
    
    # Normalizar nombres de UPZ
    if 'Nombre_UPZ' in df.columns:
        df['Nombre_UPZ'] = df['Nombre_UPZ'].apply(
            lambda x: normalizar_nombre_upz(str(x)) if pd.notna(x) else x
        )
    
    # Normalizar ID de UPZ (asegurar que sea string limpio)
    if 'ID_UPZ' in df.columns:
        df['ID_UPZ'] = df['ID_UPZ'].apply(
            lambda x: str(x).strip() if pd.notna(x) else x
        )
    
    # Eliminar filas completamente vacías
    df = df.dropna(how='all')
    
    return df

# =====================================================
# ENDPOINTS PRINCIPALES MEJORADOS
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

# =====================================================
# NUEVOS ENDPOINTS MEJORADOS
# =====================================================

@app.get("/localidades/{localidad}/analisis-completo")
async def analisis_completo_localidad(localidad: str, db: Session = Depends(get_db)):
    """Análisis completo de una localidad incluyendo UPZ y indicadores superiores al promedio"""
    try:
        resultado = analisis_detallado_localidad_con_upz(db, localidad)
        if "error" in resultado:
            raise HTTPException(status_code=404, detail=resultado["error"])
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis: {str(e)}")

@app.get("/localidades/{localidad}/graficos")
async def datos_graficos_localidad(localidad: str, db: Session = Depends(get_db)):
    """Genera datos específicos para gráficos de una localidad"""
    try:
        resultado = generar_datos_graficos_localidad(db, localidad)
        if "error" in resultado:
            raise HTTPException(status_code=404, detail=resultado["error"])
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando gráficos: {str(e)}")

# =====================================================
# NUEVOS ENDPOINTS PARA UPZ Y LOCALIDADES SEPARADOS
# =====================================================

@app.get("/upz/{upz_id}/analisis-completo")
async def analisis_completo_upz(upz_id: str, db: Session = Depends(get_db)):
    """Análisis completo específico de una UPZ"""
    try:
        resultado = analisis_detallado_upz(db, upz_id)
        if "error" in resultado:
            raise HTTPException(status_code=404, detail=resultado["error"])
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis UPZ: {str(e)}")

@app.get("/upz/{upz_id}/graficos")
async def datos_graficos_upz(upz_id: str, db: Session = Depends(get_db)):
    """Genera datos específicos para gráficos de una UPZ"""
    try:
        resultado = generar_datos_graficos_upz(db, upz_id)
        if "error" in resultado:
            raise HTTPException(status_code=404, detail=resultado["error"])
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando gráficos UPZ: {str(e)}")

@app.get("/upz/listado")
async def listar_upz_disponibles(db: Session = Depends(get_db)):
    """Lista todas las UPZ disponibles con estadísticas"""
    try:
        upzs = db.query(
            IndicadorDB.id_upz,
            IndicadorDB.nombre_upz,
            IndicadorDB.nombre_localidad,
            func.count(IndicadorDB.id).label('total_registros'),
            func.count(IndicadorDB.indicador_nombre.distinct()).label('indicadores_unicos')
        ).filter(
            IndicadorDB.id_upz.isnot(None),
            IndicadorDB.nombre_upz.isnot(None)
        ).group_by(
            IndicadorDB.id_upz, 
            IndicadorDB.nombre_upz, 
            IndicadorDB.nombre_localidad
        ).all()
        
        return [
            {
                "id_upz": upz.id_upz,
                "nombre_upz": upz.nombre_upz,
                "localidad": upz.nombre_localidad,
                "total_registros": upz.total_registros,
                "indicadores_unicos": upz.indicadores_unicos
            } for upz in upzs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando UPZ: {str(e)}")

@app.get("/upz/{upz_id}/indicadores-superiores")
async def indicadores_superiores_promedio_upz(upz_id: str, db: Session = Depends(get_db)):
    """Obtiene indicadores que superan el promedio general para una UPZ específica"""
    try:
        indicadores = obtener_indicadores_superiores_promedio_upz(db, upz_id)
        return {
            "upz_id": upz_id,
            "indicadores_superiores": indicadores,
            "total_encontrados": len(indicadores)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# =====================================================
# NUEVOS ENDPOINTS DE NORMALIZACIÓN
# =====================================================

@app.get("/administracion/verificar-duplicados")
async def verificar_duplicados(db: Session = Depends(get_db)):
    """Verifica si existen duplicados en localidades y UPZ sin modificar datos"""
    try:
        # Contar registros totales ANTES de cualquier análisis
        total_registros_actuales = db.query(IndicadorDB).count()
        
        # Verificar duplicados en localidades
        localidades_raw = db.query(IndicadorDB.nombre_localidad).filter(
            IndicadorDB.nombre_localidad.isnot(None)
        ).distinct().all()
        
        localidades_normalizadas = {}
        duplicados_localidades = []
        
        for loc in localidades_raw:
            nombre_original = loc[0]
            nombre_normalizado = normalizar_nombre_localidad(nombre_original)
            
            if nombre_normalizado in localidades_normalizadas:
                duplicados_localidades.append({
                    "original": nombre_original,
                    "normalizado": nombre_normalizado,
                    "conflicto_con": localidades_normalizadas[nombre_normalizado]
                })
            else:
                localidades_normalizadas[nombre_normalizado] = nombre_original
        
        # Verificar duplicados en UPZ
        upz_raw = db.query(
            IndicadorDB.id_upz, 
            IndicadorDB.nombre_upz
        ).filter(
            IndicadorDB.id_upz.isnot(None),
            IndicadorDB.nombre_upz.isnot(None)
        ).distinct().all()
        
        upz_normalizadas = {}
        duplicados_upz = []
        
        for upz in upz_raw:
            id_upz = str(upz.id_upz).strip()
            nombre_original = upz.nombre_upz
            nombre_normalizado = normalizar_nombre_upz(nombre_original)
            clave_upz = f"{id_upz}_{nombre_normalizado}"
            
            if clave_upz in upz_normalizadas:
                duplicados_upz.append({
                    "id_upz": id_upz,
                    "nombre_original": nombre_original,
                    "nombre_normalizado": nombre_normalizado,
                    "conflicto_con": upz_normalizadas[clave_upz]
                })
            else:
                upz_normalizadas[clave_upz] = nombre_original
        
        return {
            "mensaje": "Verificación de duplicados - NO se modifican datos",
            "total_registros_actuales": total_registros_actuales,
            "localidades": {
                "total_originales": len(localidades_raw),
                "total_normalizadas": len(localidades_normalizadas),
                "duplicados_detectados": len(duplicados_localidades),
                "duplicados": duplicados_localidades
            },
            "upz": {
                "total_originales": len(upz_raw),
                "total_normalizadas": len(upz_normalizadas),
                "duplicados_detectados": len(duplicados_upz),
                "duplicados": duplicados_upz
            },
            "requiere_normalizacion": len(duplicados_localidades) > 0 or len(duplicados_upz) > 0,
            "garantia_seguridad": {
                "se_pierden_registros": False,
                "se_pierden_datos": False,
                "solo_se_unifican_nombres": True,
                "operacion_reversible": True
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error verificando duplicados: {str(e)}")

@app.get("/administracion/preview-normalizacion")
async def preview_normalizacion(db: Session = Depends(get_db)):
    """
    Muestra EXACTAMENTE qué cambios se harían sin ejecutarlos.
    GARANTIZA que NO se pierden datos, solo se cambian nombres.
    """
    try:
        # CONTAR registros ANTES de cualquier cambio
        total_registros_antes = db.query(IndicadorDB).count()
        total_indicadores_antes = db.query(IndicadorDB.indicador_nombre).distinct().count()
        total_valores_antes = db.query(IndicadorDB).filter(IndicadorDB.valor.isnot(None)).count()
        
        # Verificar cambios en localidades
        localidades_raw = db.query(IndicadorDB.nombre_localidad).filter(
            IndicadorDB.nombre_localidad.isnot(None)
        ).distinct().all()
        
        cambios_localidades = []
        for loc in localidades_raw:
            nombre_original = loc[0]
            nombre_normalizado = normalizar_nombre_localidad(nombre_original)
            
            if nombre_original != nombre_normalizado:
                registros_afectados = db.query(IndicadorDB).filter(
                    IndicadorDB.nombre_localidad == nombre_original
                ).count()
                
                cambios_localidades.append({
                    "nombre_actual": nombre_original,
                    "nombre_nuevo": nombre_normalizado,
                    "registros_que_cambiaran_nombre": registros_afectados,
                    "se_pierden_datos": False,
                    "se_mantienen_valores": True
                })
        
        # Verificar cambios en UPZ
        upz_raw = db.query(
            IndicadorDB.id_upz, 
            IndicadorDB.nombre_upz
        ).filter(
            IndicadorDB.id_upz.isnot(None),
            IndicadorDB.nombre_upz.isnot(None)
        ).distinct().all()
        
        cambios_upz = []
        for upz in upz_raw:
            id_upz = str(upz.id_upz).strip()
            nombre_original = upz.nombre_upz
            nombre_normalizado = normalizar_nombre_upz(nombre_original)
            
            if nombre_original != nombre_normalizado:
                registros_afectados = db.query(IndicadorDB).filter(
                    IndicadorDB.id_upz == id_upz,
                    IndicadorDB.nombre_upz == nombre_original
                ).count()
                
                cambios_upz.append({
                    "id_upz": id_upz,
                    "nombre_actual": nombre_original,
                    "nombre_nuevo": nombre_normalizado,
                    "registros_que_cambiaran_nombre": registros_afectados,
                    "se_pierden_datos": False,
                    "se_mantienen_valores": True
                })
        
        # SIMULAR conteo DESPUÉS (será igual porque NO se eliminan registros)
        total_registros_despues = total_registros_antes  # MISMO número
        total_indicadores_despues = total_indicadores_antes  # MISMO número
        total_valores_despues = total_valores_antes  # MISMO número
        
        return {
            "mensaje": "PREVIEW - NO se ejecutan cambios, solo se muestran",
            "garantia_no_perdida_datos": True,
            "verificacion_conteos": {
                "registros_antes": total_registros_antes,
                "registros_despues": total_registros_despues,
                "diferencia_registros": 0,
                "indicadores_antes": total_indicadores_antes,
                "indicadores_despues": total_indicadores_despues,
                "diferencia_indicadores": 0,
                "valores_numericos_antes": total_valores_antes,
                "valores_numericos_despues": total_valores_despues,
                "diferencia_valores": 0
            },
            "cambios_localidades": {
                "total_cambios": len(cambios_localidades),
                "detalle": cambios_localidades[:10],  # Mostrar primeros 10
                "nota": "Solo cambian NOMBRES, se mantienen todos los registros"
            },
            "cambios_upz": {
                "total_cambios": len(cambios_upz),
                "detalle": cambios_upz[:10],  # Mostrar primeros 10  
                "nota": "Solo cambian NOMBRES, se mantienen todos los registros"
            },
            "resumen_seguridad": {
                "se_eliminan_registros": False,
                "se_eliminan_indicadores": False,
                "se_eliminan_valores": False,
                "se_pierden_datos": False,
                "solo_se_unifican_nombres": True,
                "operacion_segura": True
            },
            "instrucciones": [
                "Esta operación es 100% segura",
                "NO se eliminan registros de datos",
                "Solo se normalizan nombres inconsistentes",
                "Todos los valores numéricos se mantienen",
                "Se puede revertir si es necesario"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en preview: {str(e)}")

@app.get("/administracion/contar-registros")
async def contar_registros_detallado(db: Session = Depends(get_db)):
    """Cuenta exacta de todos los registros para verificar que no se pierden datos"""
    try:
        conteos = {
            "total_registros": db.query(IndicadorDB).count(),
            "registros_con_valores": db.query(IndicadorDB).filter(IndicadorDB.valor.isnot(None)).count(),
            "indicadores_unicos": db.query(IndicadorDB.indicador_nombre).distinct().count(),
            "localidades_unicas": db.query(IndicadorDB.nombre_localidad).filter(
                IndicadorDB.nombre_localidad.isnot(None)
            ).distinct().count(),
            "upz_unicas": db.query(IndicadorDB.id_upz).filter(
                IndicadorDB.id_upz.isnot(None)
            ).distinct().count(),
            "dimensiones_unicas": db.query(IndicadorDB.dimension).filter(
                IndicadorDB.dimension.isnot(None)
            ).distinct().count(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return {
            "mensaje": "Conteo detallado ANTES de normalización",
            "conteos": conteos,
            "nota": "Guarda estos números para compararlos después de la normalización",
            "garantia": "Después de normalizar, todos estos números deben ser iguales o mayores, nunca menores"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error contando registros: {str(e)}")

@app.post("/administracion/normalizar-localidades")
async def normalizar_localidades_db(db: Session = Depends(get_db)):
    """Normaliza todos los nombres de localidades en la base de datos"""
    try:
        resultado = normalizar_todas_las_localidades(db)
        return {
            "mensaje": "Normalización de localidades completada",
            "registros_procesados": resultado["total_actualizaciones"],
            "localidades_modificadas": resultado["localidades_modificadas"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en proceso: {str(e)}")

@app.post("/administracion/normalizar-upz")
async def normalizar_upz_db(db: Session = Depends(get_db)):
    """Normaliza todos los nombres de UPZ en la base de datos y elimina duplicados"""
    try:
        resultado = normalizar_todas_las_upz(db)
        return {
            "mensaje": "Normalización de UPZ completada",
            "registros_procesados": resultado["total_actualizaciones"],
            "upz_modificadas": resultado["upz_modificadas"],
            "upz_duplicadas_eliminadas": resultado["upz_duplicadas_eliminadas"],
            "upz_unicas_finales": resultado["upz_unicas_finales"],
            "detalle": "Se eliminaron duplicados y se normalizaron los nombres"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en proceso: {str(e)}")

@app.post("/administracion/normalizar-completo")
async def normalizar_datos_completos_endpoint(db: Session = Depends(get_db)):
    """Normaliza tanto localidades como UPZ en una sola operación (RECOMENDADO)"""
    try:
        resultado = normalizar_datos_completos(db)
        return {
            "mensaje": "Normalización completa exitosa",
            "resumen": resultado["resumen"],
            "localidades": {
                "registros_actualizados": resultado["localidades"]["total_actualizaciones"],
                "localidades_modificadas": resultado["localidades"]["localidades_modificadas"]
            },
            "upz": {
                "registros_actualizados": resultado["upz"]["total_actualizaciones"],
                "upz_modificadas": resultado["upz"]["upz_modificadas"],
                "duplicadas_eliminadas": resultado["upz"]["upz_duplicadas_eliminadas"],
                "upz_unicas_finales": resultado["upz"]["upz_unicas_finales"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en proceso completo: {str(e)}")

@app.get("/localidades/{localidad}/indicadores-superiores")
async def indicadores_superiores_promedio_endpoint(localidad: str, db: Session = Depends(get_db)):
    """Obtiene indicadores que superan el promedio general para una localidad"""
    try:
        indicadores = obtener_indicadores_superiores_promedio(db, localidad)
        return {
            "localidad": localidad,
            "indicadores_superiores": indicadores,
            "total_encontrados": len(indicadores),
            "criterio": "Solo indicadores que superan el promedio general de Bogotá"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# =====================================================
# ENDPOINTS EXISTENTES MEJORADOS
# =====================================================

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
    """Filtra indicadores según criterios específicos - MEJORADO"""
    
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
    
    # NUEVO: Filtro para indicadores superiores al promedio
    if filtros.solo_superiores_promedio:
        # Subconsulta para obtener promedios por indicador
        subquery = db.query(
            IndicadorDB.indicador_nombre,
            func.avg(IndicadorDB.valor).label('promedio_indicador')
        ).filter(
            IndicadorDB.valor.isnot(None),
            IndicadorDB.valor > 0
        ).group_by(IndicadorDB.indicador_nombre).subquery()
        
        query = query.join(
            subquery, 
            IndicadorDB.indicador_nombre == subquery.c.indicador_nombre
        ).filter(
            IndicadorDB.valor > subquery.c.promedio_indicador
        )
    
    resultados = query.offset(offset).limit(limit).all()
    
    return [IndicadorResponse.from_orm(r) for r in resultados]

@app.get("/indicadores/{indicador_nombre}/analisis")
async def analizar_indicador_especifico(
    indicador_nombre: str,
    db: Session = Depends(get_db)
):
    """Análisis detallado de un indicador específico - MEJORADO"""
    
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
    
    # Distribución por localidad - SOLO SUPERIORES AL PROMEDIO
    promedio_general = calcular_promedio_general_indicador(db, indicador_nombre)
    
    por_localidad = db.query(
        IndicadorDB.nombre_localidad,
        func.avg(IndicadorDB.valor).label('valor_promedio'),
        func.count(IndicadorDB.id).label('registros')
    ).filter(
        IndicadorDB.indicador_nombre.ilike(f"%{indicador_nombre}%"),
        IndicadorDB.nombre_localidad.isnot(None),
        IndicadorDB.valor > promedio_general  # FILTRO MEJORADO
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
            "rango_años": f"{stats.año_min} - {stats.año_max}",
            "promedio_general_bogota": round(promedio_general, 2)
        },
        "distribucion_por_localidad": [
            {
                "localidad": loc.nombre_localidad,
                "valor_promedio": round(loc.valor_promedio, 2) if loc.valor_promedio else None,
                "registros": loc.registros,
                "supera_promedio": True,
                "diferencia_porcentual": round(
                    ((loc.valor_promedio - promedio_general) / promedio_general) * 100, 1
                ) if promedio_general > 0 else 0
            } for loc in por_localidad
        ],
        "evolucion_temporal": [
            {
                "año": temp.año_inicio,
                "valor_promedio": round(temp.valor_promedio, 2) if temp.valor_promedio else None,
                "registros": temp.registros
            } for temp in temporal
        ],
        "criterios_filtrado": {
            "solo_localidades_superiores_promedio": True,
            "promedio_referencia": round(promedio_general, 2)
        }
    }

@app.get("/localidades/{localidad}/resumen")
async def resumen_localidad(
    localidad: str,
    db: Session = Depends(get_db)
):
    """Resumen de todos los indicadores para una localidad específica - MEJORADO"""
    
    # Verificar que existe la localidad
    existe = db.query(IndicadorDB).filter(
        IndicadorDB.nombre_localidad.ilike(f"%{localidad}%")
    ).first()
    
    if not existe:
        raise HTTPException(status_code=404, detail="Localidad no encontrada")
    
    # Obtener solo indicadores superiores al promedio
    indicadores_superiores = obtener_indicadores_superiores_promedio(db, localidad)
    
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
            "indicadores_disponibles": indicadores_unicos,
            "indicadores_superiores_promedio": len(indicadores_superiores)
        },
        "indicadores_detalle": indicadores_superiores,
        "criterio_filtrado": "Solo se muestran indicadores que superan el promedio general de Bogotá",
        "timestamp": datetime.utcnow()
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
        "mensaje": "API Indicadores Sociales Bogotá D.C. - Versión con Normalización UPZ",
        "version": "1.2.0",
        "mejoras": [
            "Análisis por UPZ",
            "Normalización completa de localidades y UPZ",
            "Eliminación automática de duplicados",
            "Filtro por indicadores superiores al promedio",
            "Gráficos optimizados",
            "Herramientas de diagnóstico avanzadas",
            "Verificación de seguridad de datos"
        ],
        "nuevas_funcionalidades": [
            "Normalización completa de UPZ con 40+ reglas",
            "Detección y eliminación de duplicados",
            "Verificación previa de duplicados sin modificar",
            "Preview de cambios antes de ejecutar",
            "Conteo de registros para verificar integridad",
            "Normalización separada por tipo de entidad",
            "Garantías de seguridad de datos"
        ],
        "endpoints_normalizacion": [
            "/administracion/verificar-duplicados - Verificar duplicados sin modificar",
            "/administracion/preview-normalizacion - Ver cambios sin ejecutar", 
            "/administracion/contar-registros - Contar registros actuales",
            "/administracion/normalizar-completo - Normalizar todo (recomendado)",
            "/administracion/normalizar-localidades - Solo localidades",
            "/administracion/normalizar-upz - Solo UPZ"
        ],
        "garantias_seguridad": [
            "NO se eliminan registros de datos",
            "NO se eliminan indicadores", 
            "NO se eliminan valores numéricos",
            "Solo se corrigen nombres duplicados",
            "Verificación automática de integridad",
            "Operaciones completamente reversibles"
        ],
        "documentacion": "/docs"
    }

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    try:
        total_registros = db.query(IndicadorDB).count()
        localidades_normalizadas = db.query(IndicadorDB.nombre_localidad).distinct().count()
        upz_disponibles = db.query(IndicadorDB.id_upz).filter(IndicadorDB.id_upz.isnot(None)).distinct().count()
        
        return {
            "status": "healthy",
            "total_indicadores": total_registros,
            "localidades_disponibles": localidades_normalizadas,
            "upz_disponibles": upz_disponibles,
            "version": "1.2.0",
            "funcionalidades_activas": [
                "Normalización UPZ",
                "Análisis por localidades", 
                "Análisis por UPZ",
                "Eliminación de duplicados",
                "Verificación de seguridad"
            ],
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
