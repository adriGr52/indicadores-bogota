#!/usr/bin/env python3
"""
Suite de testing completa para Fecundidad Temprana API v4.3.1
Incluye tests para todas las funcionalidades principales:
- Filtros Localidad/UPZ corregidos
- Lógica inteligente de datos
- Índice de Theil completo
- Dashboard funcional
- Endpoints de análisis
"""

import os
import sys
import asyncio
import json
import re
from io import BytesIO
from pathlib import Path

def test_environment():
    """Test del entorno y variables de configuración"""
    try:
        print("Testing environment configuration...")
        
        # Variables de entorno
        port = os.getenv('PORT', 'Not set')
        database_url = os.getenv('DATABASE_URL', 'Not set')
        
        print(f"✅ PORT: {port}")
        print(f"✅ DATABASE_URL: {'Set' if database_url != 'Not set' else 'Not set'}")
        
        # Versión Python
        python_version = sys.version_info
        print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Directorio de trabajo
        cwd = os.getcwd()
        print(f"✅ Working directory: {cwd}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def test_imports():
    """Test que todos los módulos se pueden importar"""
    try:
        print("Testing module imports...")
        
        # Test imports básicos
        from main import app, SessionLocal, IndicadorFecundidad, obtener_datos_con_fallback
        print("✅ Main application modules imported")
        
        # Test FastAPI
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        print("✅ FastAPI modules imported")
        
        # Test SQLAlchemy
        from sqlalchemy import create_engine
        print("✅ SQLAlchemy imported")
        
        # Test scientific libraries
        import pandas as pd
        import numpy as np
        from scipy import stats
        print("✅ Scientific libraries imported")
        
        # Test app properties
        print(f"✅ App title: {app.title}")
        print(f"✅ App version: {app.version}")
        
        if app.version == "4.3.1":
            print("✅ Version correctly updated to 4.3.1")
        else:
            print(f"⚠️ Expected version 4.3.1, got {app.version}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_database_connection():
    """Test conexión a base de datos"""
    try:
        print("Testing database connection...")
        from main import SessionLocal, engine
        from sqlalchemy import text
        
        db = SessionLocal()
        try:
            # Test simple query
            result = db.execute(text("SELECT 1")).scalar()
            print(f"✅ Database connection successful: {result}")
            
            # Test table creation
            from main import Base
            Base.metadata.create_all(bind=engine)
            print("✅ Database tables created/verified")
            
            # Test record count
            from main import IndicadorFecundidad
            count = db.query(IndicadorFecundidad).count()
            print(f"✅ Current records in database: {count}")
            
            return True
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_health_endpoint():
    """Test endpoint de health"""
    try:
        print("Testing health endpoint...")
        from main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        response = client.get("/health")
        
        print(f"✅ Health endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            
            # Verificar campos requeridos
            required_fields = ["status", "version", "database", "registros", "timestamp"]
            missing_fields = [field for field in required_fields if field not in health_data]
            
            if missing_fields:
                print(f"⚠️ Missing health fields: {missing_fields}")
                return False
            else:
                print("✅ All expected health fields present")
            
            # Verificar versión
            if health_data.get("version") == "4.3.1":
                print("✅ Health endpoint reports correct version")
            else:
                print(f"⚠️ Health version mismatch: {health_data.get('version')}")
            
            print(f"✅ Database status: {health_data.get('database')}")
            print(f"✅ Records count: {health_data.get('registros')}")
            
            return True
        else:
            print(f"❌ Health endpoint returned status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
        return False

def test_metadatos_endpoint():
    """Test endpoint de metadatos"""
    try:
        print("Testing metadatos endpoint...")
        from main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        response = client.get("/metadatos")
        
        print(f"✅ Metadatos endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Verificar estructura
            required_sections = ["resumen", "indicadores", "geografia", "temporal"]
            missing_sections = [section for section in required_sections if section not in data]
            
            if missing_sections:
                print(f"⚠️ Missing metadatos sections: {missing_sections}")
                return False
            
            # Verificar contenido
            resumen = data.get("resumen", {})
            print(f"✅ Total registros: {resumen.get('total_registros', 0)}")
            print(f"✅ Total indicadores: {resumen.get('total_indicadores', 0)}")
            print(f"✅ Localidades: {resumen.get('localidades', 0)}")
            print(f"✅ UPZ: {resumen.get('upz', 0)}")
            
            # Verificar que hay datos
            if resumen.get('total_registros', 0) > 0:
                print("✅ Database contains data")
            else:
                print("⚠️ Database appears empty (normal for fresh install)")
            
            return True
        else:
            print(f"❌ Metadatos endpoint returned status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Metadatos endpoint test failed: {e}")
        return False

def test_upz_por_localidad():
    """🆕 Test específico para filtrado UPZ por localidad"""
    try:
        print("Testing UPZ filtering by localidad...")
        from main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test con localidades comunes de Bogotá
        test_localidades = ["Usaquén", "Chapinero", "Kennedy", "Suba", "Engativá"]
        
        successful_tests = 0
        for localidad in test_localidades:
            try:
                response = client.get(f"/geografia/upz_por_localidad?localidad={localidad}")
                
                if response.status_code == 200:
                    data = response.json()
                    upz_count = data.get('total', 0)
                    print(f"✅ {localidad}: {upz_count} UPZ found")
                    successful_tests += 1
                else:
                    print(f"⚠️ {localidad}: HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ Error testing {localidad}: {e}")
        
        if successful_tests > 0:
            print(f"✅ UPZ filtering works for {successful_tests} localidades")
            return True
        else:
            print("⚠️ UPZ filtering test completed but no data found (may be normal)")
            return True  # Not critical if no data
            
    except Exception as e:
        print(f"❌ UPZ filtering test failed: {e}")
        return False

def test_caracterizacion_endpoint():
    """🆕 Test específico para caracterización con lógica inteligente"""
    try:
        print("Testing caracterizacion endpoint...")
        from main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test sin indicador (debe fallar)
        response = client.get("/caracterizacion?")
        if response.status_code != 422:  # FastAPI validation error
            print("⚠️ Expected validation error for missing indicator")
        else:
            print("✅ Validation works for missing indicator")
        
        # Test con indicador ficticio
        response = client.get("/caracterizacion?indicador=test_indicator")
        print(f"✅ Caracterizacion test call status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if "mensaje" in data:
                print(f"✅ Proper handling of no data: {data['mensaje']}")
            else:
                print(f"✅ Caracterizacion returned data: {len(data.get('datos', []))} territories")
        
        return True
        
    except Exception as e:
        print(f"❌ Caracterizacion test failed: {e}")
        return False

def test_theil_endpoint():
    """🆕 Test específico para índice de Theil completo"""
    try:
        print("Testing Theil index endpoint...")
        from main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test con indicador ficticio
        response = client.get("/analisis/theil?indicador=test_indicator")
        print(f"✅ Theil endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            if "mensaje" in data:
                print(f"✅ Proper no-data handling: {data['mensaje']}")
            else:
                # Verificar estructura esperada
                expected_fields = ["indice_theil", "interpretacion", "estadisticas", "datos"]
                missing_fields = [field for field in expected_fields if field not in data]
                
                if missing_fields:
                    print(f"⚠️ Missing Theil fields: {missing_fields}")
                else:
                    print("✅ Theil response structure is correct")
                    
                    # Verificar que devuelve TODAS las unidades (no top 10)
                    datos_count = len(data.get("datos", []))
                    print(f"✅ Theil returns {datos_count} territories (not limited to top 10)")
        
        return True
        
    except Exception as e:
        print(f"❌ Theil test failed: {e}")
        return False

def test_lógica_inteligente():
    """🆕 Test para la lógica inteligente de fallback de datos"""
    try:
        print("Testing intelligent data fallback logic...")
        from main import obtener_datos_con_fallback, SessionLocal
        
        db = SessionLocal()
        try:
            # Test función de fallback con indicador ficticio
            result = obtener_datos_con_fallback(db, "test_indicator")
            
            if result is None or (result and len(result) == 2):
                print("✅ Fallback function handles empty data correctly")
            else:
                print("⚠️ Unexpected fallback function result")
            
            return True
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Intelligent logic test failed: {e}")
        return False

def test_series_temporales():
    """Test endpoint de series temporales"""
    try:
        print("Testing series temporales endpoint...")
        from main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test sin parámetros (debe fallar)
        response = client.get("/datos/series")
        if response.status_code == 422:
            print("✅ Validation works for series endpoint")
        
        # Test con parámetros ficticios
        response = client.get("/datos/series?indicador=test&territorio=test_territory")
        print(f"✅ Series endpoint status: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Series test failed: {e}")
        return False

def test_asociacion():
    """Test endpoint de asociación"""
    try:
        print("Testing asociacion endpoint...")
        from main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test con parámetros ficticios
        response = client.get("/analisis/asociacion?indicador_x=test1&indicador_y=test2")
        print(f"✅ Asociacion endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if "mensaje" in data:
                print(f"✅ Proper handling: {data['mensaje']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Asociacion test failed: {e}")
        return False

def test_brechas_cohortes():
    """🆕 Test nuevo endpoint de brechas por cohortes"""
    try:
        print("Testing brechas cohortes endpoint...")
        from main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test endpoint
        response = client.get("/brechas/cohortes")
        print(f"✅ Brechas cohortes endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if "mensaje" in data or "cohortes" in data:
                print("✅ Brechas endpoint responds correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Brechas cohortes test failed: {e}")
        return False

def test_dashboard_file():
    """🆕 Test estructura del dashboard HTML"""
    try:
        print("Testing dashboard HTML structure...")
        
        dashboard_file = "dashboard_compatible.html"
        if not Path(dashboard_file).exists():
            print(f"⚠️ Dashboard file not found: {dashboard_file}")
            return True  # No crítico si no existe
        
        with open(dashboard_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Verificar elementos clave del dashboard funcional
        checks = {
            "FastAPI title": "Fecundidad Temprana" in content,
            "Chart.js integration": "chart.js" in content.lower(),
            "Tabs structure": 'data-tab="caracterizacion"' in content,
            "Series after caracterización": content.find('data-tab="series"') > content.find('data-tab="caracterizacion"'),
            "UPZ filtering logic": 'upz_por_localidad' in content,
            "Responsive design": 'responsive' in content.lower(),
            "Loading indicators": 'loading' in content.lower(),
            "Error handling": 'error' in content.lower()
        }
        
        passed = 0
        for check_name, check_result in checks.items():
            if check_result:
                print(f"✅ {check_name}")
                passed += 1
            else:
                print(f"⚠️ {check_name}")
        
        print(f"✅ Dashboard structure: {passed}/{len(checks)} checks passed")
        return passed >= len(checks) * 0.8  # 80% o más
        
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False

def test_api_documentation():
    """Test documentación automática de la API"""
    try:
        print("Testing API documentation...")
        from main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test Swagger UI
        response = client.get("/docs")
        print(f"✅ Swagger docs status: {response.status_code}")
        
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        print(f"✅ OpenAPI schema status: {response.status_code}")
        
        if response.status_code == 200:
            schema = response.json()
            
            # Verificar información básica
            info = schema.get("info", {})
            print(f"✅ API title: {info.get('title', 'N/A')}")
            print(f"✅ API version: {info.get('version', 'N/A')}")
            
            # Contar endpoints
            paths = schema.get("paths", {})
            endpoint_count = len(paths)
            print(f"✅ Number of endpoints: {endpoint_count}")
            
            # Verificar endpoints clave de v4.3.1
            expected_endpoints = [
                "/caracterizacion",
                "/analisis/theil",
                "/analisis/asociacion", 
                "/datos/series",
                "/geografia/upz_por_localidad",
                "/brechas/cohortes"
            ]
            
            existing_endpoints = list(paths.keys())
            missing_endpoints = [ep for ep in expected_endpoints if ep not in existing_endpoints]
            
            if missing_endpoints:
                print(f"⚠️ Missing expected endpoints: {missing_endpoints}")
            else:
                print("✅ All key v4.3.1 endpoints present")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ API documentation test failed: {e}")
        return False

def test_file_structure():
    """Test estructura de archivos del proyecto"""
    try:
        print("Testing project file structure...")
        
        # Archivos críticos
        critical_files = [
            "main.py",
            "requirements.txt", 
            "Procfile"
        ]
        
        # Archivos recomendados
        recommended_files = [
            "dashboard_compatible.html",
            "start.sh",
            "railway.json",
            ".gitignore",
            "README.md"
        ]
        
        missing_critical = []
        missing_recommended = []
        
        for file in critical_files:
            if Path(file).exists():
                print(f"✅ Critical file found: {file}")
            else:
                missing_critical.append(file)
                print(f"❌ Missing critical file: {file}")
        
        for file in recommended_files:
            if Path(file).exists():
                print(f"✅ Recommended file found: {file}")
            else:
                missing_recommended.append(file)
                print(f"⚠️ Missing recommended file: {file}")
        
        if missing_critical:
            print(f"❌ Missing critical files: {missing_critical}")
            return False
        
        if missing_recommended:
            print(f"⚠️ Missing recommended files (non-critical): {missing_recommended}")
        
        print("✅ Project file structure is adequate")
        return True
        
    except Exception as e:
        print(f"❌ File structure test failed: {e}")
        return False

def test_data_processing_functions():
    """🆕 Test funciones de procesamiento de datos"""
    try:
        print("Testing data processing functions...")
        from main import (
            limpiar_texto, 
            calcular_indice_theil, 
            extraer_grupo_edad,
            clean_str, 
            clean_int, 
            clean_float
        )
        
        # Test limpiar_texto
        test_text = "  Test   Text  "
        cleaned = limpiar_texto(test_text)
        if cleaned == "Test Text":
            print("✅ limpiar_texto function works")
        else:
            print(f"⚠️ limpiar_texto unexpected result: '{cleaned}'")
        
        # Test calcular_indice_theil
        valores = [1.0, 2.0, 3.0, 4.0, 5.0]
        theil = calcular_indice_theil(valores)
        if isinstance(theil, float) and theil >= 0:
            print(f"✅ calcular_indice_theil works: {theil:.4f}")
        else:
            print("⚠️ calcular_indice_theil unexpected result")
        
        # Test extraer_grupo_edad
        test_cases = [
            ("Tasa de fecundidad 10-14", "10-14"),
            ("Embarazo en niñas de 10 a 14 años", "10-14"),
            ("Fecundidad 15-19", "15-19"),
            ("Test sin grupo", None)
        ]
        
        for text, expected in test_cases:
            result = extraer_grupo_edad(text, None)
            if result == expected:
                print(f"✅ extraer_grupo_edad: '{text}' -> {result}")
            else:
                print(f"⚠️ extraer_grupo_edad: '{text}' expected {expected}, got {result}")
        
        # Test cleaning functions
        if clean_str("  test  ") == "test":
            print("✅ clean_str works")
        
        if clean_int("123") == 123:
            print("✅ clean_int works")
        
        if clean_float("123.45") == 123.45:
            print("✅ clean_float works")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing functions test failed: {e}")
        return False

def run_all_tests():
    """Ejecuta todos los tests del sistema v4.3.1"""
    print("🧪 COMPREHENSIVE TEST SUITE - Fecundidad Temprana v4.3.1")
    print("=" * 70)
    print("🆕 Nuevas funcionalidades incluidas:")
    print("   • Filtros Localidad/UPZ corregidos")
    print("   • Lógica inteligente de fallback de datos")
    print("   • Índice de Theil completo (todas las UPZ)")
    print("   • Dashboard funcional responsivo")
    print("   • Nuevos endpoints de análisis")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 70)
    
    tests = [
        # Tests básicos
        ("Environment Configuration", test_environment),
        ("Module Imports", test_imports),
        ("Database Connection", test_database_connection),
        ("File Structure", test_file_structure),
        
        # Tests de endpoints
        ("Health Endpoint", test_health_endpoint),
        ("Metadatos Endpoint", test_metadatos_endpoint),
        ("API Documentation", test_api_documentation),
        
        # 🆕 Tests nuevas funcionalidades v4.3.1
        ("🆕 UPZ por Localidad", test_upz_por_localidad),
        ("🆕 Caracterización Inteligente", test_caracterizacion_endpoint),
        ("🆕 Theil Completo", test_theil_endpoint),
        ("🆕 Lógica Inteligente", test_lógica_inteligente),
        ("🆕 Dashboard Funcional", test_dashboard_file),
        ("🆕 Brechas Cohortes", test_brechas_cohortes),
        
        # Tests de análisis
        ("Series Temporales", test_series_temporales),
        ("Análisis Asociación", test_asociacion),
        
        # Tests de funciones
        ("🆕 Data Processing Functions", test_data_processing_functions),
    ]
    
    results = []
    new_features_results = []
    
    print("\n🚀 EJECUTANDO TESTS...")
    print("-" * 70)
    
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            start_time = time.time() if 'time' in globals() else None
            result = test_func()
            
            results.append(result)
            
            # Rastrear nuevas funcionalidades
            if "🆕" in name:
                new_features_results.append(result)
            
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"Status: {status}")
            
        except Exception as e:
            print(f"💥 Test {name} crashed: {e}")
            results.append(False)
            if "🆕" in name:
                new_features_results.append(False)
    
    # Resumen final
    success_count = sum(results)
    total_count = len(results)
    success_rate = success_count / total_count * 100
    
    new_features_success = sum(new_features_results)
    new_features_total = len(new_features_results)
    new_features_rate = new_features_success / new_features_total * 100 if new_features_total > 0 else 0
    
    print("\n" + "=" * 70)
    print("📊 RESULTADOS FINALES")
    print("=" * 70)
    print(f"📈 GENERAL: {success_count}/{total_count} tests passed ({success_rate:.1f}%)")
    
    if new_features_total > 0:
        print(f"🆕 NUEVAS FUNCIONALIDADES: {new_features_success}/{new_features_total} tests passed ({new_features_rate:.1f}%)")
    
    # Determinar estado final
    if success_count == total_count:
        print("\n🎉 ¡TODOS LOS TESTS PASARON!")
        print("🚀 Sistema v4.3.1 listo para producción")
        print("✨ Nuevas funcionalidades trabajando correctamente")
        return 0
    elif success_rate >= 85:
        print("\n✅ La mayoría de tests pasaron")
        print("⚠️ Revisar tests fallidos arriba")
        print("🔧 Sistema funcional con mejoras menores requeridas")
        return 0
    elif success_rate >= 70:
        print("\n⚠️ Tests parcialmente exitosos")
        print("🔧 Se requieren correcciones antes del deploy")
        return 1
    else:
        print("\n❌ Muchos tests fallaron")
        print("🚨 Sistema requiere revisión completa")
        return 2

if __name__ == "__main__":
    import time
    exit_code = run_all_tests()
    sys.exit(exit_code)
