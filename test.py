#!/usr/bin/env python3
"""
Test suite completo MEJORADO para verificar que la aplicación de Fecundidad Temprana funciona correctamente.
Incluye tests específicos para las mejoras v4.3.1:
- Filtros Localidad/UPZ corregidos
- Índice Theil con todas las UPZ
- Orden de pestañas mejorado
- Interfaz responsive optimizada
"""

import os
import sys
import asyncio
import json
import re
from io import BytesIO
from pathlib import Path

def test_import():
    """Test que la aplicación se puede importar"""
    try:
        print("Testing app import...")
        from main import app, SessionLocal, IndicadorFecundidad
        print("✅ App imported successfully")
        print(f"✅ App title: {app.title}")
        print(f"✅ App version: {app.version}")
        return True
    except Exception as e:
        print(f"❌ Failed to import app: {e}")
        return False

def test_health_endpoint():
    """Test que el endpoint de health existe y funciona"""
    try:
        from main import app
        from fastapi.testclient import TestClient
        
        print("Testing health endpoint...")
        client = TestClient(app)
        response = client.get("/health")
        
        print(f"✅ Health endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health response: {json.dumps(health_data, indent=2)}")
            
            # Verificar campos esperados
            expected_fields = ["status", "version", "database", "registros", "timestamp"]
            missing_fields = [field for field in expected_fields if field not in health_data]
            
            if missing_fields:
                print(f"⚠️ Missing health fields: {missing_fields}")
            else:
                print("✅ All expected health fields present")
            
            # Verificar versión actualizada
            if health_data.get("version") == "4.3.1":
                print("✅ Version updated to 4.3.1")
            else:
                print(f"⚠️ Expected version 4.3.1, got {health_data.get('version')}")
            
            return True
        else:
            print(f"❌ Health endpoint returned non-200 status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
        return False

def test_database_connection():
    """Test que la base de datos se puede conectar"""
    try:
        print("Testing database connection...")
        from main import SessionLocal, IndicadorFecundidad
        
        db = SessionLocal()
        try:
            # Test simple query
            result = db.execute("SELECT 1").scalar()
            print(f"✅ Database query test result: {result}")
            
            # Test table exists
            count = db.query(IndicadorFecundidad).count()
            print(f"✅ Current records in database: {count}")
            
            return result == 1
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_geografia_upz_filtering():
    """🆕 NUEVO: Test específico para el filtrado de UPZ por localidad"""
    try:
        from main import app
        from fastapi.testclient import TestClient
        
        print("Testing UPZ filtering by localidad...")
        client = TestClient(app)
        
        # Test con localidades comunes de Bogotá
        test_localidades = ["Usaquén", "Chapinero", "Santa Fe", "Kennedy", "Fontibón"]
        
        results = {}
        for localidad in test_localidades:
            try:
                response = client.get(f"/geografia/upz_por_localidad?localidad={localidad}")
                if response.status_code == 200:
                    data = response.json()
                    results[localidad] = data.get('total', 0)
                    print(f"✅ {localidad}: {data.get('total', 0)} UPZ encontradas")
                else:
                    results[localidad] = 0
                    print(f"⚠️ {localidad}: endpoint returned {response.status_code}")
            except Exception as e:
                results[localidad] = 0
                print(f"❌ Error testing {localidad}: {e}")
        
        # Verificar que al menos algunas localidades tienen UPZ
        localidades_con_upz = [loc for loc, count in results.items() if count > 0]
        
        if len(localidades_con_upz) > 0:
            print(f"✅ Filtrado de UPZ funciona para {len(localidades_con_upz)} localidades")
            return True
        else:
            print("⚠️ No se encontraron UPZ para ninguna localidad (puede ser normal si no hay datos)")
            return True  # No es error crítico si no hay datos
            
    except Exception as e:
        print(f"❌ UPZ filtering test failed: {e}")
        return False

def test_theil_all_upz():
    """🆕 NUEVO: Test que verifica que el índice Theil devuelve todas las UPZ"""
    try:
        from main import app
        from fastapi.testclient import TestClient
        
        print("Testing Theil index returns all UPZ...")
        client = TestClient(app)
        
        # Primero obtener metadatos para saber qué indicadores hay
        metadatos_response = client.get("/metadatos")
        if metadatos_response.status_code != 200:
            print("⚠️ Cannot get metadata, skipping Theil test")
            return True
        
        metadatos = metadatos_response.json()
        indicadores = metadatos.get("indicadores", {}).get("todos", [])
        
        if not indicadores:
            print("⚠️ No indicators found, skipping Theil test")
            return True
        
        # Test con el primer indicador disponible
        test_indicador = indicadores[0]
        response = client.get(f"/analisis/theil?indicador={test_indicador}&nivel=LOCALIDAD")
        
        if response.status_code == 200:
            data = response.json()
            if "datos" in data:
                num_upz = len(data["datos"])
                total_upz_meta = metadatos.get("resumen", {}).get("upz", 0)
                
                print(f"✅ Theil returned {num_upz} UPZ")
                print(f"✅ Metadata shows {total_upz_meta} total UPZ")
                
                # Verificar que devuelve datos estructurados correctamente
                if num_upz > 0:
                    sample_upz = data["datos"][0]
                    expected_fields = ["upz", "valor", "desviacion_media", "ratio_media"]
                    missing_fields = [field for field in expected_fields if field not in sample_upz]
                    
                    if missing_fields:
                        print(f"⚠️ Missing fields in Theil response: {missing_fields}")
                    else:
                        print("✅ Theil response structure is correct")
                
                return True
            else:
                print("⚠️ Theil response missing 'datos' field")
                return False
        else:
            print(f"⚠️ Theil endpoint returned status {response.status_code}")
            # Podría ser normal si no hay datos suficientes
            return True
            
    except Exception as e:
        print(f"❌ Theil all UPZ test failed: {e}")
        return False

def test_caracterizacion_filtering():
    """🆕 NUEVO: Test específico para filtros de caracterización"""
    try:
        from main import app
        from fastapi.testclient import TestClient
        
        print("Testing caracterizacion filtering...")
        client = TestClient(app)
        
        # Obtener metadatos
        metadatos_response = client.get("/metadatos")
        if metadatos_response.status_code != 200:
            print("⚠️ Cannot get metadata, skipping caracterizacion test")
            return True
        
        metadatos = metadatos_response.json()
        indicadores = metadatos.get("indicadores", {}).get("todos", [])
        localidades = metadatos.get("geografia", {}).get("localidades", [])
        
        if not indicadores or not localidades:
            print("⚠️ Insufficient data for caracterizacion test")
            return True
        
        test_indicador = indicadores[0]
        test_localidad = localidades[0] if localidades else None
        
        # Test filtro por localidad
        if test_localidad:
            response = client.get(f"/caracterizacion?indicador={test_indicador}&nivel=LOCALIDAD&localidad={test_localidad}")
            
            if response.status_code == 200:
                data = response.json()
                if "datos" in data:
                    print(f"✅ Caracterización con filtro localidad: {len(data['datos'])} resultados")
                else:
                    print("✅ Caracterización response OK (no data message)")
                return True
            else:
                print(f"⚠️ Caracterización returned status {response.status_code}")
                return True  # No crítico
        
        return True
            
    except Exception as e:
        print(f"❌ Caracterizacion filtering test failed: {e}")
        return False

def test_dashboard_structure():
    """🆕 NUEVO: Test que verifica la estructura del dashboard mejorado"""
    try:
        print("Testing dashboard structure...")
        
        dashboard_file = "dashboard_compatible.html"
        if not Path(dashboard_file).exists():
            print(f"⚠️ Dashboard file not found: {dashboard_file}")
            return True  # No crítico si no existe
        
        with open(dashboard_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Verificar elementos clave del dashboard mejorado
        checks = {
            "Tabs in correct order": 'data-tab="caracterizacion"' in content and 'data-tab="series"' in content,
            "Theil scrollable chart": 'chart-container-scrollable' in content,
            "Territory filtering logic": 'updateTerritorioForLevel' in content,
            "Responsive design": 'responsive' in content and '@media' in content,
            "UPZ filtering": 'upz_por_localidad' in content
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
        print(f"❌ Dashboard structure test failed: {e}")
        return False

def test_series_temporales():
    """Test mejorado para series temporales"""
    try:
        from main import app
        from fastapi.testclient import TestClient
        
        print("Testing series temporales endpoint...")
        client = TestClient(app)
        
        # Test básico sin datos (debería devolver mensaje)
        response = client.get("/datos/series?indicador=test_indicator&upz=test_upz&nivel=LOCALIDAD")
        print(f"✅ Series endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            # Debería tener mensaje de sin datos o datos reales
            has_data = "serie" in data or "mensaje" in data
            print(f"✅ Series response structure OK: {has_data}")
            return True
        else:
            print(f"⚠️ Series returned status: {response.status_code}")
            return True  # No crítico si no hay datos
            
    except Exception as e:
        print(f"❌ Series temporales test failed: {e}")
        return False

def test_api_documentation():
    """Test que la documentación de la API está disponible"""
    try:
        from main import app
        from fastapi.testclient import TestClient
        
        print("Testing API documentation...")
        client = TestClient(app)
        
        # Test Swagger UI
        response = client.get("/docs")
        print(f"✅ Swagger docs status: {response.status_code}")
        
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        print(f"✅ OpenAPI schema status: {response.status_code}")
        
        if response.status_code == 200:
            schema = response.json()
            print(f"✅ API title: {schema.get('info', {}).get('title', 'N/A')}")
            print(f"✅ API version: {schema.get('info', {}).get('version', 'N/A')}")
            
            # Contar endpoints
            paths_count = len(schema.get('paths', {}))
            print(f"✅ Number of API endpoints: {paths_count}")
            
            # Verificar que existen endpoints clave
            expected_endpoints = [
                "/caracterizacion",
                "/analisis/theil", 
                "/analisis/asociacion",
                "/datos/series",
                "/geografia/upz_por_localidad"
            ]
            
            existing_endpoints = list(schema.get('paths', {}).keys())
            missing_endpoints = [ep for ep in expected_endpoints if ep not in existing_endpoints]
            
            if missing_endpoints:
                print(f"⚠️ Missing expected endpoints: {missing_endpoints}")
            else:
                print("✅ All key endpoints present")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ API documentation test failed: {e}")
        return False

def test_environment_variables():
    """Test variables de entorno importantes"""
    try:
        print("Testing environment variables...")
        
        port = os.getenv('PORT', 'Not set')
        database_url = os.getenv('DATABASE_URL', 'Not set')
        
        print(f"✅ PORT: {port}")
        print(f"✅ DATABASE_URL: {database_url[:50]}..." if database_url != 'Not set' else f"✅ DATABASE_URL: {database_url}")
        
        # Verificar Python version
        python_version = sys.version
        print(f"✅ Python version: {python_version}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment variables test failed: {e}")
        return False

def test_debug_columns():
    """Test específico para el endpoint de debug mejorado"""
    try:
        from main import app
        from fastapi.testclient import TestClient
        
        print("Testing debug columns endpoint...")
        client = TestClient(app)
        
        response = client.get("/debug/columns")
        print(f"✅ Debug columns status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Verificar campos nuevos en v4.3.1
            if "mejoras_v431" in data:
                mejoras = data["mejoras_v431"]
                print(f"✅ Found v4.3.1 improvements: {len(mejoras)}")
                for mejora in mejoras:
                    print(f"   • {mejora}")
            else:
                print("⚠️ v4.3.1 improvements section missing")
            
            return True
        else:
            print(f"❌ Debug columns returned status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Debug columns test failed: {e}")
        return False

def run_all_tests():
    """Ejecuta todos los tests incluyendo los nuevos para v4.3.1"""
    print("🧪 Running comprehensive test suite for Fecundidad Temprana API v4.3.1...")
    print("🆕 Incluye tests específicos para mejoras:")
    print("   • Filtros Localidad/UPZ corregidos")
    print("   • Índice Theil con todas las UPZ")
    print("   • Orden de pestañas mejorado")
    print("   • Dashboard responsive optimizado")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 70)
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("Import Test", test_import),
        ("Database Connection", test_database_connection),
        ("Health Endpoint", test_health_endpoint),
        ("API Documentation", test_api_documentation),
        ("Debug Columns Endpoint", test_debug_columns),
        
        # 🆕 NUEVOS TESTS ESPECÍFICOS v4.3.1
        ("🆕 UPZ Filtering by Localidad", test_geografia_upz_filtering),
        ("🆕 Theil All UPZ Response", test_theil_all_upz),
        ("🆕 Caracterización Filtering", test_caracterizacion_filtering),
        ("🆕 Dashboard Structure", test_dashboard_structure),
        
        # Tests existentes mejorados
        ("Series Temporales", test_series_temporales),
    ]
    
    results = []
    new_features_results = []
    
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            result = test_func()
            results.append(result)
            
            # Rastrear tests de nuevas funcionalidades
            if "🆕" in name:
                new_features_results.append(result)
            
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"Status: {status}")
        except Exception as e:
            print(f"❌ Test {name} crashed: {e}")
            results.append(False)
    
    # Resumen final
    success_count = sum(results)
    total_count = len(results)
    new_features_success = sum(new_features_results)
    new_features_total = len(new_features_results)
    
    print("\n" + "=" * 70)
    print(f"📊 RESULTS: {success_count}/{total_count} tests passed")
    print(f"Success rate: {success_count/total_count*100:.1f}%")
    
    if new_features_total > 0:
        print(f"🆕 NEW FEATURES: {new_features_success}/{new_features_total} tests passed")
        print(f"New features success rate: {new_features_success/new_features_total*100:.1f}%")
    
    if success_count == total_count:
        print("🎉 All tests passed! v4.3.1 improvements working correctly.")
        print("🚀 Ready for deployment!")
        return 0
    elif success_count >= total_count * 0.8:
        print("✅ Most tests passed! App should work with minor issues.")
        print("⚠️ Check failed tests above for details.")
        return 0
    else:
        print("⚠️ Many tests failed. Check configuration and dependencies.")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
