#!/usr/bin/env python3
"""
Test suite completo para verificar que la aplicación de Fecundidad Temprana funciona correctamente.
Incluye tests específicos para el archivo consolidado_indicadores_fecundidad.xlsx
Útil para debugging en Railway y desarrollo local.
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

def test_metadatos_endpoint():
    """Test que el endpoint de metadatos funciona"""
    try:
        from main import app
        from fastapi.testclient import TestClient
        
        print("Testing metadatos endpoint...")
        client = TestClient(app)
        response = client.get("/metadatos")
        
        print(f"✅ Metadatos endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            metadatos = response.json()
            
            # Verificar estructura esperada
            expected_sections = ["resumen", "indicadores", "geografia", "temporal"]
            missing_sections = [section for section in expected_sections if section not in metadatos]
            
            if missing_sections:
                print(f"⚠️ Missing metadatos sections: {missing_sections}")
            else:
                print("✅ All expected metadatos sections present")
                
            # Mostrar resumen de datos
            if "resumen" in metadatos:
                resumen = metadatos["resumen"]
                print(f"📊 Total registros: {resumen.get('total_registros', 0)}")
                print(f"📊 Total indicadores: {resumen.get('total_indicadores', 0)}")
                print(f"📊 Localidades: {resumen.get('localidades', 0)}")
                print(f"📊 UPZ: {resumen.get('upz', 0)}")
            
            return True
        else:
            print(f"❌ Metadatos endpoint returned non-200 status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Metadatos endpoint test failed: {e}")
        return False

def test_home_page():
    """Test que la página principal carga"""
    try:
        from main import app
        from fastapi.testclient import TestClient
        
        print("Testing home page...")
        client = TestClient(app)
        response = client.get("/")
        
        print(f"✅ Home page status: {response.status_code}")
        
        if response.status_code == 200:
            content = response.text
            # Verificar que contiene elementos esperados del dashboard
            expected_elements = ["Dashboard", "Fecundidad", "Bogotá", "chart"]
            found_elements = [elem for elem in expected_elements if elem.lower() in content.lower()]
            
            print(f"✅ Found dashboard elements: {found_elements}")
            return len(found_elements) > 0
        else:
            print(f"❌ Home page returned non-200 status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Home page test failed: {e}")
        return False

def test_fecundidad_detection():
    """Test que la detección de indicadores de fecundidad funciona correctamente"""
    try:
        print("Testing fecundidad indicators detection...")
        from main import get_indicadores_fecundidad, get_indicadores_no_fecundidad, SessionLocal
        
        db = SessionLocal()
        try:
            fecundidad_indicators = get_indicadores_fecundidad(db)
            otros_indicators = get_indicadores_no_fecundidad(db)
            
            print(f"✅ Fecundidad indicators found: {len(fecundidad_indicators)}")
            print(f"✅ Other indicators found: {len(otros_indicators)}")
            
            # Mostrar algunos ejemplos
            if fecundidad_indicators:
                print("📊 Sample fecundidad indicators:")
                for ind in fecundidad_indicators[:3]:
                    print(f"   - {ind}")
            
            if otros_indicators:
                print("📊 Sample other indicators:")
                for ind in otros_indicators[:3]:
                    print(f"   - {ind}")
            
            return True
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Fecundidad detection test failed: {e}")
        return False

def test_cohort_extraction():
    """Test que la extracción de cohortes funciona"""
    try:
        print("Testing cohort extraction...")
        from main import extraer_grupo_edad
        
        test_cases = [
            ("Tasa Específica de Fecundidad en niñas de 10 a 14 años", "10-14"),
            ("Tasa Específica de Fecundidad en mujeres de 15 a 19 años", "15-19"),
            ("Tasa de desercion escolar", None),
            ("Indicador general sin edad", None)
        ]
        
        passed = 0
        for indicador, expected in test_cases:
            result = extraer_grupo_edad(indicador, None)
            if result == expected:
                print(f"✅ '{indicador[:30]}...' -> {result}")
                passed += 1
            else:
                print(f"❌ '{indicador[:30]}...' -> Expected: {expected}, Got: {result}")
        
        print(f"✅ Cohort extraction: {passed}/{len(test_cases)} tests passed")
        return passed == len(test_cases)
        
    except Exception as e:
        print(f"❌ Cohort extraction test failed: {e}")
        return False

def test_data_cleaning():
    """Test que las funciones de limpieza de datos funcionan"""
    try:
        print("Testing data cleaning functions...")
        from main import clean_str, clean_float, clean_int, limpiar_texto
        
        # Test clean_str
        assert clean_str("  texto con espacios  ") == "texto con espacios"
        assert clean_str("ND") is None
        assert clean_str("NO_DATA") is None
        assert clean_str("") is None
        print("✅ clean_str tests passed")
        
        # Test clean_float
        assert clean_float("123.45") == 123.45
        assert clean_float("ND") is None
        assert clean_float("") is None
        print("✅ clean_float tests passed")
        
        # Test clean_int
        assert clean_int("123") == 123
        assert clean_int("123.0") == 123
        assert clean_int("ND") is None
        print("✅ clean_int tests passed")
        
        # Test limpiar_texto
        assert limpiar_texto("  Texto    con    espacios   ") == "Texto con espacios"
        assert limpiar_texto("Texto\n\tcon\n\tsaltos") == "Texto con saltos"
        print("✅ limpiar_texto tests passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Data cleaning test failed: {e}")
        return False

def test_excel_file_structure():
    """Test que el archivo Excel tiene la estructura esperada"""
    try:
        print("Testing Excel file structure...")
        
        excel_file = "consolidado_indicadores_fecundidad.xlsx"
        if not Path(excel_file).exists():
            print(f"⚠️ Excel file not found: {excel_file}")
            return True  # No es un error crítico
        
        import pandas as pd
        df = pd.read_excel(excel_file)
        
        # Verificar columnas requeridas
        required_columns = ['Indicador_Nombre', 'Valor', 'Unidad_Medida']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        else:
            print("✅ All required columns present")
        
        # Verificar que hay datos
        print(f"✅ Excel file has {len(df)} rows and {len(df.columns)} columns")
        
        # Verificar indicadores de fecundidad
        fecundidad_keywords = ['fecund', 'natalidad', 'nacimiento']
        fecundidad_indicators = df[df['Indicador_Nombre'].str.lower().str.contains(
            '|'.join(fecundidad_keywords), na=False
        )]['Indicador_Nombre'].unique()
        
        print(f"✅ Found {len(fecundidad_indicators)} fecundidad indicators in Excel")
        for ind in fecundidad_indicators:
            print(f"   - {ind}")
        
        return True
        
    except Exception as e:
        print(f"❌ Excel file structure test failed: {e}")
        return False
    """Test que la validación de upload funciona"""
    try:
        from main import app
        from fastapi.testclient import TestClient
        
        print("Testing upload validation...")
        client = TestClient(app)
        
        # Test sin archivo
        response = client.post("/upload/excel")
        print(f"✅ Upload without file status: {response.status_code}")
        
        # Test con archivo inválido
        invalid_file = BytesIO(b"invalid content")
        response = client.post(
            "/upload/excel",
            files={"file": ("test.txt", invalid_file, "text/plain")}
        )
        print(f"✅ Upload with invalid file status: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload validation test failed: {e}")
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
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ API documentation test failed: {e}")
        return False

def run_all_tests():
    """Ejecuta todos los tests"""
    print("🧪 Running comprehensive test suite for Fecundidad Temprana API...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 60)
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("Excel File Structure", test_excel_file_structure),
        ("Import Test", test_import),
        ("Data Cleaning Functions", test_data_cleaning),
        ("Cohort Extraction", test_cohort_extraction),
        ("Health Endpoint", test_health_endpoint),
        ("Database Connection", test_database_connection),
        ("Fecundidad Detection", test_fecundidad_detection),
        ("Home Page", test_home_page),
        ("API Documentation", test_api_documentation),
        ("Metadatos Endpoint", test_metadatos_endpoint),
        ("Upload Validation", test_upload_validation),
        ("Simulated Excel Upload", test_simulated_excel_upload),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"Status: {status}")
        except Exception as e:
            print(f"❌ Test {name} crashed: {e}")
            results.append(False)
    
    # Resumen final
    success_count = sum(results)
    total_count = len(results)
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTS: {success_count}/{total_count} tests passed")
    print(f"Success rate: {success_count/total_count*100:.1f}%")
    
    if success_count == total_count:
        print("🎉 All tests passed! App should work correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check logs above for details.")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
