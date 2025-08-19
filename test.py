#!/usr/bin/env python3
"""
Test simple para verificar que la aplicación se puede importar y crear correctamente.
Útil para debugging en Railway.
"""

import os
import sys

def test_import():
    """Test que la aplicación se puede importar"""
    try:
        print("Testing app import...")
        from main import app
        print("✅ App imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import app: {e}")
        return False

def test_health_endpoint():
    """Test que el endpoint de health existe"""
    try:
        from main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/health")
        print(f"✅ Health endpoint status: {response.status_code}")
        print(f"✅ Health response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
        return False

def test_database_connection():
    """Test que la base de datos se puede conectar"""
    try:
        from main import SessionLocal
        db = SessionLocal()
        # Test simple
        result = db.execute("SELECT 1").scalar()
        db.close()
        print(f"✅ Database test result: {result}")
        return result == 1
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Fecundidad Temprana API startup...")
    print(f"Python version: {sys.version}")
    print(f"PORT: {os.getenv('PORT', 'Not set')}")
    print(f"DATABASE_URL: {os.getenv('DATABASE_URL', 'Not set')[:50]}...")
    
    tests = [
        ("Import test", test_import),
        ("Health endpoint test", test_health_endpoint), 
        ("Database connection test", test_database_connection)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {name} crashed: {e}")
            results.append(False)
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n🏁 Results: {success_count}/{total_count} tests passed")
    
    if success_count == total_count:
        print("🎉 All tests passed! App should start correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check logs above.")
        sys.exit(1)
