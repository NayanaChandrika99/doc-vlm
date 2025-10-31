#!/usr/bin/env python3
"""
Milestone 1 Verification Script

Verifies that ML Platform Infrastructure is properly set up:
- Docker services are running
- Database schema is created
- Model registry is operational
- DVC is initialized
"""
import sys
import subprocess
import time
from pathlib import Path
import requests


def check_docker_service(service_name: str, port: int, health_endpoint: str = None) -> bool:
    """Check if a Docker service is running and healthy"""
    print(f"Checking {service_name}...", end=" ")
    
    # Check if container is running
    result = subprocess.run(
        ["docker", "ps", "--filter", f"name=raelm_{service_name}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )
    
    if f"raelm_{service_name}" not in result.stdout:
        print("❌ Container not running")
        return False
    
    # Check health endpoint if provided
    if health_endpoint:
        try:
            response = requests.get(health_endpoint, timeout=5)
            if response.status_code == 200:
                print("✅ Healthy")
                return True
            else:
                print(f"⚠️  Running but unhealthy (HTTP {response.status_code})")
                return False
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Running but not responding: {e}")
            return False
    else:
        print("✅ Running")
        return True


def check_database():
    """Check database connection and schema"""
    print("Checking database schema...", end=" ")
    
    result = subprocess.run(
        ["docker", "exec", "raelm_postgres", "psql", "-U", "raelm", "-d", "raelm", 
         "-c", "SELECT tablename FROM pg_tables WHERE schemaname='public';"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("❌ Failed to connect")
        return False
    
    required_tables = ['documents', 'datasets', 'annotations', 'annotation_queue', 
                      'model_versions', 'prompt_stats', 'inference_results']
    
    missing_tables = [t for t in required_tables if t not in result.stdout]
    
    if missing_tables:
        print(f"⚠️  Missing tables: {', '.join(missing_tables)}")
        return False
    
    print(f"✅ All {len(required_tables)} tables created")
    return True


def check_model_registry():
    """Check if model registry module is importable"""
    print("Checking model registry module...", end=" ")
    
    try:
        from ml_platform.model_registry import ModelRegistry
        registry = ModelRegistry(
            mlflow_uri="http://localhost:5000",
            artifact_root=Path("artifacts")
        )
        print("✅ Module loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import: {e}")
        return False


def check_dvc():
    """Check if DVC is initialized"""
    print("Checking DVC initialization...", end=" ")
    
    dvc_dir = Path(".dvc")
    if not dvc_dir.exists():
        print("❌ .dvc directory not found")
        return False
    
    config_file = dvc_dir / "config"
    if not config_file.exists():
        print("❌ .dvc/config not found")
        return False
    
    print("✅ DVC initialized")
    return True


def main():
    """Run all verification checks"""
    print("=" * 60)
    print("RaeLM Milestone 1 Verification")
    print("=" * 60)
    print()
    
    checks = []
    
    # Docker services
    print("📦 Docker Services:")
    checks.append(check_docker_service("postgres", 5432))
    checks.append(check_docker_service("redis", 6379))
    checks.append(check_docker_service("minio", 9000))
    checks.append(check_docker_service("mlflow", 5000, "http://localhost:5000/health"))
    checks.append(check_docker_service("prometheus", 9090))
    checks.append(check_docker_service("grafana", 3000))
    print()
    
    # Database
    print("🗄️  Database:")
    checks.append(check_database())
    print()
    
    # Python modules
    print("🐍 Python Modules:")
    checks.append(check_model_registry())
    print()
    
    # DVC
    print("📊 Data Version Control:")
    checks.append(check_dvc())
    print()
    
    # Summary
    print("=" * 60)
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"✅ All checks passed ({passed}/{total})")
        print("\nMilestone 1 complete! Ready for Milestone 2.")
        return 0
    else:
        print(f"⚠️  {passed}/{total} checks passed, {total - passed} failed")
        print("\nPlease fix failing checks before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

