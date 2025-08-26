{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS",
    "buildCommand": "pip install --upgrade pip && pip install -r requirements.txt"
  },
  "deploy": {
    "startCommand": "bash start.sh",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 120,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  },
  "environment": {
    "PYTHONPATH": "/app",
    "PYTHONUNBUFFERED": "1",
    "PORT": "8000"
  },
  "regions": ["us-west1"],
  "plugins": ["postgresql"]
}
