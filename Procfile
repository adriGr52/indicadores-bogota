web: uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --log-level info
worker: python -c "print('No worker processes defined')"
release: python -c "from main import Base, engine; Base.metadata.create_all(bind=engine); print('Database tables created/verified')"
