web: bash start.sh
worker: python -c "print('No worker processes defined for v4.3.1')"
release: python -c "from main import Base, engine; Base.metadata.create_all(bind=engine); print('✅ Database tables created/verified for v4.3.1'); print('🆕 Features: Filtros corregidos, Theil mejorado, Dashboard responsive')"
