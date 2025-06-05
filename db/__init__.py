from db.database import engine, Base

# Create tables
Base.metadata.create_all(bind=engine)
