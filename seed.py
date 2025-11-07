from app.database import SessionLocal
from app import crud, schemas

def seed():
    db = SessionLocal()
    try:
        # Create a test device
        crud.get_or_create_device(db, "esp32-bed-001", "Test Bed")
        
        # Create test reading
        crud.create_reading(db, schemas.ReadingCreate(
            device_id="esp32-bed-001",
            sensor="temperature",
            value=22.5
        ))
        
        print("✅ Seeded test data!")
    finally:
        db.close()

if __name__ == "__main__":
    seed()