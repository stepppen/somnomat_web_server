from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime
from app import schemas, crud
from app.database import get_db

router = APIRouter(prefix="/readings", tags=["readings"])

@router.get("", response_model=schemas.Paginated)
def list_readings(
    device_id: str | None = None,
    sensor: str | None = None,
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    since: datetime | None = None,
    until: datetime | None = None,
    db: Session = Depends(get_db)
):
    """Get readings with optional filters"""
    total, items = crud.list_readings(
        db, 
        device_id=device_id,
        sensor=sensor,
        limit=limit,
        offset=offset,
        since=since,
        until=until
    )
    return schemas.Paginated(
        total=total,
        items=[schemas.ReadingOut.model_validate(r) for r in items]
    )

@router.post("", response_model=schemas.ReadingOut, status_code=201)
def create_reading(
    payload: schemas.ReadingCreate,
    db: Session = Depends(get_db)
):
    """ESP32 posts sensor readings here"""
    reading = crud.create_reading(db, payload)
    return schemas.ReadingOut.model_validate(reading)

@router.post("/batch", status_code=201)
def create_readings_batch(
    payloads: list[schemas.ReadingCreate],
    db: Session = Depends(get_db)
):
    """ESP32 can post multiple readings at once"""
    readings = []
    for payload in payloads:
        reading = crud.create_reading(db, payload)
        readings.append(reading)
    
    return {
        "success": True,
        "count": len(readings),
        "message": f"Created {len(readings)} readings"
    }
