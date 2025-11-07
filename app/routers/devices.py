from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app import schemas, crud
from app.database import get_db

router = APIRouter(prefix="/devices", tags=["devices"])

@router.get("", response_model=list[schemas.DeviceOut])
def list_devices(limit: int = 100, offset: int = 0, db: Session = Depends(get_db)):
    devices = db.query(crud.models.Device).offset(offset).limit(limit).all()
    # Use from_orm instead of model_validate
    return [schemas.DeviceOut.from_orm(d) for d in devices]

@router.post("", response_model=schemas.DeviceOut)
def create_device(device: schemas.DeviceCreate, db: Session = Depends(get_db)):
    if crud.get_device_by_id(db, device.device_id):
        raise HTTPException(status_code=400, detail="Device already exists")
    return crud.create_device(db, device)