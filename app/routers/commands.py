from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app import schemas, crud
from app.database import get_db

router = APIRouter(prefix="/commands", tags=["commands"])

@router.patch("/{command_id}/status")
def update_command_status(
    command_id: int,
    payload: schemas.CommandStatusUpdate,
    db: Session = Depends(get_db)
):
    """
    ESP32 calls this after executing a command to mark it as completed/failed.
    This is called AFTER the ESP32 polls and receives the command.
    """
    command = crud.update_command_status(db, command_id, payload.status)
    if not command:
        raise HTTPException(status_code=404, detail="Command not found")
    return schemas.CommandOut.model_validate(command)