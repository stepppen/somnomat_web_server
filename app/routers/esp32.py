from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from app import schemas, crud
from app.database import get_db

router = APIRouter(prefix="/esp32", tags=["esp32"])

@router.post("/{device_id}/startup", status_code=200)
def esp32_startup(
    device_id: str,
    payload: schemas.ESPStatusPackage,
    db: Session = Depends(get_db)
):
    """
    ESP32 sends startup package on boot (ID 1001-1007).
    Contains: CustomName, Status, ActualPartition, versions, MAC.
    """
    device = crud.update_device_status(db, device_id, payload)
    return {
        "success": True,
        "message": f"Device {device_id} registered/updated",
        "device": schemas.DeviceOut.model_validate(device)
    }

@router.post("/{device_id}/sensors", status_code=201)
def post_sensor_reading(
    device_id: str,
    reading: schemas.SensorReading,
    db: Session = Depends(get_db)
):
    """
    ESP32 posts sensor readings (ID 1008-1011):
    - Temperature
    - SD Free Storage
    - Motor Status
    """
    # Override device_id from path
    reading_data = reading.model_dump()
    reading_data['device_id'] = device_id
    
    sensor_reading = crud.create_sensor_reading(db, device_id, reading_data)
    return {
        "success": True,
        "reading_id": sensor_reading.id
    }

@router.post("/{device_id}/sensors/batch", status_code=201)
def post_sensor_readings_batch(
    device_id: str,
    batch: schemas.BatchSensorReadings,
    db: Session = Depends(get_db)
):
    """
    ESP32 can post multiple sensor readings at once for efficiency.
    """
    readings_created = []
    for reading_data in batch.readings:
        reading = crud.create_sensor_reading(db, device_id, reading_data)
        readings_created.append(reading.id)
    
    return {
        "success": True,
        "count": len(readings_created),
        "reading_ids": readings_created
    }

@router.post("/{device_id}/bed/pressure", status_code=201)
def post_bed_pressure(
    device_id: str,
    adc_value_scaled: float,
    timestamp: datetime | None = None,
    db: Session = Depends(get_db)
):
    """
    ESP32 posts bed pressure/occupancy data (ADC scaled 0-1).
    Value < 0.2 typically indicates bed is occupied.
    """
    reading = crud.create_bed_pressure_reading(db, device_id, adc_value_scaled, timestamp)
    return {
        "success": True,
        "reading_id": reading.id
    }

@router.post("/{device_id}/bed/power", status_code=201)
def post_bed_power_event(
    device_id: str,
    event: schemas.BedPowerEvent,
    db: Session = Depends(get_db)
):
    """
    ESP32 posts bed power button state changes.
    button_on: "OFF->ON" or "ON->OFF"
    """
    power_event = crud.create_bed_power_event(
        db, 
        device_id, 
        event.button_on,
        event.timestamp
    )
    return {
        "success": True,
        "event_id": power_event.id
    }

@router.get("/{device_id}/poll")
def poll_for_commands(
    device_id: str,
    db: Session = Depends(get_db)
):
    """
    ESP32 polls this endpoint every 1-5 seconds to check for pending commands.
    Returns all pending commands for this device.
    
    ESP32 should then:
    1. Execute each command
    2. Call PATCH /api/v1/commands/{command_id}/status to mark completed/failed
    """
    # Update last_seen
    crud.get_or_create_device(db, device_id)
    
    commands = crud.get_pending_commands(db, device_id)
    
    return schemas.PollResponse(
        device_id=device_id,
        commands=[schemas.CommandOut.model_validate(c) for c in commands],
        timestamp=datetime.now(timezone.utc)
    )

@router.get("/{device_id}/status")
def get_device_status(
    device_id: str,
    db: Session = Depends(get_db)
):
    """
    Get current device status and latest readings.
    Useful for debugging and monitoring.
    """
    device = crud.get_device_by_id(db, device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    # Get latest sensor reading
    _, latest_sensors = crud.list_sensor_readings(db, device_id, limit=1)
    
    # Get latest pressure reading
    _, latest_pressure = crud.list_bed_pressure_readings(db, device_id, limit=1)
    
    # Get pending commands count
    pending_commands = crud.get_pending_commands(db, device_id)
    
    return {
        "device": schemas.DeviceOut.model_validate(device),
        "latest_sensor": schemas.SensorReadingOut.model_validate(latest_sensors[0]) if latest_sensors else None,
        "latest_pressure": latest_pressure[0].adc_value_scaled if latest_pressure else None,
        "pending_commands": len(pending_commands)
    }