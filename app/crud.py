from sqlalchemy.orm import Session
from sqlalchemy import select, func
from datetime import datetime, timezone, timedelta
from app import models, schemas

# ----- Devices -----
def get_or_create_device(
    db: Session, 
    device_id: str, 
    name: str | None = None
) -> models.Device:
    device = db.execute(
        select(models.Device).where(models.Device.device_id == device_id)
    ).scalar_one_or_none()
    
    if device:
        if name is not None and device.name != name:
            device.name = name
            db.add(device)
            db.commit()
            db.refresh(device)
        return device
    
    device = models.Device(device_id=device_id, name=name)
    db.add(device)
    db.commit()
    db.refresh(device)
    return device

def list_devices(db: Session, limit: int = 100, offset: int = 0):
    stmt = select(models.Device).order_by(models.Device.id.asc()).limit(limit).offset(offset)
    total = db.execute(select(func.count()).select_from(models.Device)).scalar_one()
    items = db.execute(stmt).scalars().all()
    return total, items

# ----- Readings -----
def create_reading(db: Session, payload: schemas.ReadingCreate) -> models.Reading:
    device = get_or_create_device(db, device_id=payload.device_id)
    ts = payload.timestamp or datetime.now(timezone.utc)
    reading = models.Reading(
        device_id=device.id,
        sensor=payload.sensor,
        value=payload.value,
        timestamp=ts
    )
    db.add(reading)
    db.commit()
    db.refresh(reading)
    return reading

def list_readings(
    db: Session,
    device_id: str | None = None,
    sensor: str | None = None,
    limit: int = 100,
    offset: int = 0,
    since: datetime | None = None,
    until: datetime | None = None,
):
    stmt = select(models.Reading)
    
    if device_id:
        dev = db.execute(
            select(models.Device).where(models.Device.device_id == device_id)
        ).scalar_one_or_none()
        if dev:
            stmt = stmt.where(models.Reading.device_id == dev.id)
        else:
            return 0, []
    
    if sensor:
        stmt = stmt.where(models.Reading.sensor == sensor)
    if since:
        stmt = stmt.where(models.Reading.timestamp >= since)
    if until:
        stmt = stmt.where(models.Reading.timestamp <= until)
    
    total = db.execute(
        select(func.count()).select_from(stmt.subquery())
    ).scalar_one()
    items = db.execute(
        stmt.order_by(models.Reading.timestamp.desc()).limit(limit).offset(offset)
    ).scalars().all()
    return total, items

# ----- Commands -----
def create_command(db: Session, payload: schemas.CommandCreate) -> models.Command:
    device = get_or_create_device(db, device_id=payload.device_id)
    command = models.Command(
        device_id=device.id,
        command=payload.command,
        payload=payload.payload,
        status="pending"
    )
    db.add(command)
    db.commit()
    db.refresh(command)
    return command

def get_pending_commands(db: Session, device_id: str) -> list[models.Command]:
    device = db.execute(
        select(models.Device).where(models.Device.device_id == device_id)
    ).scalar_one_or_none()
    
    if not device:
        return []
    
    stmt = select(models.Command).where(
        models.Command.device_id == device.id,
        models.Command.status == "pending"
    ).order_by(models.Command.created_at.asc())
    
    return db.execute(stmt).scalars().all()

def update_command_status(
    db: Session, 
    command_id: int, 
    status: str
) -> models.Command | None:
    command = db.get(models.Command, command_id)
    if command:
        command.status = status
        command.executed_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(command)
    return command

# ----- Device Status Updates -----
def update_device_status(db: Session, device_id: str, status_data: schemas.ESPStatusPackage):
    """Update or create a device entry based on ESP status package."""
    device = get_or_create_device(db, device_id=device_id, name=status_data.custom_name)
    device.status = status_data.status
    device.actual_partition = status_data.actual_partition
    device.version_factory = status_data.version_factory
    device.version_ota_1 = status_data.version_ota_1
    device.version_ota_2 = status_data.version_ota_2
    device.mac = status_data.mac
    device.last_seen = datetime.now(timezone.utc)

    db.add(device)
    db.commit()
    db.refresh(device)
    return device


# ----- Sensor Reading (temperature, motor status, etc.) -----
def create_sensor_reading(db: Session, device_id: str, reading_data: dict):
    """Insert a single sensor reading into the database."""
    device = get_or_create_device(db, device_id=device_id)
    ts = reading_data.get("timestamp") or datetime.now(timezone.utc)

    reading = models.SensorReading(
        device_id=device.id,
        temperature=reading_data.get("temperature"),
        sd_free_storage=reading_data.get("sd_free_storage"),
        motor_status=reading_data.get("motor_status"),
        timestamp=ts,
    )
    db.add(reading)
    db.commit()
    db.refresh(reading)
    return reading


# ----- Bed Pressure Reading -----
def create_bed_pressure_reading(db: Session, device_id: str, adc_value: float, timestamp: datetime):
    """Insert bed pressure / occupancy sensor data."""
    device = get_or_create_device(db, device_id=device_id)
    reading = models.BedPressureReading(
        device_id=device.id,
        adc_value_scaled=adc_value,
        timestamp=timestamp,
    )
    db.add(reading)
    db.commit()
    db.refresh(reading)
    return reading


# ----- Bed Power Event -----
def create_bed_power_event(db: Session, device_id: str, button_on: str, timestamp: datetime):
    """Insert power button event (ON/OFF)."""
    device = get_or_create_device(db, device_id=device_id)
    event = models.BedPowerEvent(
        device_id=device.id,
        button_on=button_on,
        timestamp=timestamp,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event

def get_device_by_id(db: Session, device_id: str):
    return db.query(models.Device).filter(models.Device.device_id == device_id).first()

def create_device(db: Session, device: schemas.DeviceCreate):
    db_device = models.Device(
        device_id=device.device_id,
        name=device.name
    )
    db.add(db_device)
    db.commit()
    db.refresh(db_device)
    return db_device