from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

# ----- Devices -----
class DeviceCreate(BaseModel):
    device_id: str = Field(min_length=1, max_length=64)
    custom_name: str | None = Field(default=None, max_length=128)

class DeviceOut(BaseModel):
    id: int
    device_id: str
    custom_name: str | None
    status: str  # "online", "offline"
    version_factory: str | None
    version_ota_1: str | None
    version_ota_2: str | None
    actual_partition: str | None
    mac: str | None
    last_seen: datetime | None

    class Config:
        from_attributes = True

# ----- ESP32 Status Package (ID 1001-1007) -----
# schemas.py
class ESPStatusPackage(BaseModel):
    name: str
    mac: str
    custom_name: str | None = None  # optional now
    status: str
    actual_partition: str
    version_factory: str
    version_ota_1: str
    version_ota_2: str

# ----- Sensor Readings (ID 1008-1011) -----
class SensorReading(BaseModel):
    """Single sensor reading"""
    device_id: str = Field(min_length=1, max_length=64)
    temperature: float | None = None
    sd_free_storage: float | None = None  # GB
    motor_status: str | None = None  # "running", "stopped", "error"
    timestamp: datetime | None = None

class SensorReadingOut(BaseModel):
    id: int
    device_id: int
    temperature: float | None
    sd_free_storage: float | None
    motor_status: str | None
    timestamp: datetime

    class Config:
        from_attributes = True

class BatchSensorReadings(BaseModel):
    """Batch upload of multiple sensor readings"""
    device_id: str
    readings: list[dict]  # [{temperature: 23.5, sd_free_storage: 4.2, motor_status: "running", timestamp: "..."}]

# ----- Bed Pressure Data (simulated from CSV) -----
class BedPressureReading(BaseModel):
    """Bed occupancy/pressure sensor data"""
    device_id: str
    adc_value_scaled: float  # 0.0 to 1.0, where < 0.2 means occupied
    timestamp: datetime | None = None

class BedPowerEvent(BaseModel):
    """Bed power button events"""
    device_id: str
    button_on: str  # "OFF->ON" or "ON->OFF"
    timestamp: datetime | None = None

# ----- Commands (PWA -> ESP32) -----
class CommandCreate(BaseModel):
    device_id: str = Field(min_length=1, max_length=64)
    command: str = Field(min_length=1, max_length=64)
    payload: dict = Field(default_factory=dict)

class CommandOut(BaseModel):
    id: int
    device_id: int
    command: str
    payload: dict
    status: str  # "pending", "completed", "failed"
    created_at: datetime
    executed_at: datetime | None = None

    class Config:
        from_attributes = True

class CommandStatusUpdate(BaseModel):
    status: str = Field(pattern="^(completed|failed)$")

# ----- Polling Response (ESP32 polls for commands) -----
class PollResponse(BaseModel):
    """Response when ESP32 polls for pending commands"""
    device_id: str
    commands: list[CommandOut]
    timestamp: datetime

# ----- Common -----
class Paginated(BaseModel):
    total: int
    items: list

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    service: str
    version: str

# ----- Generic Readings (for database) -----
class ReadingCreate(BaseModel):
    """Single reading from a sensor type"""
    device_id: str = Field(min_length=1, max_length=64)
    sensor: str = Field(min_length=1, max_length=64)
    value: float
    timestamp: datetime | None = None

class ReadingOut(BaseModel):
    id: int
    device_id: int
    sensor: str
    value: float
    timestamp: datetime

    class Config:
        from_attributes = True

class DeviceCreate(BaseModel):
    device_id: str
    name: str | None = None

class DeviceOut(BaseModel):
    device_id: str
    name: str | None = None

    class Config:
        orm_mode = True