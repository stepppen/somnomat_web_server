from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, JSON, func
from datetime import datetime, timezone
from app.database import Base

class Device(Base):
    __tablename__ = "devices"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    device_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    
    readings: Mapped[list["Reading"]] = relationship(
        back_populates="device", cascade="all, delete-orphan"
    )
    commands: Mapped[list["Command"]] = relationship(
        back_populates="device", cascade="all, delete-orphan"
    )
    sensor_readings: Mapped[list["SensorReading"]] = relationship(
        back_populates="device", cascade="all, delete-orphan"
    )
    bed_pressure_readings: Mapped[list["BedPressureReading"]] = relationship(
        back_populates="device", cascade="all, delete-orphan"
    )
    bed_power_events: Mapped[list["BedPowerEvent"]] = relationship(
        back_populates="device", cascade="all, delete-orphan"
    )

class Reading(Base):
    __tablename__ = "readings"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    device_id: Mapped[int] = mapped_column(
        ForeignKey("devices.id", ondelete="CASCADE"), index=True
    )
    timestamp: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    sensor: Mapped[str] = mapped_column(String(64), index=True)
    value: Mapped[float] = mapped_column(Float)
    
    device: Mapped["Device"] = relationship(back_populates="readings")

class Command(Base):
    __tablename__ = "commands"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    device_id: Mapped[int] = mapped_column(
        ForeignKey("devices.id", ondelete="CASCADE"), index=True
    )
    command: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    executed_at: Mapped[DateTime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    
    device: Mapped["Device"] = relationship(back_populates="commands")


class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    device_id: Mapped[int] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"), index=True)
    data: Mapped[dict] = mapped_column(JSON)  # store temperature, sd_free_storage, etc.
    timestamp: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    device: Mapped["Device"] = relationship(back_populates="sensor_readings")


# --- Bed Pressure Sensor Data ---
class BedPressureReading(Base):
    __tablename__ = "bed_pressure_readings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    device_id: Mapped[int] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"), index=True)
    adc_value_scaled: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    device: Mapped["Device"] = relationship(back_populates="bed_pressure_readings")


# --- Bed Power Button Events ---
class BedPowerEvent(Base):
    __tablename__ = "bed_power_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    device_id: Mapped[int] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"), index=True)
    button_on: Mapped[str] = mapped_column(String(16), nullable=False)  # OFF->ON or ON->OFF
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    device: Mapped["Device"] = relationship(back_populates="bed_power_events")


