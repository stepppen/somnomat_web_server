from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
import random
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import asyncio
import math

active_push_tasks: Dict[str, asyncio.Task] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for task in active_push_tasks.values():
        task.cancel()


load_dotenv()
app = FastAPI(title="MVP 1.2 API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# In-memory storage for aggregation
latest_data = {}  
pending_commands = {} 
sleep_data = {} 
aggregated_occupancy = [] 
occ_intervals = []

default_metrics = {
    "consistency_score": 0,
    "consistency_sd_minutes": 0,
    "bed_on_coverage_pct": 0,
    "avg_sleep_per_night": 0,
    "total_intervals": 0,
    "total_nights": 0,
    "avg_awakenings": 0
}


push_task = None

class ESPStartupData(BaseModel):
    CustomName: str
    Status: str
    ActualPartition: int
    VersionFactory: str
    VersionOTA1: str
    VersionOTA2: str
    MAC: str
    Temperature: float
    SDFreeStorage: float

class ESPSensorData(BaseModel):
    Occupancy: bool
    Speed: float
    Acceleration: float
    Vibration: float
    Intensity: float
    Safety: float
    MotorStatus: str

class PWACommand(BaseModel):
    device_id: str
    command: str
    payload: Optional[dict] = {}

class SleepMetricsCache:
    def __init__(self):
        self.last_full_compute = None
        self.cached_intervals = []
        
sleep_cache = SleepMetricsCache()


#aggregation function occupancy
def aggregate_occupancy(data: bool):
    aggregated_occupancy.append({
        "occupied": data,
        "timestamp": datetime.now().isoformat()
        })

def summarize_minute(batch):
    """Return true if more than half are occ"""
    if not batch:
        return None
    occupied_count = sum(1 for x in batch if x["occupied"])
    occupancy_value = occupied_count >= len(batch) / 2
    return occupancy_value

async def push_aggregated_data(device_id: str):
    """Runs every min"""
    while True:
        await asyncio.sleep(60)
        if not aggregated_occupancy:
            continue

        batch = aggregated_occupancy.copy()
        aggregated_occupancy.clear()

        # One bool for whole minute
        occupied = summarize_minute(batch)

        # Check if device exists before computing metrics
        try:
            result = supabase.table("devices").select("id").eq("id", int(device_id)).execute()
            if not result.data:
                print(f"Device {device_id} not registered. Skipping metric computation.")
                continue
        except Exception as e:
            print(f"Error checking device in push_aggregated_data: {e}")
            continue

        try:
            current_record = {
                "device_id": int(device_id),
                "created_at": datetime.now().isoformat(),
                "occupied": occupied
            }
            supabase.table("raw_occupancy").insert(current_record).execute()

        except Exception as e:
            print(f"Error inserting raw occupancy: {e}")
            return default_metrics


#calculating sleep metrics
def compute_sleep_metrics(device_id: str) -> Dict:
    """
    Compute sleep metrics based on current occupancy and historical data.
    Returns a dictionary with all metrics, even if errors occur.
    """

    now = datetime.now()
    seven_days_ago = now - timedelta(days=7)
    
    try:
        response = supabase.table("raw_occupancy")\
            .select("*")\
            .eq("device_id", int(device_id))\
            .gte("created_at", seven_days_ago.isoformat())\
            .order("created_at")\
            .execute()
        
        occupancy_rows = response.data
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return default_metrics
    
    occ_intervals = build_occupancy_intervals(occupancy_rows)
    
    if not occ_intervals:
        return default_metrics
    
    total_sleep_min = compute_total_sleep(occ_intervals)
    total_nights = count_nights(occ_intervals)
    avg_sleep = compute_avg_sleep_per_night(occ_intervals)
    avg_awakenings = compute_avg_awakenings(occ_intervals)
    
    bedtimes = []
    for iv in occ_intervals:
        bedtime_minutes = iv["start"].hour * 60 + iv["start"].minute
        bedtimes.append(bedtime_minutes)
    
    if bedtimes:
        import statistics
        consistency_sd = statistics.stdev(bedtimes) if len(bedtimes) > 1 else 0
        consistency_score = max(0, 100 - (consistency_sd / 60 * 100))
    else:
        consistency_sd = 0
        consistency_score = 0
    
    bed_use_pct = (total_sleep_min / (7 * 24 * 60)) * 100 if total_sleep_min > 0 else 0
    
    return {
        "consistency_score": round(consistency_score, 2),
        "consistency_sd_minutes": round(consistency_sd, 2),
        "bed_on_coverage_pct": round(bed_use_pct, 2),
        "avg_sleep_per_night_min": round(avg_sleep, 2),
        "avg_sleep_per_night_hours": round((avg_sleep / 60), 2),
        "total_intervals": len(occ_intervals),
        "total_nights": total_nights,
        "avg_awakenings": round(avg_awakenings, 2),
        "total_sleep_min": round(total_sleep_min, 2),
        "total_sleep_hours": round((total_sleep_min / 60), 2)
    }

def build_occupancy_intervals(rows: List[Dict]) -> List[Dict]:
    """
    bools occupancy -> continuous intervals.
    """
    if not rows:
        return []
    
    samples = sorted(rows, key=lambda x: x["created_at"]) 
    MIN_SEG_SEC = 120 
    
    intervals = []
    in_occ = False
    current_start = None
    last_timestamp = None
    
    for sample in samples:
        timestamp = datetime.fromisoformat(sample["created_at"].replace("Z", "")) 
        occupied = sample.get("occupied", False)

        if occupied and not in_occ:
            in_occ = True
            current_start = timestamp
        elif not occupied and in_occ:
            in_occ = False
            if last_timestamp and (last_timestamp - current_start).total_seconds() >= MIN_SEG_SEC:
                intervals.append({
                    "start": current_start,
                    "end": last_timestamp,
                    "duration_min": (last_timestamp - current_start).total_seconds() / 60
                })
            current_start = None
        
        last_timestamp = timestamp
    
    if in_occ and current_start and last_timestamp:
        if (last_timestamp - current_start).total_seconds() >= MIN_SEG_SEC:
            intervals.append({
                "start": current_start,
                "end": last_timestamp,
                "duration_min": (last_timestamp - current_start).total_seconds() / 60
            })
    
    return intervals


def compute_total_sleep(intervals: List[Dict]) -> float:
    """Total minutes of occupancy across all intervals"""
    return sum(iv["duration_min"] for iv in intervals)


def count_nights(intervals: List[Dict]) -> int:
    """Count unique days with occupancy data"""
    if not intervals:
        return 0
    
    days = set()
    for iv in intervals:
        day_key = iv["start"].date().isoformat()
        days.add(day_key)
    
    return len(days)


def compute_avg_sleep_per_night(intervals: List[Dict]) -> float:
    """Average sleep duration per night"""
    if not intervals:
        return 0.0
    
    by_day = {}
    for iv in intervals:
        day = iv["start"].date().isoformat()
        if day not in by_day:
            by_day[day] = 0
        by_day[day] += iv["duration_min"]
    
    if not by_day:
        return 0.0
    
    return sum(by_day.values()) / len(by_day)


def compute_avg_awakenings(intervals: List[Dict]) -> float:
    """
    Average number of awakenings per night.
    """
    if not intervals:
        return 0.0
    
    by_day = {}
    for iv in intervals:
        day = iv["start"].date().isoformat()
        if day not in by_day:
            by_day[day] = []
        by_day[day].append(iv)
    
    if not by_day:
        return 0.0
    
    total_awakenings = 0
    for day, day_intervals in by_day.items():
        awakenings = max(0, len(day_intervals) - 1)
        total_awakenings += awakenings
    
    return total_awakenings / len(by_day)


def compute_awakenings_last_night(intervals: List[Dict]) -> int:
    """Number of awakenings on the most recent night"""
    if not intervals:
        return 0
    
    latest_day = max(iv["start"].date() for iv in intervals)
    day_intervals = [iv for iv in intervals if iv["start"].date() == latest_day]
    
    return max(0, len(day_intervals) - 1)


def compute_consistency_score(intervals: List[Dict]) -> int:
    """
    Bedtime consistency score (0-100).
    Based on circular standard deviation of first sleep interval each night.
    """
    sd_min = compute_consistency_sd(intervals)
    if sd_min is None:
        return 0
    
    clamp_180 = min(180, max(0, sd_min))
    score = round(100 * (1 - clamp_180 / 180))
    
    return score


def compute_consistency_sd(intervals: List[Dict]) -> Optional[float]:
    """
    Circular standard deviation of bedtime in minutes.
    """
    if not intervals:
        return None
    
    by_day = {}
    for iv in intervals:
        day = iv["start"].date().isoformat()
        if day not in by_day:
            by_day[day] = []
        by_day[day].append(iv)
    
    first_starts = []
    for day in sorted(by_day.keys()):
        day_intervals = sorted(by_day[day], key=lambda x: x["start"])
        first = next((iv for iv in day_intervals if iv["duration_min"] >= 2), None)
        if first:
            first_starts.append(first["start"])
    
    if len(first_starts) < 2:
        return 0.0
    
    mins = [
        dt.hour * 60 + dt.minute + dt.second / 60
        for dt in first_starts
    ]
    
    rad = [m / (24 * 60) * 2 * math.pi for m in mins]
    
    C = sum(math.cos(t) for t in rad) / len(rad)
    S = sum(math.sin(t) for t in rad) / len(rad)
    R = math.sqrt(C * C + S * S)
    
    if R == 0:
        sd_min = 12 * 60  # max spread
    else:
        s = math.sqrt(-2 * math.log(R))
        sd_min = s / (2 * math.pi) * 24 * 60
    
    return round(sd_min)


# ----- ESP32 Endpoints -----
@app.post("/esp32/{device_id}/startup")
def esp32_startup(device_id: str, data: ESPStartupData):
    """ESP32 -> sends info on boot"""
    if device_id not in latest_data:
        latest_data[device_id] = {}
        pending_commands[device_id] = []
        
    latest_data[device_id].update({
        "custom_name": data.CustomName,
        "status": data.Status,
        "partition": data.ActualPartition,
        "version": data.VersionFactory,
        "version_ota1": data.VersionOTA1,
        "version_ota2": data.VersionOTA2,
        "sd_free_storage": data.SDFreeStorage,
        "mac": data.MAC,
        "last_seen": datetime.now().isoformat()
    })
    
    try:
        device_record = {
            "id": int(device_id),  
            "name": data.CustomName,
            "status": data.Status,
            "temperature": data.Temperature,
            "mac": data.MAC,
        }
        firmware_record = {
            "device_id": int(device_id),  
            "version": data.VersionFactory,
            "partition": data.ActualPartition,
            "version_ota1": data.VersionOTA1,
            "version_ota2": data.VersionOTA2,
            "sd_free_storage": data.SDFreeStorage,
            "last_seen": datetime.now().isoformat()
        }
        
        result_devices = supabase.table("devices")\
            .upsert(device_record)\
            .execute()
        
        result_firmware = supabase.table("firmware")\
            .upsert(firmware_record)\
            .execute()
        
        print(f"upserted device {device_id}: {result_devices.data}")
        return {"success": True, "message": f"Device {device_id} registered", "device_id": device_id}
        
    except Exception as e:
        print(f"err upserting device {device_id}: {e}")
        return {"success": False, "message": f"Failed to register device: {str(e)}", "device_id": device_id}

    
@app.post("/esp32/{device_id}/sensors")
async def esp32_sensors(device_id: str, data: ESPSensorData):
    """ESP32 -> sends sensor readings"""
    global active_push_tasks
    
    try:
        result = supabase.table("devices").select("id").eq("id", int(device_id)).execute()
        print(f"Result from eq: {result}")
        if not result.data:
            # Device doesn't exist - log but don't fail
            print(f"⚠ Device {device_id} not found in database. Sensor data stored in memory only.")
            # Store in memory but skip database operations
            if device_id not in latest_data:
                latest_data[device_id] = {}
            
            latest_data[device_id].update({
                "occupancy": data.Occupancy,
                "speed": data.Speed,
                "acceleration": data.Acceleration,
                "vibration": data.Vibration,
                "intensity": data.Intensity,
                "safety": data.Safety,
                "motor_status": data.MotorStatus,
                "last_seen": datetime.now().isoformat()
            })
            
            #aggregate occ for when device gets registered
            aggregate_occupancy(data.Occupancy)
            
            return {
                "success": True, 
                "warning": "Device not registered. Call /startup endpoint first."
            }
    except Exception as e:
        print(f"Error checking device existence: {e}")
    
    # Update in-memory data
    if device_id not in latest_data:
        latest_data[device_id] = {}
    
    latest_data[device_id].update({
        "motor_status": data.MotorStatus,
        "occupancy": data.Occupancy,
        "speed": data.Speed,
        "acceleration": data.Acceleration,
        "vibration": data.Vibration,
        "intensity": data.Intensity,
        "safety": data.Safety,
        "last_seen": datetime.now().isoformat()
    })

    # Aggregate occupancy data
    aggregate_occupancy(data.Occupancy)
    
    # Start background task if not already running
    if device_id not in active_push_tasks or active_push_tasks[device_id].done():
        active_push_tasks[device_id] = asyncio.create_task(push_aggregated_data(device_id))
    
    return {"success": True}

@app.get("/esp32/{device_id}/poll")
def esp32_poll(device_id: str):
    """ESP32 -> poll for pending commands"""
    if device_id not in pending_commands:
        pending_commands[device_id] = []
    
    # Get current commands
    commands = pending_commands[device_id].copy()
    pending_commands[device_id] = []
    
    return {
        "device_id": device_id,
        "commands": commands,
        "timestamp": datetime.now().isoformat()
    }

# ----- PWA endpoints -----
@app.get("/devices")
def list_devices():
    """Get all devices with data"""
    try:
        response = supabase.table("devices")\
            .select("*")\
            .execute()
        
    except Exception as e:
        print(f"Error fetching devices: {e}")
        # return default_metrics
    
    # devices = []
    # for device_id, data in latest_data.items():
    #     devices.append({
    #         "device_id": device_id,
    #         **data
    #     })
    return response

#where the real-time values are fetched from specific ESP e.g. occupied
@app.get("/devices/{device_id}")
def get_device(device_id: str):
    """Get specific device data"""

    # if device_id not in latest_data:
    #     raise HTTPException(status_code=404, detail="Device not found")
    
    response = supabase.table("devices").select("*").eq("id", int(device_id)).execute()
    
    # echo
    return response

@app.post("/commands")
def create_command(cmd: PWACommand):
    """create command for ESP32"""
    if cmd.device_id not in pending_commands:
        pending_commands[cmd.device_id] = []
    
    command_obj = {
        "command": cmd.command,
        "payload": cmd.payload,
        "created_at": datetime.now().isoformat()
    }
    
    pending_commands[cmd.device_id].append(command_obj)
    
    # echo
    return {
        "success": True,
        "echo": command_obj,
        "message": f"Command '{cmd.command}' queued for {cmd.device_id}"
    }



#where the aggregated values are fetched from func 
@app.get("/sleep/{device_id}/summary")
def get_sleep_summary(device_id: str):
    """Get full sleep summary for dashboard"""

    metrics: Dict = compute_sleep_metrics(device_id)
    devices = supabase.table("devices").select("*").eq("id", int(device_id)).execute()

    # debuggin
    print(f"[DEBUG] Summary requested for device: {device_id}")
    print(f"[DEBUG] Available device: {devices.data}")
    # print(f"[DEBUG] Sleep data devices: {list(sleep_data.keys())}")

    return {
        "device_id": device_id,
        "summary": {
            "created_at": datetime.now().isoformat(),
            "sleep_consistency": metrics.get("consistency_score", 0),
            "bedtime_consistency": metrics.get("consistency_sd_minutes", 0),
            "bed_use": metrics.get("bed_on_coverage_pct", 0),

            # average values
            "daily_occupancy": metrics.get("avg_sleep_per_night", 0), 
            "avg_sleep_per_night_min": metrics.get("avg_sleep_per_night_min", 0),
            "avg_sleep_per_night_hours": metrics.get("avg_sleep_per_night_hours", 0),
            "avg_awakenings_per_night": metrics.get("avg_awakenings", 0),

            # total values
            "total_intervals": metrics.get("total_intervals", 0),
            "total_nights": metrics.get("total_nights", 0),
            "total_sleep_min": metrics.get("total_sleep_min", 0),
            "total_sleep_hours": metrics.get("total_sleep_hours", 0),

            "intervals": occ_intervals
        }
    }

# ----- Debug Endpoint -----
@app.get("/debug")
def debug_state():
    """See all stored data (for dev only)"""
    return {
        "status": "API is running",
        "timestamp": datetime.now().isoformat(),
        "latest_data": latest_data,
        "pending_commands": pending_commands,
        "sleep_data_keys": list(sleep_data.keys()),
        "total_devices": len(latest_data),
        "total_sleep_devices": len(sleep_data)
    }

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "status": "API is running",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "test": "/test",
            "devices": "/devices",
            "esp32_startup": "/esp32/{device_id}/startup",
            "esp32_sensors": "/esp32/{device_id}/sensors",
            "esp32_poll": "/esp32/{device_id}/poll"
        }
    }


#main guard
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)