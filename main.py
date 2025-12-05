from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
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
aggregated_occupancy_binary = [] 
occ_intervals = []
device_metadata = {}

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
    Speed: float
    Acceleration: float
    Vibration: int
    Intensity: int
    MotorStatus: int
    Safety: int
    Occupancy: bool
    OccupancyRaw: float
    OccupancyBinary: int

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
def aggregate_occupancy_binary(data: int):
    aggregated_occupancy_binary.append({
        "occupancy_binary": data,
        "timestamp": datetime.now().isoformat()
        })

def summarize_minute(batch):
    """Return true if more than half are occ"""
    if not batch:
        return None
    occupied_count = sum(1 for x in batch if x["occupancy_binary"] == 1 )
    occupancy_value = occupied_count >= len(batch) / 2
    return occupancy_value

async def push_aggregated_data(device_id: str):
    """Runs every min"""
    while True:
        await asyncio.sleep(60)
        if not aggregated_occupancy_binary:
            continue

        batch = aggregated_occupancy_binary.copy()
        aggregated_occupancy_binary.clear()

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
                "occupied": occupied,
            }
            supabase.table("raw_occupancy").insert(current_record).execute()

        except Exception as e:
            print(f"Error inserting raw occupancy: {e}")
            return default_metrics

def build_occupancy_intervals(rows: List[Dict]) -> List[Dict]:
    """Convert boolean occupancy -> continuous intervals"""
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

def filter_intervals_by_date_range(intervals: List[Dict], start_date: datetime, end_date: datetime) -> List[Dict]:
    """Filter intervals that fall within the date range"""
    filtered = []
    for iv in intervals:
        interval_start = iv["start"] if isinstance(iv["start"], datetime) else datetime.fromisoformat(str(iv["start"]).replace("Z", ""))
        
        # Check if interval overlaps with our date range
        if interval_start.date() >= start_date.date() and interval_start.date() <= end_date.date():
            filtered.append(iv)
    
    return filtered

def compute_sleep_metrics_for_range(intervals: List[Dict], start_date: datetime, end_date: datetime) -> Dict:
    """
    Compute comprehensive sleep metrics for a specific date range
    """
    if not intervals:
        return {
            "sleep_duration_min": 0,
            "sleep_duration_hours": 0,
            "bed_activity_min": 0,
            "bed_activity_hours": 0,
            "consistency_score": 0,
            "consistency_sd_minutes": 0,
            "total_intervals": 0,
            "avg_awakenings": 0,
            "wake_up_time": None,
            "bed_time": None,
            "time_in_bed_min": 0,
            "time_in_bed_hours": 0,
            "nights_count": 0
        }
    
    # Total sleep duration
    total_sleep_min = sum(iv["duration_min"] for iv in intervals)
    
    # Count nights
    nights = set()
    for iv in intervals:
        interval_start = iv["start"] if isinstance(iv["start"], datetime) else datetime.fromisoformat(str(iv["start"]).replace("Z", ""))
        nights.add(interval_start.date())
    nights_count = len(nights)
    
    # Bed activity (time motor was ON)
    bed_activity_min = total_sleep_min  # Same as sleep duration for now
    
    # Calculate awakenings (gaps between intervals on same night)
    awakenings_by_night = {}
    for iv in intervals:
        interval_start = iv["start"] if isinstance(iv["start"], datetime) else datetime.fromisoformat(str(iv["start"]).replace("Z", ""))
        night_key = interval_start.date()
        if night_key not in awakenings_by_night:
            awakenings_by_night[night_key] = []
        awakenings_by_night[night_key].append(iv)
    
    total_awakenings = 0
    for night, night_intervals in awakenings_by_night.items():
        awakenings = max(0, len(night_intervals) - 1)
        total_awakenings += awakenings
    
    avg_awakenings = total_awakenings / nights_count if nights_count > 0 else 0
    
    # Bedtime consistency (using first interval of each night)
    bedtimes_minutes = []
    for night, night_intervals in awakenings_by_night.items():
        sorted_intervals = sorted(night_intervals, key=lambda x: x["start"] if isinstance(x["start"], datetime) else datetime.fromisoformat(str(x["start"]).replace("Z", "")))
        if sorted_intervals:
            first_start = sorted_intervals[0]["start"] if isinstance(sorted_intervals[0]["start"], datetime) else datetime.fromisoformat(str(sorted_intervals[0]["start"]).replace("Z", ""))
            bedtime_min = first_start.hour * 60 + first_start.minute
            bedtimes_minutes.append(bedtime_min)
    
    # Circular standard deviation for bedtime consistency
    if len(bedtimes_minutes) >= 2:
        rad = [m / (24 * 60) * 2 * math.pi for m in bedtimes_minutes]
        C = sum(math.cos(t) for t in rad) / len(rad)
        S = sum(math.sin(t) for t in rad) / len(rad)
        R = math.sqrt(C * C + S * S)
        
        if R == 0:
            consistency_sd = 12 * 60
        else:
            s = math.sqrt(-2 * math.log(R))
            consistency_sd = s / (2 * math.pi) * 24 * 60
        
        consistency_score = max(0, 100 - (consistency_sd / 180 * 100))
    else:
        consistency_sd = 0
        consistency_score = 100 if len(bedtimes_minutes) == 1 else 0
    
    # Average bedtime and wake time
    all_starts = [iv["start"] if isinstance(iv["start"], datetime) else datetime.fromisoformat(str(iv["start"]).replace("Z", "")) for iv in intervals]
    all_ends = [iv["end"] if isinstance(iv["end"], datetime) else datetime.fromisoformat(str(iv["end"]).replace("Z", "")) for iv in intervals]
    
    # Get average bedtime (first interval start)
    first_intervals_by_night = []
    for night, night_intervals in awakenings_by_night.items():
        sorted_intervals = sorted(night_intervals, key=lambda x: x["start"] if isinstance(x["start"], datetime) else datetime.fromisoformat(str(x["start"]).replace("Z", "")))
        if sorted_intervals:
            first_intervals_by_night.append(sorted_intervals[0]["start"] if isinstance(sorted_intervals[0]["start"], datetime) else datetime.fromisoformat(str(sorted_intervals[0]["start"]).replace("Z", "")))
    
    avg_bedtime = None
    if first_intervals_by_night:
        avg_bedtime_minutes = sum(dt.hour * 60 + dt.minute for dt in first_intervals_by_night) / len(first_intervals_by_night)
        avg_bedtime_hour = int(avg_bedtime_minutes // 60)
        avg_bedtime_min = int(avg_bedtime_minutes % 60)
        avg_bedtime = f"{avg_bedtime_hour:02d}:{avg_bedtime_min:02d}"
    
    # Get average wake time (last interval end)
    last_intervals_by_night = []
    for night, night_intervals in awakenings_by_night.items():
        sorted_intervals = sorted(night_intervals, key=lambda x: x["end"] if isinstance(x["end"], datetime) else datetime.fromisoformat(str(x["end"]).replace("Z", "")))
        if sorted_intervals:
            last_intervals_by_night.append(sorted_intervals[-1]["end"] if isinstance(sorted_intervals[-1]["end"], datetime) else datetime.fromisoformat(str(sorted_intervals[-1]["end"]).replace("Z", "")))
    
    avg_wake_time = None
    if last_intervals_by_night:
        avg_wake_minutes = sum(dt.hour * 60 + dt.minute for dt in last_intervals_by_night) / len(last_intervals_by_night)
        avg_wake_hour = int(avg_wake_minutes // 60)
        avg_wake_min = int(avg_wake_minutes % 60)
        avg_wake_time = f"{avg_wake_hour:02d}:{avg_wake_min:02d}"
    
    # Time in bed (from first start to last end each night)
    time_in_bed_total = 0
    for night, night_intervals in awakenings_by_night.items():
        sorted_intervals = sorted(night_intervals, key=lambda x: x["start"] if isinstance(x["start"], datetime) else datetime.fromisoformat(str(x["start"]).replace("Z", "")))
        if sorted_intervals:
            first_start = sorted_intervals[0]["start"] if isinstance(sorted_intervals[0]["start"], datetime) else datetime.fromisoformat(str(sorted_intervals[0]["start"]).replace("Z", ""))
            last_end = sorted_intervals[-1]["end"] if isinstance(sorted_intervals[-1]["end"], datetime) else datetime.fromisoformat(str(sorted_intervals[-1]["end"]).replace("Z", ""))
            time_in_bed_total += (last_end - first_start).total_seconds() / 60
    
    return {
        "sleep_duration_min": round(total_sleep_min, 2),
        "sleep_duration_hours": round(total_sleep_min / 60, 2),
        "bed_activity_min": round(bed_activity_min, 2),
        "bed_activity_hours": round(bed_activity_min / 60, 2),
        "consistency_score": round(consistency_score, 2),
        "consistency_sd_minutes": round(consistency_sd, 2),
        "total_intervals": len(intervals),
        "avg_awakenings": round(avg_awakenings, 2),
        "wake_up_time": avg_wake_time,
        "bed_time": avg_bedtime,
        "time_in_bed_min": round(time_in_bed_total, 2),
        "time_in_bed_hours": round(time_in_bed_total / 60, 2),
        "nights_count": nights_count
    }



#calculating sleep metrics
# def compute_sleep_metrics(device_id: str) -> Dict:
#     """
#     Compute sleep metrics based on current occupancy and historical data.
#     Returns a dictionary with all metrics, even if errors occur.
#     """

#     now = datetime.now()
#     seven_days_ago = now - timedelta(days=7)
    
#     try:
#         response = supabase.table("raw_occupancy")\
#             .select("*")\
#             .eq("device_id", int(device_id))\
#             .gte("created_at", seven_days_ago.isoformat())\
#             .order("created_at")\
#             .execute()
        
#         occupancy_rows = response.data
#     except Exception as e:
#         print(f"Error fetching historical data: {e}")
#         return default_metrics
    
#     occ_intervals = build_occupancy_intervals(occupancy_rows)
    
#     if not occ_intervals:
#         return default_metrics
    
#     total_sleep_min = compute_total_sleep(occ_intervals)
#     total_nights = count_nights(occ_intervals)
#     avg_sleep = compute_avg_sleep_per_night(occ_intervals)
#     avg_awakenings = compute_avg_awakenings(occ_intervals)
    
#     bedtimes = []
#     for iv in occ_intervals:
#         bedtime_minutes = iv["start"].hour * 60 + iv["start"].minute
#         bedtimes.append(bedtime_minutes)
    
#     if bedtimes:
#         import statistics
#         consistency_sd = statistics.stdev(bedtimes) if len(bedtimes) > 1 else 0
#         consistency_score = max(0, 100 - (consistency_sd / 60 * 100))
#     else:
#         consistency_sd = 0
#         consistency_score = 0
    
#     bed_use_pct = (total_sleep_min / (7 * 24 * 60)) * 100 if total_sleep_min > 0 else 0
    
#     return {
#         "consistency_score": round(consistency_score, 2),
#         "consistency_sd_minutes": round(consistency_sd, 2),
#         "bed_on_coverage_pct": round(bed_use_pct, 2),
#         "avg_sleep_per_night_min": round(avg_sleep, 2),
#         "avg_sleep_per_night_hours": round((avg_sleep / 60), 2),
#         "total_intervals": len(occ_intervals),
#         "total_nights": total_nights,
#         "avg_awakenings": round(avg_awakenings, 2),
#         "total_sleep_min": round(total_sleep_min, 2),
#         "total_sleep_hours": round((total_sleep_min / 60), 2)
#     }

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
        device_metadata[device_id] = {
            "custom_name": data.CustomName,
        }
        
        result_devices = supabase.table("devices")\
            .upsert(device_record, on_conflict='id')\
            .execute()
        
        result_firmware = supabase.table("firmware")\
            .upsert(firmware_record, on_conflict='device_id')\
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
            aggregate_occupancy_binary(data.OccupancyBinary)
            
            return {
                "success": True, 
                "warning": "Device not registered. Call /startup endpoint first."
            }
        else:
            user_settings = {
                "device_id": int(device_id), 
                "vibration": data.Vibration,
                "intensity": data.Intensity,
                "motor_status": data.MotorStatus,
            }
            add_safety = {
                "id": int(device_id),
                "name": device_metadata[device_id]["custom_name"],
                "safety": data.Safety,
            }
            supabase.table("user_settings")\
                .upsert(user_settings, on_conflict='device_id')\
                .execute()
            supabase.table("devices")\
                .upsert(add_safety, on_conflict='id')\
                .execute()
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
    aggregate_occupancy_binary(data.OccupancyBinary)
    
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
        raise HTTPException(status_code=500, detail=str(e))
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

#User retrieves user settings
@app.get("/devices/{device_id}/settings")
def get_device(device_id: str):
    """Get device settings"""

    # if device_id not in latest_data:
    #     raise HTTPException(status_code=404, detail="Device not found")
    
    response = supabase.table("user_settings").select("*").eq("device_id", int(device_id)).execute()
    
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
def get_sleep_summary(
    device_id: str,
    period: Literal["day", "week", "month"] = Query(default="week"),
    date: Optional[str] = Query(default=None, description="ISO date string (YYYY-MM-DD)")
):
    """
    Get sleep summary for a specific period (day/week/month)
    
    Parameters:
    - device_id: Device identifier
    - period: 'day', 'week', or 'month'
    - date: Optional ISO date string. If not provided, uses today.
            For 'day': returns data for that specific day
            For 'week': returns data for the week containing that date
            For 'month': returns data for the month containing that date
    """
    
    # Parse target date
    if date:
        try:
            target_date = datetime.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        target_date = datetime.now()
    
    # Calculate date range based on period
    if period == "day":
        start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1) - timedelta(microseconds=1)
    elif period == "week":
        # Week starts on Monday
        days_since_monday = target_date.weekday()
        start_date = (target_date - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=7) - timedelta(microseconds=1)
    elif period == "month":
        start_date = target_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # Get last day of month
        if target_date.month == 12:
            end_date = target_date.replace(year=target_date.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0) - timedelta(microseconds=1)
        else:
            end_date = target_date.replace(month=target_date.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0) - timedelta(microseconds=1)
    else:
        raise HTTPException(status_code=400, detail="Invalid period. Must be 'day', 'week', or 'month'")
    
    # Fetch occupancy data from database
    try:
        response = supabase.table("raw_occupancy")\
            .select("*")\
            .eq("device_id", int(device_id))\
            .gte("created_at", start_date.isoformat())\
            .lte("created_at", end_date.isoformat())\
            .order("created_at")\
            .execute()
        
        occupancy_rows = response.data
    except Exception as e:
        print(f"Error fetching occupancy data: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    # Build intervals from occupancy data
    all_intervals = build_occupancy_intervals(occupancy_rows)
    
    # Filter intervals for the specific date range
    filtered_intervals = filter_intervals_by_date_range(all_intervals, start_date, end_date)
    
    # Compute metrics for this range
    metrics = compute_sleep_metrics_for_range(filtered_intervals, start_date, end_date)
    
    # Prepare intervals for response (convert datetime to ISO strings)
    intervals_response = []
    for iv in filtered_intervals:
        intervals_response.append({
            "start": iv["start"].isoformat() if isinstance(iv["start"], datetime) else iv["start"],
            "end": iv["end"].isoformat() if isinstance(iv["end"], datetime) else iv["end"],
            "duration_min": iv["duration_min"]
        })
    
    return {
        "device_id": device_id,
        "period": period,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "summary": {
            **metrics,
            "intervals": intervals_response
        }
    }
# @app.get("/sleep/{device_id}/summary")
# def get_sleep_summary(device_id: str):
#     """Get full sleep summary for dashboard"""

#     metrics: Dict = compute_sleep_metrics(device_id)
#     devices = supabase.table("devices").select("*").eq("id", int(device_id)).execute()

#     # debuggin
#     print(f"[DEBUG] Summary requested for device: {device_id}")
#     print(f"[DEBUG] Available device: {devices.data}")
#     # print(f"[DEBUG] Sleep data devices: {list(sleep_data.keys())}")

#     return {
#         "device_id": device_id,
#         "summary": {
#             "created_at": datetime.now().isoformat(),
#             "sleep_consistency": metrics.get("consistency_score", 0),
#             "bedtime_consistency": metrics.get("consistency_sd_minutes", 0),
#             "bed_use": metrics.get("bed_on_coverage_pct", 0),

#             # average values
#             "daily_occupancy": metrics.get("avg_sleep_per_night", 0), 
#             "avg_sleep_per_night_min": metrics.get("avg_sleep_per_night_min", 0),
#             "avg_sleep_per_night_hours": metrics.get("avg_sleep_per_night_hours", 0),
#             "avg_awakenings_per_night": metrics.get("avg_awakenings", 0),

#             # total values
#             "total_intervals": metrics.get("total_intervals", 0),
#             "total_nights": metrics.get("total_nights", 0),
#             "total_sleep_min": metrics.get("total_sleep_min", 0),
#             "total_sleep_hours": metrics.get("total_sleep_hours", 0),

#             "intervals": occ_intervals
#         }
#     }

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