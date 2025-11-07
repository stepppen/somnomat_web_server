"""
Seed script to load CSV data into the database.
Run with: python seed_data.py
"""
import csv
from datetime import datetime
from app.database import SessionLocal
from app import crud, schemas

def parse_timestamp(ts_str):
    """Parse ISO timestamp from CSV"""
    try:
        return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except:
        return datetime.now()

def load_bed_power_csv(filepath, device_id="esp32-bed-001"):
    """Load bed power events from CSV (OFF->ON, ON->OFF)"""
    db = SessionLocal()
    count = 0
    
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp = parse_timestamp(row['Timestamp'])
                button_state = row['button_on']
                
                crud.create_bed_power_event(
                    db,
                    device_id=device_id,
                    button_on=button_state,  # was event_type
                    timestamp=timestamp
                )
                count += 1
        
        print(f"✓ Loaded {count} bed power events from {filepath}")
    except Exception as e:
        print(f"✗ Error loading {filepath}: {e}")
    finally:
        db.close()

def load_occupancy_csv(filepath, device_id="esp32-bed-001"):
    """Load bed occupancy/pressure data from CSV"""
    db = SessionLocal()
    count = 0
    
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp = parse_timestamp(row.get('Timestamp') or row.get('timestamp'))
                
                # Get ADC value (try different column names)
                adc_value = None
                for col in ['adc_value_scaled', 'ADC_value_scaled', 'value', 'scaled_value']:
                    if col in row:
                        try:
                            adc_value = float(row[col])
                            break
                        except:
                            pass
                
                if adc_value is not None:
                    crud.create_bed_pressure_reading(
                        db,
                        device_id=device_id,
                        adc_value=adc_value,  # was pressure_value
                        timestamp=timestamp
                    )
                    count += 1
        
        print(f"✓ Loaded {count} occupancy readings from {filepath}")
    except Exception as e:
        print(f"✗ Error loading {filepath}: {e}")
    finally:
        db.close()

def create_sample_device():
    """Create a sample device with status"""
    db = SessionLocal()
    
    try:
        # inside create_sample_device()
        device = crud.update_device_status(
            db,
            device_id="esp32-bed-001",
            status_data=schemas.ESPStatusPackage(
                name="Prototype Bed #1",  # was custom_name
                custom_name="Prototype Bed #1",
                status="online",
                actual_partition="OTA1",
                version_factory="1.0.0",
                version_ota_1="1.1.0",
                version_ota_2="1.0.5",
                mac="AABBCCDDEEFF"  # shortened, no colons
            )
        )
        print(f"✓ Created device: {device.name} ({device.device_id})")
        
        # Add some sample sensor readings
        for i in range(10):
            crud.create_sensor_reading(
                db,
                device_id="esp32-bed-001",
                reading_data={
                    "temperature": 22.0 + (i * 0.5),
                    "sd_free_storage": 28.5 - (i * 0.1),
                    "motor_status": "stopped" if i < 5 else "running"
                }
            )
        print(f"✓ Created 10 sample sensor readings")
        
    except Exception as e:
        print(f"✗ Error creating sample device: {e}")
    finally:
        db.close()

def main():
    print("=== Seeding Database ===\n")
    
    # Create sample device
    create_sample_device()
    print()
    
    # Load CSV files
    # Update these paths to match your CSV file locations
    load_bed_power_csv("data/bed_power_day1.csv")
    load_occupancy_csv("data/sleep_occupancy_day1.csv") 
    
    print("\n=== Seeding Complete ===")

if __name__ == "__main__":
    main()