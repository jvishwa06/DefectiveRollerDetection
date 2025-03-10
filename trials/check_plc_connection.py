import snap7
import time
from snap7.util import get_bool, set_bool

# Initialize and connect to the PLC
plc = snap7.client.Client()
plc.connect("172.17.8.17", 0, 1)


def read_proximity_status(byte_index, bool_index):
    """Read proximity sensor status."""
    data = plc.read_area(snap7.type.Areas.DB, 86, 0, 1)
    return get_bool(data, byte_index=byte_index, bool_index=bool_index)

def trigger_slot_opening(defect_detected):
    """Signal the PLC to open the slot."""
    data = bytearray(2)
    if defect_detected:
        set_bool(data, byte_index=1, bool_index=1, value=True)
    else:
        set_bool(data, byte_index=1, bool_index=0, value=True)
    plc.write_area(snap7.type.Areas.DB, 86, 0, data)

    data = bytearray(2)
    set_bool(data, byte_index=1, bool_index=1, value=False)
    set_bool(data, byte_index=1, bool_index=0, value=False)
    plc.write_area(snap7.type.Areas.DB, 86, 0, data)



try:
    while True:
        start_time = time.time()

        # Read proximity status
        proximity_status = read_proximity_status(0,1)
        # Trigger slot opening or closing based on proximity status
        if (proximity_status):
            trigger_slot_opening(True)

        # Record the end time
        end_time = time.time()

        # Calculate and print the time taken
        time_taken = end_time - start_time
        print(f"Proximity Status: {proximity_status}, Time Taken: {time_taken:.6f} seconds")

        # Add a small delay to avoid rapid polling
        time.sleep(0.1)

finally:
    # Disconnect from the PLC when done
    plc.disconnect()
