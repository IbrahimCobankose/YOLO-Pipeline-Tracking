from pymavlink import mavutil
import time

class AUVController:
    def __init__(self, connection_address="tcp:127.0.0.1:5762"):
        """Establishes a connection to the vehicle via the specified address."""
        print(f"Connecting to AUV at [{connection_address}]...")
        self.vehicle = mavutil.mavlink_connection(connection_address, baud=57600, autoreconnect=True)
        self.vehicle.wait_heartbeat()
        print(f"Connection successful! System ID: {self.vehicle.target_system}")

    def arm_and_set_mode(self, mode_name="GUIDED"):
        """Sets the vehicle mode and arms the motors."""
        # Set mode
        mode_id = self.vehicle.mode_mapping()[mode_name]
        self.vehicle.mav.set_mode_send(
            self.vehicle.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )
        print(f"Mode set to {mode_name}.")
        
        # Arm motors
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1, 0, 0, 0, 0, 0, 0
        )
        self.vehicle.motors_armed_wait()
        print("AUV motors armed. Ready to move.")

    def send_body_velocity(self, vx, vy, vz, yaw_rate):
        """
        Sends velocity commands relative to the vehicle's body frame (camera view).
        vx: Forward/Backward speed (m/s) -> Positive is forward
        vy: Right/Left speed (m/s) -> Positive is right
        vz: Up/Down speed (m/s) -> Positive is down (diving in underwater)
        yaw_rate: Rotation speed around its own axis (rad/s)
        """
        # MAV_FRAME_BODY_NED: Reference point is the vehicle itself (Camera angle)
        frame = mavutil.mavlink.MAV_FRAME_BODY_NED
        
        # Bitmask to ignore position and acceleration, only use velocity and yaw_rate
        mask = 0b0000011111000111 

        self.vehicle.mav.set_position_target_local_ned_send(
            0, # Time boot ms (ignored)
            self.vehicle.target_system, 
            self.vehicle.target_component,
            frame, 
            mask,
            0, 0, 0,     # x, y, z positions (Ignored by mask)
            vx, vy, vz,  # vx, vy, vz velocities (m/s)
            0, 0, 0,     # Accelerations (Ignored by mask)
            0, yaw_rate  # Yaw angle (Ignored), Yaw rate (rad/s)
        )