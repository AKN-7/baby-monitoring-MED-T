"""
Arducam Pan-Tilt module wrapper for the PCA9685 PWM controller.
This file acts as a bridge between our app and the Arducam PCA9685 library.
"""

import os
import subprocess
import time

class PanTiltController:
    """
    Python wrapper for the Arducam Pan-Tilt module.
    Uses the compiled binary from the PCA9685 repository.
    """
    
    def __init__(self):
        """Initialize the controller with default step size and positions."""
        self.step_size = 10  # Default step size (in degrees)
        self.current_pan = 90  # Middle position (range 0-180)
        self.current_tilt = 90  # Middle position (range 0-180)
        
        # Path to the Arducam binary - you may need to adjust this path
        # based on where you cloned the PCA9685 repository
        self.binary_path = "/home/pi/PCA9685/RunServoDemo"
        
        if not os.path.exists(self.binary_path):
            print(f"WARNING: Arducam binary not found at {self.binary_path}")
            print("Please update the binary_path in arducam_pan_tilt.py")
        else:
            print(f"Found Arducam binary at {self.binary_path}")
            
        # Initialize servos to center position
        self._set_servo_position(0, self.current_pan)  # Pan servo
        self._set_servo_position(1, self.current_tilt)  # Tilt servo
    
    def _set_servo_position(self, channel, angle):
        """
        Set a servo to a specific angle.
        Channel 0 = Pan, Channel 1 = Tilt
        """
        if not os.path.exists(self.binary_path):
            print(f"Would set servo {channel} to {angle}° (binary not found)")
            return
        
        try:
            # Using Arducam's binary to control the servo
            cmd = f"sudo {self.binary_path} --channel={channel} --angle={angle}"
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            print(f"Error controlling servo: {e}")
    
    def pan_right(self):
        """Pan the camera right (decrease angle)."""
        if self.current_pan > self.step_size:
            self.current_pan -= self.step_size
            self._set_servo_position(0, self.current_pan)
            print(f"Pan right: {self.current_pan}°")
        else:
            print("Pan at right limit")
    
    def pan_left(self):
        """Pan the camera left (increase angle)."""
        if self.current_pan < (180 - self.step_size):
            self.current_pan += self.step_size
            self._set_servo_position(0, self.current_pan)
            print(f"Pan left: {self.current_pan}°")
        else:
            print("Pan at left limit")
    
    def tilt_up(self):
        """Tilt the camera up (decrease angle)."""
        if self.current_tilt > self.step_size:
            self.current_tilt -= self.step_size
            self._set_servo_position(1, self.current_tilt)
            print(f"Tilt up: {self.current_tilt}°")
        else:
            print("Tilt at upper limit")
    
    def tilt_down(self):
        """Tilt the camera down (increase angle)."""
        if self.current_tilt < (180 - self.step_size):
            self.current_tilt += self.step_size
            self._set_servo_position(1, self.current_tilt)
            print(f"Tilt down: {self.current_tilt}°")
        else:
            print("Tilt at lower limit")
            
    def center(self):
        """Center both pan and tilt servos."""
        self.current_pan = 90
        self.current_tilt = 90
        self._set_servo_position(0, self.current_pan)
        self._set_servo_position(1, self.current_tilt)
        print("Centered pan and tilt")

# Test the controller if this file is run directly
if __name__ == "__main__":
    print("Testing Pan-Tilt controller...")
    controller = PanTiltController()
    
    print("Centering...")
    controller.center()
    time.sleep(1)
    
    print("Panning left...")
    controller.pan_left()
    time.sleep(1)
    
    print("Panning right...")
    controller.pan_right()
    time.sleep(1)
    
    print("Tilting up...")
    controller.tilt_up()
    time.sleep(1)
    
    print("Tilting down...")
    controller.tilt_down()
    time.sleep(1)
    
    print("Centering again...")
    controller.center()
    print("Test complete!")