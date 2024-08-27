import serial
import tkinter as tk

# Configure the serial port
# ser = serial.Serial('COM3', 9600)  # Replace 'COM3' with your STM32's serial port

# Function to send command to STM32
def rotate_clockwise():
    ser.write(b'C')  # Send 'C' for clockwise rotation

def rotate_anticlockwise():
    ser.write(b'A')  # Send 'A' for anticlockwise rotation

# Setup the GUI
root = tk.Tk()
root.title("Motor Control")

# Clockwise button
clockwise_button = tk.Button(root, text="Rotate Clockwise", command=rotate_clockwise)
clockwise_button.pack(pady=10)

# Anticlockwise button
anticlockwise_button = tk.Button(root, text="Rotate Anticlockwise", command=rotate_anticlockwise)
anticlockwise_button.pack(pady=10)

# Start the GUI
root.mainloop()

# Close the serial port when done
ser.close()
