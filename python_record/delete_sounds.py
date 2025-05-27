import subprocess
import time

print("Eski dosyalar siliniyor...")
subprocess.run(['sudo', 'rm', '-f', '/home/pi/model_v5/selected.wav'])
subprocess.run(['sudo', 'rm', '-f', '/home/pi/python_record2/recorded.wav'])
subprocess.run(['sudo', 'rm', '-f', '/home/pi/python_record2/recorded_16khz.wav'])
