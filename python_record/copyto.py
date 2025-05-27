import subprocess

print("Dosya taşınıyor...")
subprocess.run([
    'sudo', 'cp',
    '/home/pi/python_record2/recorded_16khz.wav',
    '/home/pi/model_v5/selected.wav'
])
