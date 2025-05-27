import pyaudio
import wave
import time

# Kayıt ayarları
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 1024
RECORD_SECONDS = 8
OUTPUT_FILENAME = "recorded.wav"

# PyAudio başlat
audio = pyaudio.PyAudio()

# Mikrofon (card 1, device 0) için aygıt indeksini bul
device_index = None
for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    if "USB" in info["name"] and info["maxInputChannels"] > 0:
        device_index = i
        break

if device_index is None:
    print("Uygun mikrofon bulunamadı.")
    exit()

print(f"Kayıt başlatılıyor... ({RECORD_SECONDS} saniye)")
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)

frames = []

for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Kayıt tamamlandı.")

# Temizlik ve dosyaya yazma
stream.stop_stream()
stream.close()
audio.terminate()

wf = wave.open(OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print(f"'{OUTPUT_FILENAME}' dosyası başarıyla kaydedildi.")
