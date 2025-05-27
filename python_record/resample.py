import soundfile as sf
from scipy.signal import resample

# Giriş ve çıkış dosyalarının yolları
input_path = '/home/pi/python_record2/recorded.wav'
output_path = '/home/pi/python_record2/recorded_16khz.wav'

# WAV dosyasını oku
data, samplerate = sf.read(input_path)

# Hedef örnekleme frekansı
target_rate = 16000

# Yeniden örnekleme (veri uzunluğu oranına göre)
number_of_samples = round(len(data) * target_rate / samplerate)
resampled_data = resample(data, number_of_samples)

# Yeni dosyayı yaz
sf.write(output_path, resampled_data, target_rate)

print(f"Dönüştürüldü: {output_path}")
