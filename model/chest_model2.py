from scipy.signal import butter, lfilter
import librosa
import numpy as np
import pywt
import cloudpickle as pickle



# Bant geçiren filtre fonksiyonu
def bandpass_filter(data, lowcut, highcut, sr, order=5):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    filtered_data = lfilter(b, a, data)
    return filtered_data







# Ses özellikleri çıkarma fonksiyonu
def gogus_sesi_oznitelikleri(dosya_yolu, sr=16000):
    # Ses dosyasını yükle
    audio_file = dosya_yolu
    sr_new = 16000 # 16kHz sample rate
    x, sr = librosa.load(audio_file, sr=sr_new)
    
    lowcut = 300
    highcut = 1500
    filtered_y = bandpass_filter(x, lowcut, highcut, sr_new)
      
   
    max_len = 8 * sr_new 
    if filtered_y.shape[0] < max_len:
      pad_width = max_len - filtered_y.shape[0]
      y_normalized = np.pad(filtered_y, (0, pad_width))
    elif filtered_y.shape[0] > max_len:
      filtered_y = filtered_y[:max_len]
    

    y_normalized = librosa.util.normalize(filtered_y)
    
    # Temel Öznitelikler
    oznitelik = {}
    oznitelik['rms'] = np.mean(librosa.feature.rms(y=y_normalized))
    oznitelik['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y=y_normalized))
    
    # Spektral Öznitelikler
    S = np.abs(librosa.stft(y_normalized))
    oznitelik['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(S=S, sr=sr_new))
    oznitelik['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr_new))
    oznitelik['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr_new))
    
    y_harmonic, y_percussive = librosa.effects.hpss(y_normalized)
    oznitelik['harmonic_ratio'] = np.mean(y_harmonic / (y_percussive + 1e-6))
    
    # Chroma Öznitelikler
    chroma = librosa.feature.chroma_stft(y=y_normalized, sr=sr_new)
    oznitelik['chroma_mean'] = np.mean(chroma)
    
    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y_normalized, sr=sr_new)
    oznitelik['spectral_contrast'] = np.mean(contrast)
    
    
    # MFCC Özellikleri
    mfcc = librosa.feature.mfcc(y=y_normalized, sr=sr_new, n_mfcc=25)
    for i in range(25):
        oznitelik[f'mfcc_{i+1}'] = np.mean(mfcc[i])
    
    # Wavelet Özellikleri
    coeffs = pywt.wavedec(y_normalized, 'db4', level=5)
    for i, coeff in enumerate(coeffs):
        oznitelik[f'wavelet_energy_level_{i+1}'] = np.sum(np.square(coeff))
    
    return oznitelik







selected_indices = np.array([ 0,  2,  3,  4,  6,  8,  9, 10, 11, 14, 17, 20, 21,24, 25, 28, 29, 30, 31, 32])




# Model yükleme
with open("xgb_model.pkl", "rb") as f:
    loaded_model, loaded_encoder, loaded_scaler = pickle.load(f)




# Hastanın sesi 
hasta_sesi = "selected.wav"

hasta_sesi_ozellik = gogus_sesi_oznitelikleri(hasta_sesi)
hasta_sesi_ozellik1 = list(hasta_sesi_ozellik.values())
hasta_sesi_ozellik2 = np.array(hasta_sesi_ozellik1).reshape(1, -1)


hasta_sesi_scaled = loaded_scaler.transform(hasta_sesi_ozellik2)


X_new_selected = hasta_sesi_scaled[:, selected_indices]


tahmin = loaded_model.predict(X_new_selected)

# Tahmini orijinal sınıf etiketine çevir
tahmin_sinif = loaded_encoder.inverse_transform(tahmin)

print("Hastanın tahmin edilen sınıfı:", tahmin_sinif[0])







