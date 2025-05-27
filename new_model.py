import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
#from tensorflow.keras.utils import to_categorical
import soundfile as sf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from tqdm import tqdm
import librosa
import librosa.display
from scipy.signal import butter, lfilter
import pywt
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# Klasör yolu
mypath_new = "Audio Files"


# Dosya isimlerini ve yolları
filenames_new = [f.strip() for f in os.listdir(mypath_new) if os.path.isfile(os.path.join(mypath_new, f)) and f.endswith('.wav')]  
filepaths_new = {f.strip(): os.path.join(mypath_new, f.strip()) for f in filenames_new}  # Dictionary olarak kaydettik



df_no_diagnosis = pd.read_csv('demographic_info.txt', names = 
                 ['Patient number', 'Age', 'Sex' , 'Adult BMI (kg/m2)', 'Child Weight (kg)' , 'Child Height (cm)'], delimiter = ' ')

diagnosis = pd.read_csv('Respiratory_Sound_Database/patient_diagnosis.csv', names = ['Patient number', 'Diagnosis'])


df =  df_no_diagnosis.join(diagnosis.set_index('Patient number'), on = 'Patient number', how = 'left')
df['Diagnosis'].value_counts()


root = 'Respiratory_Sound_Database/audio_and_txt_files'
filenames = [s.split('.')[0] for s in os.listdir(path = root) if '.txt' in s]


def Extract_Annotation_Data(file_name, root):
    tokens = file_name.split('_')
    recording_info = pd.DataFrame(data = [tokens], columns = ['Patient number', 'Recording index', 'Chest location','Acquisition mode','Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(root, file_name + '.txt'), names = ['Start', 'End', 'Crackles', 'Wheezes'], delimiter= '\t')
    return (recording_info, recording_annotations)



i_list = []
rec_annotations = []
rec_annotations_dict = {}
for s in filenames:
    (i,a) = Extract_Annotation_Data(s, root)
    i_list.append(i)
    rec_annotations.append(a)
    rec_annotations_dict[s] = a
recording_info = pd.concat(i_list, axis = 0)
recording_info.tail()



no_label_list = []
crack_list = []
wheeze_list = []
both_sym_list = []
filename_list = []
for f in filenames:
    d = rec_annotations_dict[f]
    no_labels = len(d[(d['Crackles'] == 0) & (d['Wheezes'] == 0)].index)
    n_crackles = len(d[(d['Crackles'] == 1) & (d['Wheezes'] == 0)].index)
    n_wheezes = len(d[(d['Crackles'] == 0) & (d['Wheezes'] == 1)].index)
    both_sym = len(d[(d['Crackles'] == 1) & (d['Wheezes'] == 1)].index)
    no_label_list.append(no_labels)
    crack_list.append(n_crackles)
    wheeze_list.append(n_wheezes)
    both_sym_list.append(both_sym)
    filename_list.append(f)




file_label_df = pd.DataFrame(data = {'filename':filename_list})
file_label_df


audio_path = 'Respiratory_Sound_Database/audio_and_txt_files/'


diagnosis = []
patient_number = []
recording_index = []
chest_location = []
acquisition_mode = []
recording_equipment = []
sample_rate = []
duration = []

for i in tqdm(range(len(file_label_df['filename']))):
  info = file_label_df['filename'][i].split('_')
  patient_id, recording_idx, chest_loc, acq_mode, equipment = info
  sound_filename = audio_path + file_label_df['filename'][i] + '.wav'
  x, sr = librosa.load(sound_filename)
  dur = round(x.shape[0]/sr, 2)
  sample_rate.append(sr)
  duration.append(dur)

  diagnosis.append(df['Diagnosis'][int(patient_id) - 101])
  patient_number.append(patient_id)
  recording_index.append(recording_idx)
  chest_location.append(chest_loc)
  acquisition_mode.append(acq_mode)
  recording_equipment.append(equipment)

file_label_df['Diagnosis'] = diagnosis
file_label_df['Patient Number'] = patient_number
file_label_df['Chest Location'] = chest_location
file_label_df['Acquisition Mode'] = acquisition_mode
file_label_df['Recording Equipment'] = recording_equipment
file_label_df['duration'] = duration
file_label_df['sample rate'] = sample_rate




diagnosis_3 = []
for i in range(len(file_label_df['Diagnosis'])):
  diagnosis = file_label_df['Diagnosis'][i]
  if diagnosis == 'COPD' or diagnosis == 'Bronchiectasis' or diagnosis == 'Asthma':
    diagnosis_3.append('Chronic Disease')
  elif diagnosis == 'URTI' or diagnosis == 'LRTI' or diagnosis == 'Pneumonia' or diagnosis == 'Bronchiolitis':
    diagnosis_3.append('Non-Chronic Disease')
  else:
    diagnosis_3.append('normal')

file_label_df['3 label diagnosis'] = diagnosis_3






# Örnek ses dosyası 
i = 666
sound_filename = audio_path + file_label_df['filename'][i] + '.wav'


y, sr = librosa.load(sound_filename, sr=None)  # sr=None, orijinal örnekleme hızını korur

# Sesin temel bilgileri
print(f"Örnekleme Hızı (Sample Rate): {sr}")
print(f"Ses Sinyali Boyutu: {len(y)}")
print(f"Minimum Değer: {np.min(y)}")
print(f"Maksimum Değer: {np.max(y)}")



"""
# FFT ile frekans analizi
fft = np.fft.fft(y)
frequencies = np.fft.fftfreq(len(fft), 1/sr)

# pozitif frekanslar
positive_freqs = frequencies[:len(frequencies)//2]
magnitude = np.abs(fft[:len(fft)//2])

# Frekans spektrumunu görselleştirme
plt.figure(figsize=(12, 6))
plt.plot(positive_freqs, magnitude)
plt.title("Frekans Spektrumu")
plt.xlabel("Frekans (Hz)")
plt.ylabel("Genlik")
plt.grid()
plt.show()
"""







# Bant geçiren filtre fonksiyonu
def bandpass_filter(data, lowcut, highcut, sr, order=5):
    nyquist = 0.5 * sr  # Nyquist frekansı
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    filtered_data = lfilter(b, a, data)
    return filtered_data


# Filtre uygula (300-1500 Hz)
lowcut = 300
highcut = 1500
filtered_y = bandpass_filter(y, lowcut, highcut, sr)




# Spektrogram çizimi
def plot_spectrogram(y, sr, title):
    plt.figure(figsize=(10, 6))
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(abs(S))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Zaman (s)")
    plt.ylabel("Frekans (Hz)")
    plt.show()

# Orijinal ve filtrelenmiş ses için spektrogram
plot_spectrogram(y, sr, "Orijinal Ses")
plot_spectrogram(filtered_y, sr, "Filtrelenmiş Ses")






# [0, 1] aralığında normalizasyon
y_min = np.min(y)
y_max = np.max(y)
y_normalized = (y - y_min) / (y_max - y_min)


y_normalized = librosa.util.normalize(filtered_y)



# Normalleştirilmiş sesler
sf.write("normalized_sound_01.wav", y_normalized, sr)  



# Orijinal ve normalleştirilmiş veri
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(y)
plt.title("Orijinal Ses")
plt.subplot(2, 1, 2)
plt.plot(y_normalized, color="orange")
plt.title("Normalleştirilmiş Ses [-1, 1]")
plt.tight_layout()
plt.show()













labels = []
labels_3 = []
preprocessed_tsfel = []
for i in tqdm(range(len(file_label_df['filename']))):
  labels.append(file_label_df['Diagnosis'][i])
  labels_3.append(file_label_df['3 label diagnosis'][i])
labels = np.array(labels)
labels_3 = np.array(labels_3)







def gogus_sesi_oznitelikleri(dosya_yolu, sr=16000):
    # Ses dosyasını yükle
    audio_file = dosya_yolu
    sr_new = 16000 # 16kHz sample rate
    x, sr = librosa.load(audio_file, sr=sr_new)
    
    lowcut = 300
    highcut = 1500
    filtered_y = bandpass_filter(x, lowcut, highcut, sr_new)
    
    y_normalized = librosa.util.normalize(filtered_y)
   
    
   
    max_len = 8 * sr_new  
    if y_normalized.shape[0] < max_len:
      # padding with zero
      pad_width = max_len - y_normalized.shape[0]
      y_normalized = np.pad(y_normalized, (0, pad_width))
    elif y_normalized.shape[0] > max_len:
      # truncated
      y_normalized = y_normalized[:max_len]
    
    
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
    
    # MFCC Öznitelikler
    mfcc = librosa.feature.mfcc(y=y_normalized, sr=sr_new, n_mfcc=25)
    for i in range(25):
        oznitelik[f'mfcc_{i+1}'] = np.mean(mfcc[i])
    
    # Wavelet Öznitelikler
    coeffs = pywt.wavedec(y_normalized, 'db4', level=5)
    for i, coeff in enumerate(coeffs):
        oznitelik[f'wavelet_energy_level_{i+1}'] = np.sum(np.square(coeff))
    
    return oznitelik




oznitelikler = []
for i in tqdm(range(len(file_label_df['filename']))):  
  audio_file = audio_path + file_label_df['filename'][i] + '.wav'
  data = gogus_sesi_oznitelikleri(audio_file)
  oznitelikler.append(list(data.values()))


df_oznitelikler = pd.DataFrame(oznitelikler)








def extract_disease_name(filename):
    parts = filename.split("/")[-1]     # Dosya adı
    parts = parts.split(",")[0]         # Hastalık adı + hasta ID
    disease_name = parts.split("_")[1]  # Hastalık adını çıkart
    return disease_name.strip()




disease_mapping = {
    'BRON': 'Bronchiectasis',
    'N': 'Healthy',
    'pneumonia': 'Pneumonia'
    
}


# Hastalıkları belirle
valid_diseases = {"Asthma", "COPD", "pneumonia", "BRON", "N"}

# Dosya isimleri ve hastalık adlarını içeren DataFrame 
file_label_df_new = pd.DataFrame({
    'filename': [f for f in filenames_new],  
    'class': [extract_disease_name(f) for f in filenames_new]
})



# Sadece belirli hastalıkları içerenleri alma
file_label_df_new = file_label_df_new[file_label_df_new['class'].isin(valid_diseases)].reset_index(drop=True)

# Hata kontrolü
missing_files = [f for f in file_label_df_new['filename'] if f not in filepaths_new]
if missing_files:
    print("Eşleşmeyen dosyalar:", missing_files)


oznitelikler_new = []

# Özellikleri çıkartma
for filename in tqdm(file_label_df_new['filename']):  
        audio_file = filepaths_new[filename]  
        data = gogus_sesi_oznitelikleri(audio_file)  
        oznitelikler_new.append(list(data.values()))  
   
# Özellikleri DataFrame olarak oluşturma
df_oznitelikler_new = pd.DataFrame(oznitelikler_new)

# Hastalık isimleri ile birleştirme
df_final = pd.concat([df_oznitelikler_new, file_label_df_new.drop("filename", axis=1)], axis=1)
df_final['class'] = df_final['class'].replace(disease_mapping)










# Özellik veri çerçevesine sınıf etiketini eklime
df_oznitelikler['class'] = file_label_df['Diagnosis']  





balanced_data = pd.concat([df_oznitelikler, df_final])






# LRTI sınıfını seç
lrti_data = balanced_data[balanced_data['class'] == 'LRTI'].copy()

def add_noise(y, noise_level=0.02):
    """Gaussian gürültü ekleme"""
    noise = noise_level * np.random.randn(len(y))
    return y + noise


augmented_samples = []
for i in range(5):  # 5 katı veri üretme
    noisy = lrti_data.copy()
    noisy.iloc[:, :-1] = noisy.iloc[:, :-1].apply(lambda x: add_noise(x), axis=1)
    augmented_samples.append(noisy)

# Hepsini birleştirme
augmented_data = pd.concat(augmented_samples, ignore_index=True)

# Yeni veriyi orijinal veriye ekleme
balanced_data = pd.concat([balanced_data, augmented_data], ignore_index=True)

print("Veri artırma tamamlandı. Yeni sınıf dağılımı:")
print(balanced_data['class'].value_counts())












# Özellikler (X) ve etiketler (y)
X = balanced_data.drop(columns=['class']).values  # Özellikler
y = balanced_data['class'].values                # Sınıf Etiketleri

# Veriyi ölçeklendirme 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)






# X = özellikler, y = etiketler 
selector = SelectKBest(score_func=f_classif, k=20)  # k: öznitelik sayısı
X_selected = selector.fit_transform(X_scaled, y)

# Seçilen özniteliklerin isimleri:
selected_indices = selector.get_support(indices=True)
print(selected_indices)







# SMOTE uygulama
smote = SMOTE( k_neighbors=5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected, y)


X_selected1 = pd.DataFrame(X_selected)

# Yeni veriyi veri çerçevesine dönüştürme
columns = X_selected1.columns[:]      # Özellik sütun isimleri 
df_resampled = pd.DataFrame(X_resampled, columns=columns)
df_resampled['class'] = y_resampled  # Sınıf etiketleri

# Kontrol
print("Orijinal sınıf dağılımı:")
print(balanced_data['class'].value_counts())

print("\nSMOTE sonrası sınıf dağılımı:")
print(df_resampled['class'].value_counts())




X = df_resampled.drop(columns=['class']).values  # Özellikler
y = df_resampled['class']                        # Etiketler




# one hot encoding labels
encoder = LabelEncoder()
i_labels = encoder.fit_transform(y)
#oh_labels = to_categorical(i_labels,num_classes=8)







X_train, X_test, y_train, y_test = train_test_split(X, i_labels, test_size=0.2, random_state=42)



model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    eval_metric='mlogloss',
    use_label_encoder=False
)
model.fit(X_train, y_train)
best_model = model

import pickle
with open("xgb_model.pkl", "wb") as f:
    pickle.dump((model, encoder, scaler), f)

"""
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200],
    'subsample': [0.6, 0.8, 1.0]
}

model = GridSearchCV(
    estimator=XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

model.fit(X_train, y_train)
"""




"""
print(model.best_params_)

best_model = model.best_estimator_
"""





# Eğitim doğruluğunu hesaplama
y_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy: ", train_accuracy)

# Test doğruluğunu hesaplama
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Testing Accuracy: ", test_accuracy)



preds = best_model.predict(X_test) # etiketler

classpreds = np.argmax(preds) # tahminler

y_testclass = np.argmax(y_test) # doğru sınıflar




# Etiketleri sınıf isimlerine dönüştürmeme
y_test_classes = encoder.inverse_transform(y_test)
preds_classes = encoder.inverse_transform(preds)





# Tahminleri ve gerçek etiketleri one-hot encoding formatına çevirme
n_classes = 8  # Sınıf sayısı

# Gerçek etiketleri one-hot encoded forma dönüştürme
y_test_oh = np.eye(n_classes)[y_test]

# Tahminleri one-hot encoded forma dönüştürme
if preds.ndim == 1:  
    preds_oh = np.eye(n_classes)[preds]
else:
    preds_oh = preds 


# ROC eğrisi ve AUC hesaplama
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_oh[:, i], preds_oh[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])



# ROC eğrilerini görselleştirme 
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class')
plt.legend(loc='lower right')
plt.show()





#conf_matrix
conf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()




cv_scores = cross_val_score(best_model, X, i_labels, cv=5, scoring='accuracy')

print(f"Çapraz Doğrulama Doğruluk Skorları: {cv_scores}")
print(f"Ortalama Doğruluk Skoru: {cv_scores.mean():.2f}")

cv_scores = cross_val_score(best_model, X, i_labels, cv=5, scoring='f1_macro')
print(f"Çapraz Doğrulama F1 Skorları: {cv_scores}")
print(f"Ortalama F1 Skoru: {cv_scores.mean():.2f}")

# Modelin performansını değerlendirme
print("Accuracy:", accuracy_score(y_test_classes, preds_classes))
print("\nClassification Report:\n", classification_report(y_test_classes, preds_classes))





# Tahmin
audio_file = audio_path + file_label_df['filename'][2] + '.wav'



hasta_sesi = "asthma.wav"



hasta_sesi_ozellikk = gogus_sesi_oznitelikleri(hasta_sesi)
hasta_sesi_ozellikk1 = list(hasta_sesi_ozellikk.values())


hasta_sesi_ozellikk2 = np.array(hasta_sesi_ozellikk1).reshape(1, -1)
hasta_sesi_scaled = scaler.transform(hasta_sesi_ozellikk2)
X_new_selected = hasta_sesi_scaled[:, selected_indices]

tahmin = best_model.predict(X_new_selected)





# Etiketi sınıf ismine dönüştürme

tahmin_sinif = encoder.inverse_transform(tahmin)

# Sonuç
print("Predicted class:", tahmin_sinif)
print("Predicted :", tahmin)



















