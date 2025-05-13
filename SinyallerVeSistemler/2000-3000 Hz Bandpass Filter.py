#2000-3000 Hz Bant Geçiren Filtre Uygulaması 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from pydub import AudioSegment

# Dosya yükleme
file_path = r"C:\Users\sertv\Desktop\Python\2000_3000_bandpass.mp3"
audio = AudioSegment.from_file(file_path)
samples = np.array(audio.get_array_of_samples())
sample_rate = 44100  # Örnekleme frekansı

# Bant Geçiren Filtre Tasarımı
low_cut = 2000
high_cut = 3000
order = 5  # Filtre mertebesi
sos = butter(order, [low_cut, high_cut], btype='bandpass', fs=sample_rate, output='sos')

# Filtreleme işlemi
filtered_signal = sosfilt(sos, samples)

# Frekans uzayı analizi
freqs = np.fft.rfftfreq(len(samples), d=1/sample_rate)
original_fft = np.abs(np.fft.rfft(samples))
filtered_fft = np.abs(np.fft.rfft(filtered_signal))

# Grafikler
plt.figure(figsize=(12, 6))

# Zaman ekseninde sinyal
plt.subplot(2, 1, 1)
plt.plot(samples, label="Orijinal Sinyal", alpha=0.7)
plt.plot(filtered_signal, label="Filtrelenmiş Sinyal", alpha=0.7)
plt.title("2000-3000 Hz Bant Geçiren Filtre - Zaman Uzayı")
plt.legend()

# Frekans ekseninde sinyal
plt.subplot(2, 1, 2)
plt.plot(freqs, original_fft, label="Orijinal FFT", alpha=0.7)
plt.plot(freqs, filtered_fft, label="Filtrelenmiş FFT", alpha=0.7)
plt.title("2000-3000 Hz Bant Geçiren Filtre - Frekans Uzayı")
plt.legend()

plt.tight_layout()
plt.show()
