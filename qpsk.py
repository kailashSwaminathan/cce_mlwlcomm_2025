# qpsk receipe src: https://www.mycompiler.io/view/6SATRpYKM8j

import numpy as np
import matplotlib.pyplot as plt

# Define parameters
# Reduced carrier frequency (Hz)
fc = 5 
# Sampling frequency (Hz)
fs = 1000
# Duration of each bit (s)
bit_duration = 1
# number of bits
num_bits = 4
# Samples per bit
num_samples_per_bit = int(bit_duration * fs)

# Define bit sequence
bit_sequence = [(0,0), (1,0), (0,1), (1,1)]

# Generate QPSK modulated signal
t = np.linspace(0, bit_duration * num_bits, num_samples_per_bit * num_bits)
qpsk_signal = np.zeros(num_samples_per_bit * num_bits)
for i, (data_I, data_Q) in enumerate(bit_sequence):
    start_idx = i * num_samples_per_bit
    end_idx = start_idx + num_samples_per_bit
    # Phase shift for QPSK modulation
    phase = (data_I * 2 + data_Q) * np.pi/2
    qpsk_signal[start_idx:end_idx] = np.cos(2 * np.pi * fc * t[start_idx:end_idx] + phase)
    
# Generate I and Q signals
data_I_signal = np.zeros(num_samples_per_bit * num_bits)
data_Q_signal = np.zeros(num_samples_per_bit * num_bits)
for i, (data_I, data_Q) in enumerate(bit_sequence):
    start_idx = i * num_samples_per_bit
    end_idx = start_idx + num_samples_per_bit
    data_I_signal[start_idx:end_idx] = data_I
    data_Q_signal[start_idx:end_idx] = data_Q

# Plot
plt.figure(figsize=(12,6))
plt.subplot(3,1,1)
plt.plot(t, qpsk_signal)
plt.title("QPSK Modulated signal with Bit Sequence")
plt.xlabel('Time (s)')
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3,1,2)
plt.step(t, data_I_signal, where='post')
plt.title('Data I signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.ylim(-0.1, 1.1)
plt.grid(True)

plt.subplot(3,1,3)
plt.step(t, data_Q_signal, where='post')
plt.title('Data Q signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.ylim(-0.1, 1.1)
plt.grid(True)

plt.tight_layout()
plt.show()