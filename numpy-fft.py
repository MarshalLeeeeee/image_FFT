import numpy as np
from matplotlib.pyplot import plot, show
x = np.linspace(0, 2 * np.pi, 30) 
wave = np.cos(x)
transformed = np.fft.fft(wave)
anti_transformed = np.fft.ifft(transformed)
shifted_tarnsformed = np.fft.fftshift(transformed)
anti_shifted_transformed = np.fft.ifft(shifted_tarnsformed)
print(x)
#plot(x) # blue line
print(wave)
#plot(wave) # orange line
print(transformed.shape)
#plot(transformed) # green line
#plot(anti_transformed)
print(shifted_tarnsformed)
plot(shifted_tarnsformed)
#plot(anti_shifted_transformed)
print(np.all(np.abs(np.fft.ifft(transformed) - wave) < 10 ** -9))
#plot(transformed)
show()