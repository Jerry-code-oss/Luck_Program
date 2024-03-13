import numpy as np
import matplotlib.pyplot as plt
import math
from ipywidgets import interact, FloatSlider, IntSlider
import ipywidgets as widgets
def fx(t):
    return 2 * np.cos(t) - np.cos(2*t)
def fy(t):
    return 2 * np.sin(t) - np.sin(2*t)
def ft(t):
    return fx(t) + 1j * fy(t)
def calF(f, dx, left, right):
    xNum = np.linspace(left, right, int((right-left) / dx))
    Sum = np.sum(f(xNum) * dx)
    return Sum
def calculate_fourier_series(t, n_min, n_max, dx):
    T = 2 * math.pi
    wo = 2 * math.pi / T
    c = []
    for i in range(n_min, n_max + 1):
        tmpf = lambda x: ft(x) * np.exp(-1j * i * wo * x)
        nowc = calF(tmpf, dx, 0, T) / T
        c.append((i, nowc))
    tx = []
    ty = []
    for ti in t:
        num = sum(nowc * np.exp(1j * n * wo * ti) for n, nowc in c)
        tx.append(num.real)
        ty.append(num.imag)
    return tx, ty
def interactive_plot(n_min=-30, n_max=30, dx=0.001):
    t = np.linspace(0, 2 * math.pi, 100)
    tx, ty = calculate_fourier_series(t, n_min, n_max, dx)
    plt.figure(figsize=(12, 6))
    x = fx(t)
    y = fy(t)
    plt.subplot(1, 2, 1)
    plt.plot(x, y, c='blue')
    plt.title('Original Data')
    plt.subplot(1, 2, 2)
    plt.scatter(tx, ty, c='black')
    plt.title('Fourier Series Approximation')
    plt.tight_layout()
    plt.show()
interact(interactive_plot,
         n_min=IntSlider(value=-30, min=-100, max=0, step=1, description='n_min:'),
         n_max=IntSlider(value=30, min=0, max=100, step=1, description='n_max:'),
         dx=FloatSlider(value=0.001, min=0.0001, max=0.01, step=0.0001, description='dx:'))
