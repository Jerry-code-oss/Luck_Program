import tkinter as tk
from tkinter import ttk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
def approximate_pi(n):
    angle = np.pi / n
    side_length = 2 * np.sin(angle)
    perimeter = n * side_length
    return perimeter / 2
def draw_polygon_and_circle(n, ax):
    if n < 3:
        n = 3
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_polygon = np.cos(angles)
    y_polygon = np.sin(angles)
    ax.clear()
    ax.plot(np.append(x_polygon, x_polygon[0]), np.append(y_polygon, y_polygon[0]), 'b-', label='Polygon')
    circle = plt.Circle((0, 0), 1, color='r', fill=False, label='Circle')
    ax.add_artist(circle)
    ax.set_aspect('equal', 'box')
    ax.legend()
def on_slider_change(event):
    n = int(slider.get())
    draw_polygon_and_circle(n, ax)
    canvas.draw()
    pi_approx = approximate_pi(n)
    sides_label.config(text=f"Number of sides: {n}")
    pi_label.config(text=f"Approximated π value: {pi_approx:.10f}\nActual π value: {np.pi:.10f}")
root = tk.Tk()
root.title("Polygon Approximation of Circle")
slider = ttk.Scale(root, from_=3, to=1000, orient='horizontal', value=100, command=on_slider_change)
slider.pack(fill=tk.X, padx=20, pady=20)
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill=tk.BOTH, expand=True)
n_init = int(slider.get())
draw_polygon_and_circle(n_init, ax)
sides_label = tk.Label(root, text="", font=('Arial', 12))
sides_label.pack(pady=5)
pi_label = tk.Label(root, text="", font=('Arial', 12))
pi_label.pack(pady=20)
sides_label.config(text=f"Number of sides: {n_init}")
pi_approx = approximate_pi(n_init)
pi_label.config(text=f"Approximated π value: {pi_approx:.10f}\nActual π value: {np.pi:.10f}")
root.mainloop()
