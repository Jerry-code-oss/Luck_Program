import tkinter as tk
from tkinter import ttk, messagebox
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import queue
class PiApproximationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Monte Carlo Pi Approximation')
        self.geometry('800x600')
        self.plot_queue = queue.Queue()
        self.create_widgets()
        self.process_plot_queue()
    def create_widgets(self):
        self.input_frame = ttk.Frame(self)
        self.input_frame.pack(padx=10, pady=10, fill='x')
        self.label = ttk.Label(self.input_frame, text='Number of Points:')
        self.label.pack(side='left', padx=(0, 10))
        self.num_points_entry = ttk.Entry(self.input_frame)
        self.num_points_entry.pack(side='left', expand=True, fill='x')
        self.plot_button = ttk.Button(self.input_frame, text='Plot', command=self.start_plot_thread)
        self.plot_button.pack(side='left', padx=(10, 0))
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
    def plot(self, total_points):
        in_circle = 0
        points = [(random.random(), random.random()) for _ in range(total_points)]
        in_circle = sum(1 for x, y in points if x**2 + y**2 <= 1)
        pi_approx = 4 * in_circle / total_points
        self.plot_queue.put((points, pi_approx))
    def start_plot_thread(self):
        try:
            total_points = int(self.num_points_entry.get())
            if 0 < total_points <= 100000:
                threading.Thread(target=self.plot, args=(total_points,), daemon=True).start()
            else:
                messagebox.showerror("Invalid Input", "Number of points must be greater than 0 and less than or equal to 100000.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer.")
    def process_plot_queue(self):
        try:
            points, pi_approx = self.plot_queue.get_nowait()
            inside_points = [(x, y) for x, y in points if x**2 + y**2 <= 1]
            outside_points = [(x, y) for x, y in points if x**2 + y**2 > 1]
            self.ax.clear()
            if inside_points:
                self.ax.scatter(*zip(*inside_points), color='green', s=1, label='Inside Circle')
            if outside_points:
                self.ax.scatter(*zip(*outside_points), color='red', s=1, label='Outside Circle')
            self.ax.set_title(f'Approximation of Pi: {pi_approx}')
            self.ax.legend()
            self.ax.axis('equal')
            self.canvas.draw()
        except queue.Empty:
            pass
        finally:
            self.after(100, self.process_plot_queue)
if __name__ == '__main__':
    app = PiApproximationGUI()
    app.mainloop()
