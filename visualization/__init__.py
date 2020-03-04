import multiprocessing as mp
import sys

import matplotlib.pyplot as plt

if plt.get_backend() == "MacOSX":
    mp.set_start_method("forkserver")


class Processor(object):
    interval = None
    pipe = None

    def call_back(self):
        pass

    def __call__(self, pipe):
        self.pipe = pipe
        self.fig, self.ax = plt.subplots()
        self.ax.set_autoscale_on(True)
        timer = self.fig.canvas.new_timer(interval=self.interval)
        timer.add_callback(self.call_back)
        timer.start()

        plt.show()

    def terminate(self):
        plt.close('all')

    def clear(self):
        self.ax.cla()

    def flush(self):
        self.fig.canvas.draw()


class WindowController(object):
    def __init__(self, plotter):
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = plotter
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()

    def broken_pipe_message(self):
        print("[WARNING] Connection to plotter " + str(self) + " is broken...", file=sys.stderr)
        return False

    def send(self, data):
        send = self.plot_pipe.send
        send(data)

    def stop(self):
        self.send(None)
