import time

import visualization.plots

if __name__ == '__main__':
    window = visualization.plots.PlotterWindow()
    window.new_line("testing", "r-")
    for i in range(-100, 100):
        print(i)
        window.add_point("testing", i, i ** 3)
        time.sleep(0.05)
