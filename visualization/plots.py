from visualization import Processor, WindowController


class ProcessPlotter(Processor):
    ADD_POINT = 1
    NEW_LINE = 2

    def __init__(self, interval=1000):
        self.data = {}
        self.interval = interval

    def new_line(self, name, style):
        self.data[str(name)] = {
            'x': [],
            'y': [],
            'fmt': style,
        }

    def add_point(self, line_name, x, y):
        line_name = str(line_name)
        data = self.data
        data[line_name]['x'].append(x)
        data[line_name]['y'].append(y)

    def draw_line(self, line_name, flush=True):
        line = self.data[line_name]
        if line['fmt'] is None:
            self.ax.plot(line['x'], line['y'], label=line_name)
        else:
            self.ax.plot(line['x'], line['y'], line['fmt'], label=line_name)
        if flush:
            self.flush()

    def draw_all(self):
        self.clear()
        for line in self.data:
            self.draw_line(line, flush=False)
        if len(self.data):
            self.ax.legend()
        self.flush()

    def call_back(self):
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            query = command[0]
            if query == self.ADD_POINT:
                self.add_point(
                    line_name=command[1],
                    x=command[2],
                    y=command[3]
                )
            elif query == self.NEW_LINE:
                self.new_line(
                    name=command[1],
                    style=None if len(command) < 3 else command[2]
                )
        self.draw_all()
        return True


class PlotterWindow(WindowController):
    def __init__(self, interval=1000):
        super().__init__(ProcessPlotter(interval=interval))

    def new_line(self, name, style=None):
        try:
            self.send((self.plotter.NEW_LINE, name, style))
        except BrokenPipeError as e:
            return self.broken_pipe_message()

        return True

    def add_point(self, line_name, x, y):
        try:
            self.send((self.plotter.ADD_POINT, line_name, x, y))
        except BrokenPipeError as e:
            return self.broken_pipe_message()

        return True
