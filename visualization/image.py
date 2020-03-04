from visualization import Processor, WindowController


class ProcessImage(Processor):
    def __init__(self, interval=1000):
        self.image = None
        self.interval = interval

    def set_image(self, image):
        self.image = image

    def draw_image(self, flush=True):
        image = self.image
        if image is not None:
            self.ax.imshow(image)
        if flush:
            self.flush()

    def draw_all(self):
        self.clear()
        self.ax.axis("off")
        self.draw_image(flush=False)
        self.flush()

    def call_back(self):
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            self.set_image(command)
        self.draw_all()
        return True


class ImageWindow(WindowController):
    def __init__(self, interval=1000):
        super().__init__(ProcessImage(interval=interval))

    def set_image(self, image):
        try:
            self.send(image)
        except BrokenPipeError as e:
            return self.broken_pipe_message()

        return True
