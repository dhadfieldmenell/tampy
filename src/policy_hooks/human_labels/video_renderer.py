import numpy as np
import pyglet
from scipy.ndimage import zoom
from threading import Thread
import time

class Im(object):
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display

    def imshow(self, arr):
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(
                width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.channels = channels
            self.isopen = True

        assert arr.shape == (self.height, self.width, self.channels), \
            "You passed in an image with the wrong number shape"
        flipped_arr = np.flip(arr, axis=0)
        image = pyglet.image.ImageData(self.width, self.height,
                                       'RGB', flipped_arr.tobytes())
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0)
        self.window.flip()

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()


class VideoRenderer:
    play_through_mode = 0
    restart_on_get_mode = 1

    def __init__(self, mode, fps=4, zoom=1, playback_speed=1, channels=3):
        assert mode == VideoRenderer.restart_on_get_mode or mode == VideoRenderer.play_through_mode
        self.mode = mode
        self.channels = channels
        if self.channels == 1:
            self.zoom_factor = zoom
        else:
            self.zoom_factor = [zoom]*(self.channels-1) + [1]
        self.playback_speed = playback_speed
        self.stop_render = False
        self.current_frames = None
        self.v = None
        self.fps = fps
        self.sleep_time = 1./self.fps
        self.cur_t = 0


    def stop(self):
        self.stop_render = True


    def threaded_render(self, frames):
        render_thread = Thread(target=self.render, args=(frames,))
        render_thread.start()


    def get_time(self):
        return self.cur_t


    def render(self, frames, max_iters=5):
        v = Im()
        t = 0
        cur_iter = -1
        self.stop_render = False
        while not self.stop_render and cur_iter < max_iters:
            if t >= len(frames): t = 0
            if t == 0: cur_iter += 1
            self.cur_t = t
            start = time.time()
            zoomed_frame = zoom(frames[t], self.zoom_factor, order=1)
            v.imshow(zoomed_frame)
            end = time.time()
            render_time = end - start
            t += min(len(frames)-t, self.playback_speed)
            sleep_time = max(0, self.sleep_time-render_time)
            time.sleep(sleep_time)
        v.close()

