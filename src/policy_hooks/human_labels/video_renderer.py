import numpy as np
from scipy.ndimage import zoom
from threading import Thread
import time

class Im(object):
    def __init__(self, display=None):
        global pyglet
        global Label
        global key
        global Window
        import pyglet
        from pyglet.text import Label
        from pyglet.window import key, Window
        self.window = None
        self.isopen = False
        self.display = display
        self.last_keystroke = ''
        self._rec_input = False
        self._text_h = 512
        self._text_w = 512
        self._text_c = 3
        self._waiting = False


    def wait(self):
        self._waiting = True
        self.txtshow('Waiting for data; no input required at this time')


    def cont(self):
        self._waiting = False
        self.txtshow('Renderer ready!')


    def wait_for_input(self, timeout=None):
        if self.window is None: return
        init_t = time.time()
        while not self._rec_input and (timeout is None or time.time() - init_t < timeout):
            self.window.dispatch_events()


    def check_input(self):
        if self.window is None: return
        self.window.dispatch_events()


    def txtshow(self, text):
        if self.window is None:
            width, height = self._text_w, self._text_h
            self.window = pyglet.window.Window(
                width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.channels = self._text_c
            self.isopen = True

            @self.window.event
            def on_key_press(symbol, modifiers):
                self.last_keystroke = key.symbol_string(symbol).lower().strip('_')
                print('\n\nReceived keyboard input', self.last_keystroke, '\n\n')
                self._rec_input = True

        label = Label(text, font_name='Times New Roman', font_size=12,
                      anchor_x='center', anchor_y='center',
                      x=self.window.width//2, y=self.window.height//2)

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        label.draw()
        self.window.flip()


    def imshow(self, arr):
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(
                width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.channels = channels
            self.isopen = True

            @self.window.event
            def on_key_press(symbol, modifiers):
                self.last_keystroke = key.symbol_string(symbol).lower().strip('_')
                print('\n\nReceived keyboard input', self.last_keystroke, '\n\n')
                self._rec_input = True

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

    def __init__(self, mode=0, fps=4, zoom=3, playback_speed=1, channels=3):
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
        self._waiting = False


    def wait(self):
        if not self._waiting:
            if self.v is not None: self.v.close()
            self.v = Im()
            self._waiting = True
            self.v.wait()

    
    def cont(self):
        if self._waiting:
            self._waiting = False
            if self.v is not None:
                self.v.cont()
                time.sleep(0.5)
                self.v.close()


    def stop(self):
        self.stop_render = True


    def threaded_render(self, frames):
        render_thread = Thread(target=self.render, args=(frames,))
        render_thread.start()


    def show_transition(self):
        if self.v is not None: self.v.close()
        v = Im()
        v.txtshow('Inputs saved; loading next trajectory')
        time.sleep(0.3)
        v.close()


    def get_time(self):
        return self.cur_t

    
    def wait_for_user(self):
        if self.v is not None: self.v.close()
        v = Im()
        v.txtshow('Render ready; please hit any key...')
        v.wait_for_input()
        v.close()


    def interactive_render(self, frames, max_iters=10):
        if self.v is not None: self.v.close()
        v = Im()
        t = 0
        cur_iter = 0
        self.stop_render = False
        keystroke = 'u'
        t_limit = 60
        init_t = time.time()
        while not self.stop_render and \
              time.time() - init_t < t_limit:
            if t >= len(frames): t = 0
            if t == 0: cur_iter += 1
            self.cur_t = t
            start = time.time()
            zoomed_frame = zoom(frames[t], self.zoom_factor, order=1)
            v.imshow(zoomed_frame)
            end = time.time()
            render_time = end - start
            t += min(len(frames)-t, self.playback_speed)
            v.check_input()
            if v._rec_input:
                keystroke = v.last_keystroke
                #v.close()
                #v = Im()
                #v.txtshow('Found user input {}'.format(keystroke))
                #time.sleep(0.1)
                break
            sleep_time = max(0, self.sleep_time-render_time)
            time.sleep(sleep_time)

        '''
        while keystroke == '':
            v.close()
            v = Im()
            v.txtshow('No label detected; please enter now (press u if unsure)')
            v.wait_for_input()
            keystroke = v.last_keystroke
        '''

        v.close()
        return keystroke, self.cur_t


    def render(self, frames, max_iters=20):
        if self.v is not None: self.v.close()
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

