import numpy as np
import os
from scipy.ndimage import zoom
import time

import pyglet
from pyglet.text import Label
from pyglet.window import key, Window


class Im(object):
    def __init__(self, display=None):
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

    def __init__(self, mode=0, fps=8, zoom=3, playback_speed=1, channels=3):
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
                break
            sleep_time = max(0, self.sleep_time-render_time)
            time.sleep(sleep_time)

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


class LabelInterface(object):
    def __init__(self):
        self.failures = []
        self.successes = []
        self.fail_files = []
        self.suc_files = []
        self.labels = []
        self.cur_file = 0
        self.label_file = 'human_labels_{}'.format(self.cur_file)
        self.n = 0
        self.renderer = VideoRenderer()
        self.user_response_timeout = 1.
        self.video_renderer = VideoRenderer()
        self.label_dir = 'labels/'
        self.stopped = False
        if not os.path.exists(self.label_dir):
            os.mkdir(self.label_dir)

        while os.path.exists(self.label_dir+self.label_file+'.npy'):
            self.cur_file += 1
            self.label_file = 'human_labels_{}'.format(self.cur_file)


    def run(self, dname):
        self._find_files(dname)
        while not self.stopped:
            val = np.random.uniform() < 0.1
            self.search_query(t=6, N=10, val=val)
            self.write_labels_to_file()

       
    def _find_files(self, dname):
        fnames = os.listdir(dname)
        for fname in fnames:
            if fname.find('suc') >= 0:
                self.suc_files.append(dname+fname)
            elif fname.find('fail') >= 0:
                self.fail_files.append(dname+fname)


    def load_from_directory(self, dname):
        fnames = os.listdir(dname)
        for fname in fnames:
            if fname.find('pkl') < 0: continue
            self.load_from_file(fname)


    def load_from_file(self, fname):
        data = np.load(fname, allow_pickle=True)
        for t in range(len(data[0])):
            x = data[1][t]
            if np.all(np.abs(x[-1]-x[0]) < 0.1): continue
            if data[-1][t]:
                self.successes.append([val[t] for val in data])
            else:
                self.failures.append([val[t] for val in data])
                self.n += 1


    def write_labels_to_file(self):
        np.save(self.label_dir+self.label_file, self.labels)


    def get_example(self, val=False):
        val = val and len(self.suc_files)
        buf = self.successes if val else self.failures
        file_buf = self.fail_files if not val else self.suc_files
        while not len(buf) and len(file_buf):
            ind = np.random.randint(len(file_buf))
            fname = file_buf.pop(ind)
            self.load_from_file(fname)

        if not len(buf):
            self.stopped = True
            return None

        ind = np.random.randint(len(buf))
        example = buf.pop(ind)
        return example


    def search_query(self, t=10, N=1, max_iters=20, val=False):
        print('\nRunning search query...\n')
        example = self.get_example(val)
        if example is None:
            self.stopped = True
            return

        assert example[-1] == val
        hor = len(example[0])
        cur_t = 0
        invalid_input = False
        ts = []
        cur_iter = 0
        st, et = 0, T
        while st >= 0 and et <= hor:
            cur_iter += 1
            invalid_input = False
            res, label_t = self.query_user(example, (st, et))

            print('\n\nInterpreted input as: {}\n\n'.format(res))
            if res == 'before':
                st -= T // 2
                et -= T // 2
            elif res == 'after':
                st += T // 2
                et += T // 2
                ts.extend([st, cur_t, et-1])
            elif res == 'during':
                ts.append(st)
                break
            elif res == 'stop':
                self.stopped = True
                break
            elif res == 'ignore':
                break
            else:
                invalid_input = True
                print('Invalid search query', res)

        for i, t in enumerate(ts[-N:]):
            self.labels.append((res, example[1][t], example[2], example[3], example[-1], len(ts[-N:])-i-1))



    def binary_search_query(self, t=10, N=1, max_iters=20, val=False):
        print('\nRunning search query...\n')
        example = self.get_example(val)
        if example is None:
            self.stopped = True
            return

        assert example[-1] == val
        hor = len(example[0])
        cur_t = hor // 2
        invalid_input = False
        visited = set()
        ts = []
        cur_iter = 0
        wind = 1 # Consider nearby timesteps visited

        a, b = 0, hor
        prev_a, prev_b = -1, -1
        st, et = max(0, cur_t - t//2), min(cur_t + t//2, hor)
        while cur_iter < max_iters and \
              (cur_t not in visited or invalid_input) and \
              (a != prev_a or b != prev_b):

            cur_iter += 1
            invalid_input = False
            cur_t = max(0, min(cur_t, hor))
            visited.update(list(range(cur_t-wind, cur_t+wind+1)))
            res, label_t = self.query_user(example, (st, et))
            prev_a, prev_b = a, b

            print('\n\nInterpreted input as: {}\n\n'.format(res))
            if res == 'before':
                b = cur_t
            elif res == 'after':
                a = cur_t
                ts.extend([st, cur_t, et-1])
            elif res == 'during':
                ts.append(st)
                break
            elif res == 'stop':
                self.stopped = True
                break
            elif res == 'ignore':
                break
            else:
                invalid_input = True
                print('Invalid search query', res)

            cur_t = (a + b) // 2
            st, et = max(0, cur_t - t//2), min(cur_t + t//2, hor)

        for i, t in enumerate(ts[-N:]):
            self.labels.append((res, example[1][t], example[2], example[3], example[-1], len(ts[-N:])-i-1))




    def search_query(self, t=10, N=1, max_iters=20, val=False):
        print('\nRunning search query...\n')
        example = self.get_example(val)
        if example is None:
            self.stopped = True
            return

        assert example[-1] == val
        hor = len(example[0])
        cur_t = hor // 2
        invalid_input = False
        visited = set()
        ts = []
        cur_iter = 0
        wind = 1 # Consider nearby timesteps visited

        a, b = 0, hor
        prev_a, prev_b = -1, -1
        st, et = max(0, cur_t - t//2), min(cur_t + t//2, hor)
        while cur_iter < max_iters and \
              (cur_t not in visited or invalid_input) and \
              (a != prev_a or b != prev_b):

            cur_iter += 1
            invalid_input = False
            cur_t = max(0, min(cur_t, hor))
            visited.update(list(range(cur_t-wind, cur_t+wind+1)))
            res, label_t = self.query_user(example, (st, et))
            prev_a, prev_b = a, b

            print('\n\nInterpreted input as: {}\n\n'.format(res))
            if res == 'before':
                b = cur_t
            elif res == 'after':
                a = cur_t
                ts.extend([st, cur_t, et-1])
            elif res == 'during':
                ts.append(st)
                break
            elif res == 'stop':
                self.stopped = True
                break
            elif res == 'ignore':
                break
            else:
                invalid_input = True
                print('Invalid search query', res)

            cur_t = (a + b) // 2
            st, et = max(0, cur_t - t//2), min(cur_t + t//2, hor)

        for i, t in enumerate(ts[-N:]):
            self.labels.append((res, example[1][t], example[2], example[3], example[-1], len(ts[-N:])-i-1))


    def query_user(self, example, seg_ts):
        choice, ts = self.renderer.interactive_render(example[0][seg_ts[0]:seg_ts[1]])
        choice = self.parse_key(choice)
        return choice, seg_ts[0] + ts


    def parse_key(self, keystroke):
        keystroke = keystroke.lstrip().rstrip()
        if keystroke.lower() in ['b', '1', 'before', 'left']:
            return 'before'

        if keystroke.lower() in ['a', '3', 'after', 'right']:
            return 'after'

        if keystroke.lower() in ['d', '2', 'during', 'space']:
            return 'during'

        if keystroke.lower() in ['s', '0', 'stop', 'q', 'quit']:
            return 'stop'

        if keystroke.lower() in ['u', 'i', '', 'unsure', 'ignore']:
            return 'ignore'

        return 'invalid'


if __name__ == "__main__":
    dname = 'rollouts/'
    labeller = LabelInterface()
    labeller.run(dname)


