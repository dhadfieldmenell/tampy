import matplotlib.pyplot as plt
import numpy as np
from select import select
import time

from policy_hooks.server import Server
from policy_hooks.human_labels.video_renderer import VideoRenderer


class LabelServer(Server):
    def __init__(self, hyperparams):
        hyperparams['load_render'] = False
        super(LabelServer, self).__init__(hyperparams)
        
        self.failures = []
        self.successes = []
        self.videos = []
        self.labelled = []
        self.max_buffer = 10
        self.in_queue = hyperparams['label_in_queue']
        self.out_queue = hyperparams['task_queue']
        self.renderer = VideoRenderer()
        self.user_response_timeout = 1.
        self.video_renderer = VideoRenderer()


    def n_examples(self, suc=False):
        buf = self.successes if suc else self.failures
        return len(buf)

    
    def load_examples(self):
        '''
        Training examples should satisfy:
            example[0] == state trajectory
            example[1] == targets
            example[-1] == success
        '''
        n = self.in_queue.qsize()
        for _ in range(n):
            pt = self.pop_queue(self.in_queue)
            if pt is None: continue
            if pt[-1]:
                self.successes.append([None,] + list(pt))
            else:
                self.failures.append([None,] + list(pt))
        self.successes = self.successes[-self.max_buffer:]
        self.failures = self.failures[-self.max_buffer:]


    def get_example(self, suc=False):
        buf = self.successes if suc else self.failures
        if not len(buf): return None
        ind = np.random.randint(len(buf))
        example = buf[ind]

        if example[0] is None:
            example[0] = self.gen_video(example)

        return example


    def gen_video(self, example, st=0, n=-1):
        buf = []
        xs = example[1]
        self.agent.target_vecs[0] = example[2]
        if n < 0: n = len(xs) - st
        self.agent.render_context.make_context_current()
        for t in range(st, st+n):
            buf.append(self.agent.get_image(xs[t]))
        return buf


    def ts_query(self, min_t=10):
        example = self.get_example()
        hor = len(example[0])
        if hor <= min_t: return

        init_t = np.random.randint(0, hor-min_t)
        res, label_t = self.query_user(example, (init_t, hor))

        if res == 'stop' or res == 'during':
            self.labelled.append(((res, label_t), example))
        elif res == 'after':
            self.labelled.append(((res, hor), example))
        elif res == 'before':
            self.labelled.append(((res, init_t), example))


    def segment_query(self, t=10):
        example = self.get_example()
        hor = len(example[0])
        mid_t = np.random.randint(t//2, hor-et//2+1)
        seg_ts = (mid_t-t//2, mid_t+t//2)
        example = self.get_example()
        res, label_t = self.query_user(example, (st, et))
        self.labelled.append(((res, seg_ts[0]), example))

    
    def search_query(self, t=10):
        print('\nRunning search query...\n')
        example = self.get_example()
        hor = len(example[0])
        cur_t = hor // 2
        prev_t = -1
        invalid_input = False
        while prev_t != cur_t or invalid_input:
            invalid_input = False
            prev_t = cur_t
            cur_t = max(0, min(cur_t, hor))
            st, et = max(0, cur_t - t//2), min(cur_t + t//2, hor)
            res, label_t = self.query_user(example, (st, et))

            if res == 'before':
                cur_t = st + (cur_t - st) // 2
            elif res == 'after':
                cur_t = cur_t + (et - cur_t) // 2
            elif res == 'during':
                break
            else:
                invalid_input = True
                print('Invalid search query')

        self.labelled.append(((res, st), example))


    def query_user(self, example, seg_ts, val=False):
        suc = False if val else np.random.uniform() < 0.5
        choice, ts = self.renderer.interactive_render(example[0][seg_ts[0]:seg_ts[1]])
        #self.renderer.threaded_render(example[0][seg_ts[0]:seg_ts[1]])
        #while True:
        #    user_input, _, _ = select([sys.stdin], [], [], self.user_response_timeout)
        #    if user_input:
        #        self.renderer.stop()
        #        choice = sys.stdin.readline().lstrip().rstrip()
        #        break
        #ts = self.renderer.get_time()
        #choice = self.parse_key(choice)
        return choice, seg_ts[0] + ts


    def wait_for_data(self):
        print('No rollouts received yet; please wait')
        while not self.n_examples():
            self.load_examples()
            self.video_renderer.wait()
            time.sleep(0.01)
        self.video_renderer.cont()
        time.sleep(1)
        self.video_renderer.wait_for_user()


    def parse_key(self, keystroke):
        keystroke = keystroke.lstrip().rstrip()
        if keystroke.lower() in ['b', '1', 'before']:
            return 'before'

        if keystroke.lower() in ['a', '3', 'after']:
            return 'after'

        if keystroke.lower() in ['d', '2', 'during']:
            return 'during'

        if keystroke.lower() in ['s', '0', 'stop']:
            return 'stop'

        if keystroke.lower() in ['u', 'i', '', 'unsure', 'ignore']:
            return 'ignore'

        return 'invalid'


    def run(self):
        print('\n\n\n\n\n\n\n\n LAUNCHED LABEL SERVER \n\n\n')
        self.wait_for_data()
        while True:
            self.load_examples()
            self.search_query()


