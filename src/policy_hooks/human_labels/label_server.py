import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from select import select
import time

from policy_hooks.search_node import HLSearchNode
from policy_hooks.server import Server
from policy_hooks.human_labels.video_renderer import VideoRenderer


DIR_KEY = 'experiment_logs/'
HUMAN_PRIORITY = 10

class LabelServer(Server):
    def __init__(self, hyperparams):
        hyperparams['load_render'] = False
        super(LabelServer, self).__init__(hyperparams)
        
        self.failures = []
        self.successes = []
        self.videos = []
        self.labelled = []
        self.labels = []
        self.max_buffer = 10
        self.cur_file = 0
        self.n = 0
        self.n_since_write = 0
        self.in_queue = hyperparams['label_in_queue']
        self.out_queue = hyperparams['task_queue']
        self.renderer = VideoRenderer()
        self.user_response_timeout = 1.
        self.video_renderer = VideoRenderer()
        self.label_dir = DIR_KEY + self.config['weight_dir'] + '/labels/'
        self.save_only = False
        if not os.path.exists(self.label_dir):
            os.mkdir(self.label_dir)


    def n_examples(self, suc=False):
        buf = self.successes if suc else self.failures
        return len(buf)


    def write_labels_to_file(self):
        np.save(self.label_dir+'{}_labels.pkl', self.labels)
        self.labels = []
        self.labelled = []


    def write_to_file(self):
        print('\n\nSaving data to label...\n\n')
        data = []
        buf = self.failures
        for i in range(len(buf[0])):
            data.append(np.array([pt[i] for pt in buf]))

        fname = self.label_dir + 'fail_{}.pkl'.format(self.cur_file)
        vid_fname = self.label_dir + 'fail_vid_{}'.format(self.cur_file)
        with open(fname, 'wb+') as f:
            pickle.dump(data, f)
        #np.save(vid_fname, data[0])

        buf = self.successes
        if len(buf) >= self.max_buffer:
            data = []
            for i in range(len(buf[0])):
                data.append(np.array([pt[i] for pt in buf]))

            fname = self.label_dir + 'suc_{}.pkl'.format(self.cur_file)
            vid_fname = self.label_dir + 'suc_vid_{}'.format(self.cur_file)
            with open(fname, 'wb+') as f:
                pickle.dump(data, f)
            #np.save(vid_fname, data[0])

        self.cur_file += 1
        self.n_since_write = 0
        print('\n\n Saved data to label \n\n', self.n)


    def load_from_directory(self, dname):
        fnames = os.listdir(dname)
        for fname in fnames:
            if fname.find('pkl') < 0: continue
            self.load_from_file(fname)


    def load_from_file(self, fname):
        data = np.load(fname, allow_pickle=True)
        for t in range(len(data[0])):
            if data[-1][t]:
                self.successes.append([val[t] for val in data])
            else:
                self.failures.append([val[t] for val in data])
                self.n += 1
                self.n_since_write += 1

    
    def load_examples(self):
        '''
        Training examples should satisfy:
            example[0] == state trajectory
            example[1] == targets
            example[-1] == success
        '''
        n = self.in_queue.qsize()
        true_n = 0
        for _ in range(n):
            pt = self.pop_queue(self.in_queue)
            if pt is None: continue
            #if not self.render and pt[0] is None: continue
            if pt[-1]:
                self.successes.append(list(pt))
            else:
                self.failures.append(list(pt))
                self.n += 1
                self.n_since_write += 1
            true_n += 1
        n_suc = len(self.successes)
        n_fail = len(self.failures)
        #if n > 0: print('Loaded {} points to label; expected {}; {} {}'.format(true_n, n, n_fail, n_suc))
        if true_n > 0: print('Labelled data:', self.n, self.n_since_write)
        self.successes = self.successes[-self.max_buffer:]
        self.failures = self.failures[-self.max_buffer:]


    def get_example(self, suc=False):
        buf = self.successes if suc else self.failures
        if not len(buf): return None
        cur_iter = 0
        no_motion = True
        while cur_iter < 5 and no_motion and len(buf):
            example = buf.pop(0)
            #ind = np.random.randint(len(buf))
            #example = buf[ind]
            x = example[1]
            no_motion = np.all(np.abs(x[-1]-x[0]) < 0.1)
            cur_iter += 1

        if example[0] is None and self.render:
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
            self.labels.append((res, example[1][label_t], example[2], example[-1]))
            self.labelled.append(((res, label_t), example))
        elif res == 'after':
            self.labels.append((res, example[1][hor], example[2]))
            self.labelled.append(((res, hor), example))
        elif res == 'before':
            self.labels.append((res, example[1][init_t], example[2], example[-1]))
            self.labelled.append(((res, init_t), example))


    def segment_query(self, t=10):
        example = self.get_example()
        hor = len(example[0])
        mid_t = np.random.randint(t//2, hor-et//2+1)
        seg_ts = (mid_t-t//2, mid_t+t//2)
        example = self.get_example()
        res, label_t = self.query_user(example, (st, et))
        self.labelled.append(((res, seg_ts[0]), example))
        self.labels.append((res, example[1][seg_ts[0]], example[2], example[-1]))


    def search_query(self, t=16, N=1, max_iters=20, val=False):
        print('\nRunning search query...\n')
        example = self.get_example(val)
        if example is None:
            #self.stopped = True
            return

        hor = len(example[0])
        cur_t = 0
        invalid_input = False
        ts = []
        cur_iter = 0
        st, et = 0, t
        while st >= 0 and et <= hor:
            cur_iter += 1
            invalid_input = False
            res, label_t = self.query_user(example, (st, et))

            print('\n\nInterpreted input as: {}\n\n'.format(res))
            if res == 'before':
                st -= t // 2
                et -= t // 2
            elif res == 'after':
                ts.extend([(res, st), (res, cur_t), (res, et-1)])
                st += t // 2
                et += t // 2
            elif res == 'during':
                ts.append((res, st))
                ts = [pt for pt in ts if pt[1] <= st]
                break
            elif res == 'stop':
                self.stopped = True
                break
            elif res == 'ignore':
                ts = []
                break
            else:
                invalid_input = True
                print('Invalid search query', res)

        labels = []
        for i, (res, t)  in enumerate(ts[-N:]):
            self.labels.append((res, example[1][t], example[2], example[3], example[-1], len(ts[-N:])-i-1))
            self.labelled.append(((res, t), example))
            labels.append((res, t))

        if self.classify_labels:
            self.send_labels(example, labels)

        if not self.stopped:
            self.renderer.show_transition()


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

        labels = []
        for i, t in enumerate(ts[-N:]):
            self.labels.append((res, example[1][t], example[2], example[3], example[-1], len(ts[-N:])-i-1))
            self.labelled.append(((res, t), example))
            labels.append((res, t))

        if self.classify_labels:
            self.send_labels(example, labels)


    def wait_for_data(self):
        print('No rollouts received yet; please wait')
        while not self.n_examples():
            self.load_examples()
            self.video_renderer.wait()
            time.sleep(0.05)
        self.video_renderer.cont()
        #time.sleep(0.2)
        #self.video_renderer.wait_for_user()


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


    def send_labels(self, pt, labels, wid=40):
        obs = []
        mu = []
        x = []
        ref_pt = len(pt[1])-1
        for (res, ts) in labels:
            if res == 'during':
                ref_pt = ts
        
        if ref_pt < 0: return
        for ts in range(0, min(len(pt[1]), ref_pt+wid)):
            if ts < ref_pt-wid or ts > ref_pt:
                mu.append([1,0])
            else:
                p = np.exp((ts-ref_pt)/10.)
                mu.append([1-p, p])
            obs.append(pt[4][ts])
            x.append(pt[1][ts])

        labels = np.array(mu)
        obs = np.array(obs)
        x = np.array(x)
        self.update_labels(labels, obs, x)


    def push_states(self):
        plan = list(self.agent.plans.values())[0]
        prob = plan.prob
        domain = plan.domain
        for (label, ts), example in self.labelled:
            X = example[1]
            x0 = X[ts]
            targets = example[2]

            initial, goal = self.agent.get_hl_info(x0, targets)
            abs_prob = self.agent.hl_solver.translate_problem(prob, goal=goal, initial=initial)
            hlnode = HLSearchNode(abs_prob,
                                  domain,
                                  prob,
                                  priority=HUMAN_PRIORITY,
                                  x0=x0,
                                  targets=targets,
                                  expansions=0,
                                  label='human_label',
                                  nodetype='human')
            self.push_queue(hlnode, self.task_queue)
        self.clear_labels()


    def clear_labels(self):
        self.labelled = []
        self.labels = []


    def run(self):
        print('\n\n\n\n\n\n\n\n LAUNCHED LABEL SERVER \n\n\n')
        if not self.save_only: self.wait_for_data()
        while True:
            self.wait_for_data()
            self.load_examples()
            self.push_states()
            if not self.save_only:
                self.search_query(N=3, t=12)
            elif self.n_since_write >= self.max_buffer:
                self.write_to_file()
            else:
                time.sleep(0.05)


