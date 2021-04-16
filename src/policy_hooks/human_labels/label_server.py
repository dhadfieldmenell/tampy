from policy_hooks.server import Server


class LabelServer(Server):
    def __init__(self, hyperparams):
        super(LabelServer, self).__init__(hyperparams)
        self.failures = []
        self.successes = []
        self.videos = []
        self.labelled = []
        self.max_buffer = 10
        self.in_queue = hyperparams['label_queue']
        self.out_queue = hyperparams['task_queue']
        self.renderer = VideoRenderer()
        self.user_response_timeout = 1.

    
    def load_examples(self):
        '''
        Training examples should satisfy:
            example[0] == state
            example[1] == targets
            example[-1] == success
        '''
        n = self.in_queue.qsize()
        for _ in range(n):
            pt = self.pop_queue(self.in_queue)
            if pt[-1]:
                self.successes.append([None,] + list(pt))
            else:
                self.failures.append([None,] + list(pt))
        self.successes = self.successes[-self.max_buffer:]
        self.failures = self.failures[-self.max_buffer:]


    def get_example(self, suc=False):
        if suc:
            example = np.random.choice(self.successes)
        else:
            example = np.random.choice(self.failures)

        if example[0] is None:
            example[0] = self.gen_video(example)

        return example


    def gen_video(self, example, st=0, n=-1):
        buf = []
        xs = example[1]
        self.agent.target_vecs[0] = example[2]
        if n < 0: n = len(xs) - st
        for t in range(st, st+n):
            buf.append(self.agent.get_image(xs[t]))
        return buf


    def query_user(self, val=False):
        suc = False if val else np.random.uniform() < 0.5
        example = self.get_example(suc=suc)
        self.renderer.threaded_render(example[0])
        while True:
            user_input, _, _ = select([sys.stdin], [], [], self.user_response_timeout)
            if user_input:
                self.renderer.stop()
                choice = sys.stdin.readline().lstrip().rstrip()
                break
        ts = self.renderer.get_time()
        self.labelled.append(((choice, ts), example))


    def run(self):
        while True:
            self.load_examples()



