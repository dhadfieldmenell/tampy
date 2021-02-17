import numpy as np
import queue
import time


MAX_BUFFER = 200000

class DataLoader(object):
    def __init__(self, config, task, in_queue, batch_size, normalize=False, policy=None, x_idx=None, aug_f=None, min_buffer=10**3):
        self.config = config
        self.in_queue = in_queue
        self.task = task
        self.min_buffer = min_buffer
        self.label_ratio = None
        self.normalize = normalize
        self.x_idx = x_idx
        self.batch_size = batch_size
        self.policy = policy
        self.aug_f = aug_f
        self.load_freq = 10

        self.scale, self.bias = None, None
        self.items = {}
        self.val_items = {}


    def pop_queue(self):
        items = []
        n = self.in_queue.qsize()
        for _ in range(n):
            try:
                data = self.in_queue.get_nowait()
                items.append(data)
            except queue.Empty:
                break

        return items


    def load_data(self, val_ratio=0.1):
        items = self.pop_queue()
        if not len(items): return 0

        start_t = time.time()
        for data in items:
            start_t = time.time()
            val = np.random.uniform() < val_ratio
            dct = self.items if not val else self.val_items
            label = data[-1]
            if label not in dct: dct[label] = []
            obs, mu, prc, wt, aux, task, label = data
            for i in range(len(obs)):
                if len(aux):
                    dct[label].append((obs[i], mu[i], prc[i], wt[i], aux[i], task, label))
                else:
                    dct[label].append((obs[i], mu[i], prc[i], wt[i], [], task, label))

            max_size = MAX_BUFFER if not val else (0.1 * MAX_BUFFER)
            max_size = int(max_size)
            if len(dct[label]) > max_size:
                try:
                    dct[label] = dct[label][-max_size:]
                except Exception as e:
                    print(label, max_size, e)
                    raise e

        return 1


    def get_batch(self, size=None, label=None, val=False):
        if size is None: size = self.batch_size
        start_t = time.time()
        dct = self.val_items if val else self.items
        
        if label is not None and label not in dct: return [], [], []
        labels = list(dct.keys()) if label is None else [label]
        data_len = sum([len(dct[lab]) for lab in labels])
        if data_len < size: return [], [], []
        
        p = [len(dct[lab])/data_len for lab in labels]
        n_per = size // len(labels) + 1
        obs, mu, prc, wt = [], [], [], []
        aux = []
        used = []
        n = 0
        true_n = 0
        while n < size:
            n += 1
            true_n += 1
            lab = np.random.choice(labels, p=p)
            if not len(dct[lab]):
                n -= 1
                continue
            ind = np.random.randint(len(dct[lab]))
            #if not val and (lab, ind) in used:
            #    n -= 1
            #    continue
            used.append((lab,ind))
            obs.append(dct[lab][ind][0])
            mu.append(dct[lab][ind][1])
            prc.append(dct[lab][ind][2])
            wt.append(dct[lab][ind][3])
            aux.append(dct[lab][ind][4])
        #if self.task == 'primitive': print('Time to collect tensor:', time.time() - start_t)
        if self.normalize or self.aug_f is not None:
            obs = np.array(obs)
        mu = np.array(mu)
        prc = np.array(prc)
        wt = np.array(wt)
        if len(prc.shape) > 2:
            wt = wt.reshape((-1,1,1))
        else:
            wt = wt.reshape((-1,1))
        aux = np.array(aux)
        if self.aug_f is not None:
            mu, obs, wt, prc = self.aug_f(mu, obs, wt, prc, aux)
        prc = wt * prc
        #if self.task == 'primitive': print('Time to build tensor:', time.time() - start_t)
        if self.scale is None: self.set_scale()
        if self.normalize:
            obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.scale) + self.bias
        #if self.task == 'primitive': print('Time to get', len(obs), 'batch:', time.time() - start_t)
        return obs, mu, prc


    def gen_items(self, label=None, val=False):
        for _ in range(self.load_freq):
            #print('Waiting for batch to train on', self.task)
            while self.wait_for_data():
                time.sleep(0.001)
            
            #print('Sending batch to train on', self.task)
            yield self.get_batch()


    def set_scale(self):
        if self.scale is not None: return self.scale, self.bias
        obs = []
        for label in self.items:
            for item in self.items[label]:
                obs.append(item[0])
        obs = np.array(obs)

        if not self.normalize:
            self.scale = np.eye(len(self.x_idx))
            self.bias = np.zeros(len(self.x_idx))
        else:
            self.scale = np.diag(1.0 / np.maximum(np.std(obs[:, self.x_idx], axis=0), 1e-1))
            self.bias = -np.mean(obs[:, self.x_idx].dot(self.scale), axis=0)
        self.policy.scale = self.scale
        self.policy.bias = self.bias
        return self.scale, self.bias


    def get_size(self, label=None, val=False):
        dct = self.items if not val else self.val_items
        labels = [label] if label is not None else list(dct.keys())
        return sum([len(dct[lab]) for lab in labels])


    def gen_load(self):
        while True:
            yield self.load_data()


    def wait_for_data(self):
        data = list(self.items.values())
        data_len = sum([len(d) for d in data]) if len(data) else 0
        if data_len < self.min_buffer:
            self.load_data()
        return data_len < self.min_buffer

    
    def set_labels(self, labels):
        for lab in labels:
            if lab not in self.items:
                self.items[lab] = []
                self.val_items[lab] = []

    
    def get_labels(self):
        return list(self.items.keys())


    def count_labels(self):
        return {key: len(data) for key, data in self.items.items()}

