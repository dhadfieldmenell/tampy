import gc
import numpy as np
import os
import queue
import time


#MAX_BUFFER = 300000
MAX_BUFFER = 50000

class DataLoader(object):
    def __init__(self, config, task, in_queue, batch_size, normalize=False, policy=None, x_idx=None, aug_f=None, min_buffer=10**3, feed_in_policy=None, feed_prob=0., feed_inds=(None, None), feed_map=None, save_dir=None):
        self.config = config
        self.in_queue = in_queue
        self.task = task
        self.min_buffer = max(batch_size, min_buffer)
        self.label_ratio = None
        self.normalize = normalize
        self.x_idx = x_idx
        self.batch_size = batch_size
        self.policy = policy
        self.aug_f = aug_f
        self.load_freq = 10
        self.feed_in_policy = feed_in_policy
        self.feed_prob = feed_prob
        self.feed_out_inds = feed_inds[1]
        self.feed_in_inds = feed_inds[0]
        self.feed_map = feed_map

        self.scale, self.bias = None, None
        self.items = {}
        self.val_items = {}

        self.save_dir = save_dir
        self.cur_save = 0


    def load_from_dir(self, dname):
        fnames = os.listdir(dname+'/samples/')
        fnames = list(filter(lambda f: f.find(self.task) >= 0, fnames))
        start_t = time.time()
        if 'optimal' not in self.items:
            self.items['optimal'] = []
            self.val_items['optimal'] = []
        print('LOADING DATA FROM {} FILES'.format(len(fnames)))
        for fname in fnames:
            full_fname = dname+'/samples/'+fname
            data = np.load(full_fname, allow_pickle=True)
            data = data[None][0]
            if fname.find('val') >= 0:
                self.val_items['optimal'].extend(data['optimal'])
            else:
                self.items['optimal'].extend(data['optimal'])
        print('TIME TO LOAD:', time.time() - start_t, len(self.items['optimal']))


    def write_data(self, n_data=None):
        if n_data is not None:
            lab = 'optimal'
            ind = np.random.choice(range(len(self.items[lab])-n_data))
            items = {lab: self.items[lab][ind:ind+n_data]}
            val = 'val' if np.random.uniform() <= 0.1 else 'train'
            np.save('{}/{}_{}data_{}.npy'.format(self.save_dir, self.task, val, self.cur_save), items)
            self.cur_save += 1
        else:
            np.save('{}/{}_traindata.npy'.format(self.save_dir, self.task), self.items)
            np.save('{}/{}_valdata.npy'.format(self.save_dir, self.task), self.val_items)


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
            obs, mu, prc, wt, aux, primobs, x, task, label = data
            for i in range(len(obs)):
                primpt = primobs[i] if len(primobs) else []
                xpt = x[i] if len(x) else []
                auxpt = aux[i] if len(aux) else []
                dct[label].append((obs[i], mu[i], prc[i], wt[i], auxpt, primpt, xpt, task, label))

        for dct in [self.items, self.val_items]:
            labels = list(dct.keys())
            max_size = MAX_BUFFER if dct is self.items else (0.1 * MAX_BUFFER)
            max_size = int(max_size)
            data_len = sum([len(dct[lab]) for lab in labels])
            p = [len(dct[lab])/data_len for lab in labels]
            #if len(dct[label]) > max_size:
            n_del = data_len - max_size
            while n_del > 0:
                del_label = np.random.choice(labels, p=p)
                rand_ind = np.random.randint(len(dct[del_label]))
                try:
                    #del dct[del_label][rand_ind]
                    dct[del_label] = dct[del_label][n_del:]
                    #dct[del_label] = dct[del_label][:rand_ind] + dct[del_label][rand_ind+256:]
                except Exception as e:
                    print(del_label, max_size, e)
                    raise e
                data_len = sum([len(dct[lab]) for lab in labels])
                n_del = data_len - max_size
                p = [len(dct[lab])/data_len for lab in labels]
            gc.collect()
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
        primobs = []
        x = []
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
            primobs.append(dct[lab][ind][5])
            x.append(dct[lab][ind][6])
        if self.normalize or self.aug_f is not None:
            obs = np.array(obs)

        mu = np.array(mu)
        x = np.array(x)
        if self.feed_prob > 0 and self.feed_in_policy is not None:
            if type(obs) is list:
                obs = np.array(obs)
            nprim = int(self.feed_prob * len(mu))
            hl_out = self.feed_in_policy.act(None, primobs[:nprim], None)
            hl_val = hl_out[:nprim, self.feed_out_inds]
            if self.feed_map is not None:
                hl_val = self.feed_map(hl_val, x[:nprim])
            obs[:nprim, self.feed_in_inds] = hl_val

        prc = np.array(prc)
        wt = np.array(wt)
        if len(prc.shape) > 2:
            wt = wt.reshape((-1,1,1))
        else:
            wt = wt.reshape((-1,1))
        aux = np.array(aux)
        if self.aug_f is not None:
            mu, obs, wt, prc = self.aug_f(mu, obs, wt, prc, aux, x)
        prc = wt * prc
        if self.scale is None: self.set_scale()
        if self.normalize:
            obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.scale) + self.bias
        return obs, mu, prc


    def gen_items(self, label=None, val=False):
        while True: #for _ in range(self.load_freq):
            while self.wait_for_data():
                time.sleep(0.001)
            
            yield self.get_batch()


    def set_scale(self):
        if self.scale is not None: return self.scale, self.bias

        if not self.normalize:
            self.scale = np.eye(len(self.x_idx))
            self.bias = np.zeros(len(self.x_idx))
        else:
            obs = []
            for label in self.items:
                for item in self.items[label]:
                    obs.append(item[0])
            obs = np.array(obs)
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

