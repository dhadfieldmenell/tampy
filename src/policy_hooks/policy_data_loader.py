import gc
import numpy as np
import os
import queue
import time

from policy_hooks.data_buffer import DataBuffer


class DataLoader(object):
    def __init__(self, config, task, in_queue, batch_size, normalize=False, policy=None, x_idx=None, aug_f=None, min_buffer=10**3, feed_in_policy=None, feed_prob=0., feed_inds=(None, None), feed_map=None, save_dir=None):
        self.config = config
        self.in_queue = in_queue
        self.task = task
        self.data_buf = DataBuffer(policy, x_idx=x_idx, normalize=normalize, min_buffer=min_buffer, ratios=config['ratios'])
        self.label_ratio = None
        self.batch_size = batch_size
        self.aug_f = aug_f
        self.load_freq = 10
        self.feed_in_policy = feed_in_policy
        self.feed_prob = feed_prob
        self.feed_out_inds = feed_inds[1]
        self.feed_in_inds = feed_inds[0]
        self.feed_map = feed_map

        self.save_dir = save_dir
        self.cur_save = 0


    def load_from_dir(self, dname):
        self.data_buf.load_from_dir(dname, self.task)


    def write_data(self, n_data):
        self.data_buf.write_data(self.save_dir, self.task, n_data)


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
            self.data_buf.update(data, val_ratio=val_ratio)
        
        del items
        gc.collect()
        return 1


    def get_batch(self, size=None, label=None, val=False):
        if size is None: size = self.batch_size
        batch = self.data_buf.get_batch(size, label, val)
        if batch is None: return [], [], []

        obs, mu, prc, wt, x, primobs, aux = batch
        mu = np.array(mu)
        x = np.array(x)
        if self.feed_prob > 0 and self.feed_in_policy is not None:
            if type(obs) is list:
                obs = np.array(obs)
            else:
                obs = obs.copy()

            nprim = int(self.feed_prob * len(mu))
            hl_out = self.feed_in_policy.act(None, primobs[:nprim], None)
            hl_val = hl_out[:nprim, self.feed_out_inds]
            if self.feed_map is not None: hl_val = self.feed_map(hl_val, x[:nprim])
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
        return obs, mu, prc


    def gen_items(self, label=None, val=False):
        while True: #for _ in range(self.load_freq):
            while self.wait_for_data():
                time.sleep(0.001)
            
            yield self.get_batch()


    def get_size(self, label=None, val=False):
        return self.data_buf.get_size(label, val)


    def gen_load(self):
        while True:
            yield self.load_data()


    def wait_for_data(self):
        cur_n = self.data_buf.get_size()
        if cur_n < self.data_buf.min_buffer:
            self.load_data()
        return cur_n < self.data_buf.min_buffer


