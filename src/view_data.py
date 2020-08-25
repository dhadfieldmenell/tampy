import h5py
import matplotlib.pyplot as plt
import numpy as np

#f = h5py.File('tf_saved/namo_4/vae_buffer.hdf5')
f = h5py.File('tf_saved/baxterleftblockstackenv_t20_vae_data_4_blocks/vae_buffer.hdf5', 'r')

obs = f['obs_data']

def get_obs(t):
    return np.array(obs[np.random.randint(0, len(obs))][t])
