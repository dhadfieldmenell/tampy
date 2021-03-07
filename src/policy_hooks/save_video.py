import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
import sys
from matplotlib.animation import FuncAnimation
import imageio

prefix = os.path.expanduser('~')
SAVEDIR = prefix+'/Dropbox/videos/'
def save_video(fname, dname='', arr=None, savepath=None):
    if savepath is None:
        if not os.path.isdir(SAVEDIR):
            os.mkdir(SAVEDIR)
        if not os.path.isdir(SAVEDIR+dname):
            os.mkdir(SAVEDIR+dname)
        savepath = SAVEDIR+dname
    vname = savepath+fname.split('/')[-1].split('.')[0]+'.gif'
    if os.path.isfile(vname): return
    if arr is None: arr = np.load(fname)
    #print('Saved video', vname)
    imageio.mimsave(vname, arr, duration=0.01)
    '''
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    def update(i):
        print('TS', i)
        ax.imshow(arr[i])
    anim = FuncAnimation(fig, update, frames=range(0, len(arr)), interval=100)
    anim.save(vname, dpi=80, writer='imagemagick')
    '''

if __name__ == '__main__':
    if len(sys.argv) > 1:
        dname = '' if len(sys.argv) <= 2 else '/{0}/'.format(sys.argv[2])
            
        if os.path.isdir(sys.argv[1]):
            pre = '' if 'videos' in sys.argv[1] else 'videos'
            for f in os.listdir(os.path.join(sys.argv[1], pre)):
                if not f.endswith('npy'): continue
                save_video(sys.argv[1]+'{0}/'.format(pre)+f, dname)
        else:
            save_video(sys.argv[1])
