from torch.utils.tensorboard import SummaryWriter
import shutil
import random
import numpy as np

import threading
import matplotlib.pyplot as plt

def rm_plot(name):
    shutil.rmtree('data/logs/' + name, ignore_errors = True)

def get_writer(name, rmifexist = False):
    if rmifexist: rm_plot(name)
    return SummaryWriter('data/logs/' + name)

def get_plotter(name, scalar_name = None,
                printalso = False, rmifexist = False):
    writer = get_writer(name, rmifexist)
    scalar_name = scalar_name or name
    def plot(x, y):
        writer.add_scalar(scalar_name, y, global_step = x)
        if printalso:
            print(x, y)
    return plot

def get_histogram_plotter(name, scalar_name = None,
                          printalso = False, rmifexist = False):
    # according to https://stackoverflow.com/a/38481873/11815215, will plot percentiles: [maximum, 93%, 84%, 69%, 50%, 31%, 16%, 7%, minimum]
    
    writer = get_writer(name, rmifexist)
    scalar_name = scalar_name or name
    def plot(x, y):
        writer.add_histogram(scalar_name, y, global_step = x)
        if printalso:
            print(x, y)
    return plot

def rand_params(Env, params='all'):
    if params == 'all':
        params = {k: v for k, v in Env.parameters_spec.items() if isinstance(v, list)}  # only for params with randomization in a range, others like observation delay needs to be handled in other ways
    value = [random.uniform(*Env.parameters_spec[param]) for param in params]  # loop through its keys and randomly sample values
    dic = {p: v for p, v in zip(params, value)}
    # print('Randomized parameters value: ', dic)
    return dic, np.array(value, dtype = np.float32)

def average_of_params(Env, params):
    value = [sum(Env.parameters_spec[param]) / 2 for param in params]
    return value

def enum_params(Env, params):
    num_each = [None, 100, 10, 5, 4, 3][len(params)]
    for i in range(num_each ** len(params)):
        pstate = [i // (num_each ** j) % num_each for j in range(len(params))]
        pstate = [np.linspace(*Env.parameters_spec[params[j]], num=num_each)[s]
                  for j, s in enumerate(pstate)]
        yield pstate

def clip_params(Env, params, p):
    p = p.reshape(len(params))
    a_min = np.array([Env.parameters_spec[param][0] - 0.1 for param in params], dtype=np.float32)
    a_max = np.array([Env.parameters_spec[param][1] + 0.1 for param in params], dtype=np.float32)
    return np.clip(p, a_min, a_max)


class Drawer:
    def __init__(self, title='Tactile Sensors'):
        self.update_plot = threading.Event()
        self.update_plot.set()
        self.stopped = False
        self.values = []
        self.title = title

    def create_thread(self):
        self.thread = threading.Thread(target=self.plot)
        self.thread.daemon = True
        self.thread.start()

    def start(self,):
        fig, ax = plt.subplots(figsize=(4,4))
        ax.title.set_text(self.title)
        # plt.show()


    def add_value(self, v):
        self.values = v
        self.update_plot.set()

    def stop(self):
        self.stopped = True
        self.update_plot.set()
        plt.ioff()

    def render(self, ax=plt):
        plt.cla()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(self.title)
        tac_sig = self.values
        # print(tac_sig)
        def split_sig(tac_sig, pins=15):
            left_sig = [tac_sig[6*i:6*i+3] for i in range(pins//3)]
            right_sig = [tac_sig[6*i+3:6*i+6] for i in range(pins//3)]
            return np.array(left_sig).reshape(-1), np.array(right_sig).reshape(-1)

        left_sig, right_sig = split_sig(tac_sig)

        # locations of each pin (x,y) acoording to their orders in reality, unit: mm
        LOCS = [[5, 3], [9, 0], [13, 3],
                [5, 9], [9, 6], [13, 9],
                [5, 15], [9, 12], [13, 15],
                [5, 21], [9, 18], [13, 21],
                [5, 27], [9, 24], [13, 27],
            ]
        right_pad_offset=30
        for i, loc in enumerate(LOCS):
            # left pad
            if left_sig[i]:
                c='red'
            else:
                c='grey'
            ax.scatter(*loc, s=500, marker='.', c=c)
            
            # right pad
            loc[0]+=right_pad_offset
            if right_sig[i]:
                c='red'
            else:
                c='grey'
            ax.scatter(*loc, s=500, marker='.', c=c)

        plt.pause(0.01)  # sleep of each step
        # plt.draw()

    def plot(self):
        plt.ion()
        while not self.stopped:
            self.update_plot.wait()
            self.update_plot.clear()
            self.render()
            plt.pause(0.2)
        plt.ioff()
        plt.close()

