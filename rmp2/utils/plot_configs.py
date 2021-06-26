# The first version was licensed as "Original Source License"(see below).
# Several enhancements and at UW Robot Learning Lab
# 
# Original Source License:
# 
# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

"""
configs for ploting
"""

from matplotlib import cm
from itertools import chain

SET2COLORS = cm.get_cmap('Set2').colors
SET2 = {'darkgreen': SET2COLORS[0],
        'orange': SET2COLORS[1],
        'blue': SET2COLORS[2],
        'pink': SET2COLORS[3],
        'lightgreen': SET2COLORS[4],
        'gold': SET2COLORS[5],
        'brown': SET2COLORS[6],
        'grey': SET2COLORS[7],
        }

SET1COLORS = cm.get_cmap('Set1').colors
SET1 = {
    'red': SET1COLORS[0],
    'blue': SET1COLORS[1],
    'green': SET1COLORS[2],
    'purple': SET1COLORS[3],
    'orange': SET1COLORS[4],
    'yellow': SET1COLORS[5],
    'brown': SET1COLORS[6],
    'pink': SET1COLORS[7],
    'grey': SET1COLORS[8]
}
code_configs = {
    'bc-nn': (r'\textsc{BC+NN}', SET1['blue']),
    'bc-rmp': (r'\textsc{BC+RMP}', SET1['purple']),
    'code-nn': (r'\textsc{CODE+NN}', SET1['green']),
    'code-rmp': (r'\textsc{CODE+RMP}', SET2['lightgreen']),
    'order': [
        'bc-nn', 'bc-rmp', 'code-nn', 'code-rmp']
}

rmp2_configs = {
    'rmp': (r'\textsc{RMP}', SET2['lightgreen']),
    'rmp-obs-feat': (r'\textsc{RMP-RESIDUAL}', SET1['blue']),
    'nn': (r'\textsc{NN}', 'gray'), # SET1['grey']),
    'nn-residual': (r'\textsc{NN-RESIDUAL}', 'indianred'), # SET1['red']),
    'order': [
        'rmp-obs-feat', 'rmp', 'nn-residual', 'nn']
}

gtc_configs = {
    'rmp-obs-feat': (r'\textsc{STRUCTURED}', [0.4627451, 0.7254902, 0.]),
    'nn': (r'\textsc{NN}', 'gray'), # SET1['grey']),
    'order': [
        'rmp-obs-feat', 'nn']
}

class Configs(object):
    def __init__(self, style=None, colormap=None):
        if not style:
            self.configs = None
            if colormap is None: 
                c1 = iter(cm.get_cmap('Set1').colors)
                c2 = iter(cm.get_cmap('Set2').colors)
                c3 = iter(cm.get_cmap('Set3').colors)
                self.colors = chain(c1, c2, c3)
            else:
                self.colors = iter(cm.get_cmap(colormap).colors)
        else:
            self.configs = globals()[style + '_configs']
            for exp_name in self.configs['order']:
                assert exp_name in self.configs, 'Unknown exp: {}'.format(exp_name)

    def color(self, exp_name):
        if self.configs is None:
            color = next(self.colors)
        else:
            color = self.configs[exp_name][1]
        return color

    def label(self, exp_name):
        if self.configs is None:
            return exp_name
        return self.configs[exp_name][0]

    def sort_dirs(self, dirs):
        if self.configs is None:
            return dirs

        def custom_key(exp_name):
            if exp_name in self.configs['order']:
                return self.configs['order'].index(exp_name)
            else:
                return 100
        return sorted(dirs, key=custom_key)
