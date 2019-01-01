"""
Forked from matplotlib tutorial
https://matplotlib.org/gallery/statistics/barchart_demo.html
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

# TITLE = 'MAE rates by Neural Nets parameters'
# EXPERIMENT_DISCRIPTIONS = ('l_r=0.02\nepoch=20\n30/relu\n70/30-sv',
#                            'l_r=0.20\nepoch=20\n50/relu\n70/30-sv',
#                            'l_r=0.20\nepoch=20\n50/relu\n3-fold-cv',
#                            'l_r=0.20\nepoch=20\n50/relu\n20-fold-cv',
#                            'l_r=0.20\nepoch=50\n50/relu\n20-fold-cv',
#                            'l_r=0.20\nepoch=200\n50/relu\n20-fold-cv',
#                            'l_r=0.15\nepoch=200\n50/relu-20/relu\n20-fold-cv',
#                            )
# means_experiment = (56.22, 42.51, 43.25, 40.85, 33.16, 28.43, 26.44)
# std_experiment = (3.25, 3.4, 1.13, 0.82, 0.34, 0.51, 0.27)


# TITLE = 'MAE rates by Neural Nets parameters'
# EXPERIMENT_DISCRIPTIONS = ('l_r=0.20\nepoch=200\n30-20 (relu)\n20-fold-cv',
#                            'l_r=0.20\nepoch=200\n50-30 (relu)\n20-fold-cv',
#                            'l_r=0.20\nepoch=200\n50-50 (relu)\n20-fold-cv',
#                            'l_r=0.20\nepoch=300\n50-30 (relu)\n20-fold-cv',
#                            'l_r=0.20\nepoch=500\n50-30 (relu)\n20-fold-cv',
#                            'l_r=0.20\nepoch=1000\n50-30 (relu)\n20-fold-cv',
#                            'l_r=0.15\nepoch=2000\n50-30 (relu)\n20-fold-cv',
#                            )
# means_experiment = (3.64, 3.29, 2.92, 2.51, 1.77, 1.29, 0.94)
# std_experiment = (0.17, 0.25, 0.22, 0.16, 0.10, 0.24, 0.02)


# TITLE = 'Velocity MAE rates by Neural Nets parameters before postprocessing'
# EXPERIMENT_DISCRIPTIONS = ('l_r=0.02\nepoch=20\n30/relu\n70/30-sv',
#                            'l_r=0.20\nepoch=20\n50/relu\n70/30-sv',
#                            'l_r=0.20\nepoch=20\n50/relu\n3-fold-cv',
#                            'l_r=0.20\nepoch=20\n50/relu\n20-fold-cv',
#                            'l_r=0.20\nepoch=50\n50/relu\n20-fold-cv',
#                            'l_r=0.20\nepoch=200\n50/relu\n20-fold-cv',
#                            'l_r=0.15\nepoch=200\n50/relu-20/relu\n20-fold-cv',
#                            )
# means_experiment = (0.0561, 0.0422, 0.0432, 0.0408 , 0.0331, 0.0284, 0.0264)
# std_experiment = (0.00326, 0.00348, 0.00113, 0.00082, 0.00036, 0.00050, 0.00028)


TITLE = 'Velocity MAE rates by Neural Nets parameters before postprocessing'
EXPERIMENT_DISCRIPTIONS = ('l_r=0.20\nepoch=200\n30-20 (relu)\n20-fold-cv',
                           'l_r=0.20\nepoch=200\n50-30 (relu)\n20-fold-cv',
                           'l_r=0.20\nepoch=200\n50-50 (relu)\n20-fold-cv',
                           'l_r=0.20\nepoch=300\n50-30 (relu)\n20-fold-cv',
                           'l_r=0.20\nepoch=500\n50-30 (relu)\n20-fold-cv',
                           'l_r=0.20\nepoch=1000\n50-30 (relu)\n20-fold-cv',
                           'l_r=0.15\nepoch=2000\n50-30 (relu)\n20-fold-cv',
                           )
means_experiment = (0.00364, 0.00330, 0.00292, 0.00251, 0.00177, 0.00128, 0.00095)
std_experiment = (0.000171, 0.000246, 0.000221, 0.000166, 0.000109, 0.00024, 0.00002)


n_groups = len(means_experiment) # 5


# means_women = (25, 32, 34, 20, 25)
# std_women = (3, 5, 2, 3, 3)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, means_experiment, bar_width,
                alpha=opacity, color='b',
                yerr=std_experiment, error_kw=error_config,
                label='Error rate')

# rects2 = ax.bar(index + bar_width, means_women, bar_width,
#                 alpha=opacity, color='r',
#                 yerr=std_women, error_kw=error_config,
#                 label='Women')

ax.set_xlabel('Experiments')
ax.set_ylabel('Error rates')
ax.set_title(TITLE)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(EXPERIMENT_DISCRIPTIONS)
ax.legend()

fig.tight_layout()
plt.show()

