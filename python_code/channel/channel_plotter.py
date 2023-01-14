import matplotlib as mpl
import matplotlib.pyplot as plt

from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.channel.channels_hyperparams import N_ANT
from python_code.utils.config_singleton import Config

conf = Config()

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['font.size'] = 8
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [9.5, 6.45]
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

colors = ['red','blue','green','orange']

def plot_channel_by_phase():
    """
    Plot for one given antenna at a time
    :param phase: Enum
    """
    channel_dataset = ChannelModelDataset(block_length=conf.block_length,
                                          pilots_length=conf.pilot_size,
                                          blocks_num=conf.blocks_num)
    transmitted_words, received_words, hs = channel_dataset.__getitem__(snr_list=[conf.snr])
    hs_array = hs.squeeze(1).cpu().numpy()
    plt.figure()
    for j in range(N_ANT):
        plt.plot(hs_array[:, j], label=f'Tap {j}',color=colors[j])
        plt.ylabel(r'magnitude', fontsize=20)
        plt.xlabel(r'block index', fontsize=20)
        plt.ylim([-0.1, 1.1])
        plt.grid(True, which='both')
        plt.legend(loc='upper left', prop={'size': 25})
    plt.show()


if __name__ == "__main__":
    plot_channel_by_phase()
