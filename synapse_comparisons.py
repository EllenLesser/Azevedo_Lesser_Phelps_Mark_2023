import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

from constants import DATA_DIR
from neuron_comparisons import NeuronComparator


def plot_synapse_number_comparisons(neurons: list[NeuronComparator]):
    """Plot the number of synapses in each neuron, both manual and auto.
    :param neurons: list of NeuronComparator objects
    """
    hfont = {'fontname': 'Arial'}
    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(figsize=[2, 6])

    for neuron in neurons:
        if not hasattr(neuron, 'predicted_synapses'):
            neuron.get_post_synapses_predicted()
        if not hasattr(neuron, 'ground_truth_synapses'):
            neuron.get_post_synapses_ground_truth()

        plt.scatter(1, neuron.ground_truth_synapses.shape[0], c=neuron.color, s=200, alpha=1, edgecolors='k')
        plt.scatter(2, neuron.predicted_synapses.shape[0], c=neuron.color, s=200, alpha=1, edgecolors='k')
        plt.plot([1.93, 1.075], [neuron.predicted_synapses.shape[0], neuron.ground_truth_synapses.shape[0]], lw=3,
                 c='k')

    plt.xticks([1, 2], ['Manual', 'Auto'], **hfont)
    plt.yticks([1000, 3000, 5000], rotation=90, **hfont)
    plt.ylabel('Number of input synapses', **hfont)
    plt.savefig(DATA_DIR / 'synapse_count_comparisons.svg', transparent=True)


def plot_synapse_distributions(neuron: NeuronComparator):
    """Plot 1-D distributions of synapse locations along each axis.
    :param neuron: NeuronComparator object
    """

    if not hasattr(neuron, 'predicted_synapses'):
        neuron.get_post_synapses_predicted()
    if not hasattr(neuron, 'ground_truth_synapses'):
        neuron.get_post_synapses_ground_truth()

    ground_truth_synapses = neuron.get_synapse_array(neuron.ground_truth_synapses['post_pt_position_v4'])
    predicted_synapses = neuron.get_synapse_array(neuron.predicted_synapses['post_pt_position'])

    axis_lookup = {'DV': 2, 'AP': 1, 'ML': 0}

    for axis_name, dim in axis_lookup.items():
        plt.figure(figsize=[8, 5])
        sns.displot(ground_truth_synapses[:, dim] / 1000, color='m', vertical=False)
        sns.displot(predicted_synapses[:, dim] / 1000, color='g', vertical=False)
        plt.xlabel(f'{axis_name} distribution')
        plt.xticks([])
        plt.savefig(DATA_DIR / f'{axis_name}_distribution.svg')


def compare_closest_synapses(neurons: list[NeuronComparator]):
    """Plot the distance between the closest manually annotated synapse and the closest predicted synapse and vice versa.
    :param neurons: list of NeuronComparator objects
    """
    plt.figure(figsize=[4, 12])

    for neuron in neurons:
        if not hasattr(neuron, 'predicted_synapses'):
            neuron.get_post_synapses_predicted()
        if not hasattr(neuron, 'ground_truth_synapses'):
            neuron.get_post_synapses_ground_truth()

        ground_truth_synapses = neuron.get_synapse_array(neuron.ground_truth_synapses['post_pt_position_v4'])
        predicted_synapses = neuron.get_synapse_array(neuron.predicted_synapses['post_pt_position'])

        distances = cdist(predicted_synapses * np.array([4.3, 4.3, 45]) / 1000,
                          ground_truth_synapses * np.array([4.3, 4.3, 45]) / 1000)
        manual_min_distances = distances.min(axis=0)
        predicted_min_distances = distances.min(axis=1)
        plt.scatter(1, np.mean(manual_min_distances), c=neuron.color, s=500, alpha=.75,
                    edgecolors=np.array(neuron.color) * .1)
        plt.scatter(2, np.mean(predicted_min_distances), c=neuron.color, s=500, alpha=.75,
                    edgecolors=np.array(neuron.color) * .1)

    
    plt.xlim([0.5, 2.5])
    plt.xticks([1, 2], ['Manual-Predicted', 'Predicted-Manual'], rotation=90)
    plt.ylabel('Distance to closest synapse (µm)')
    plt.ylim([0, 5])
    plt.savefig(DATA_DIR / 'synapse_distances.svg')


