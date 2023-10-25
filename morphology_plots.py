import json
from tqdm import tqdm

from constants import NEURON_METADATA_PATH
from neuron_comparisons import NeuronComparator
from plotting_utils import mesh_neuron_plot, skel_neuron_plot

RIGHT_T1_CENTER = (95184.48025184708, 119185.49864607782, 384689.7268014647)
LEFT_T1_CENTER = (174034.11026983315, 119310.89525140286, 336026.3672285372)


def generate_morphology_plots(neurons: list[NeuronComparator]):
    """Generate mesh and skeleton plots for each neuron in the list.
    :param neurons: list of NeuronComparator objects"""

    for neuron in tqdm(neurons, 'Generating morphology plots'):
        mesh_neuron_plot(neuron)
        skel_neuron_plot(neuron)

