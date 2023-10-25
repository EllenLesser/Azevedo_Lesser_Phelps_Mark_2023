import json
import numpy as np

from constants import NEURON_METADATA_PATH
from connectivity_comparisons import plot_connectivity_metrics
from morphology_plots import generate_morphology_plots
from neuron_comparisons import NeuronComparator
from synapse_comparisons import compare_closest_synapses, plot_synapse_distributions, plot_synapse_number_comparisons

# SET UP AND LOAD DATA
with open(NEURON_METADATA_PATH, 'r') as f:
    neuron_metadata = json.load(f)
neuron_comparisons = [NeuronComparator(neuron_meta) for neuron_meta in neuron_metadata]


# MORPHOLOGY PLOTS FOR Figure 2A
generate_morphology_plots(neuron_comparisons)


# SYNAPSE COMPARISONS FOR Figure 2E, S1D, and S1E
# Figure 2E
plot_synapse_number_comparisons(neuron_comparisons)
# Figure S1D
neuron_example = [neuron for neuron in neuron_comparisons if neuron.name == 'sternal_poserior_rotator_r'][0]
plot_synapse_distributions(neuron_example)
# Figure S1E
compare_closest_synapses(neuron_comparisons)

# CONNECTIVITY COMPARISONS FOR FIGURE 2F and 2G
K_MEASURES = np.arange(1, 50, 3)
SYNAPSE_THRESHOLD = 5

plot_connectivity_metrics(neuron_comparisons, K_MEASURES, SYNAPSE_THRESHOLD)

if __name__ == '__main__':
    pass
