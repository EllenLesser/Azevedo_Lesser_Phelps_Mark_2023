from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from constants import DATA_DIR
from neuron_comparisons import NeuronComparator

matplotlib.use('Agg')

def calculate_connectivity_metrics(neuron: NeuronComparator,
                                   k_measures: np.ndarray,
                                   synapse_threshold: int = 5) -> dict[str, list]:
    """Calculate metrics for connectivity comparisons. Currently only precision and recall at k are implemented.
    :param neuron: NeuronComparator object
    :param k_measures: array of k values to calculate precision and recall at
    :param synapse_threshold: minimum number of synapses to consider a partner
    :return: dictionary of metrics
    """

    precisions = []
    recalls = []

    if not hasattr(neuron, 'predicted_synapses'):
        neuron.get_post_synapses_predicted()
    if not hasattr(neuron, 'ground_truth_synapses'):
        neuron.get_post_synapses_ground_truth()

    actual_counts = neuron.ground_truth_synapses.pre_pt_root_id.value_counts()
    predicted_counts = neuron.predicted_synapses.pre_pt_root_id.value_counts()
    actual_counts = actual_counts.loc[actual_counts >= synapse_threshold]
    predicted_counts = predicted_counts.loc[predicted_counts >= synapse_threshold]
    print('actual = {}, predicted = {}'.format(len(actual_counts), len(predicted_counts)))

    for k in k_measures:
        precision = precision_at_k(actual_counts, predicted_counts, k)
        if np.isnan(precision):
            break
        else:
            precisions.append(precision)


    for k in k_measures:
        recall = recall_at_k(actual_counts, predicted_counts, k)
        if np.isnan(recall):
            break
        else:
            recalls.append(recall)

    return {'precision': precisions, 'recall': recalls}


def recall_at_k(actual_counts: pd.Series, predicted_counts: pd.Series, k: int) -> float:
    """Calculate recall at k for a given neuron. Defined as the fraction of top k ground truth partners that are predicted
    partners.
    :param actual_counts: A series of ground truth counts for each partner.
    :param predicted_counts: A series of predicted counts for each partner.
    :param k: The number of top partners to consider.
    :return: The recall at k.
    """
    if k>len(actual_counts):
        print("recall done at {} actual_counts".format(k))
        return np.nan

    # if actual_counts.iloc[k]<=3 and actual_counts.iloc[k-1]>3:
    #     print("{} is first actual connection at threshold".format(k))

    return sum(np.isin(actual_counts.index[0:k], predicted_counts.index)) / k


def precision_at_k(actual_counts: pd.Series, predicted_counts: pd.Series, k) -> float:
    """Calculate precision at k for a given neuron. Defined as the fraction of top k predicted partners that are ground
       truth partners.
       :param actual_counts: A series of ground truth counts for each partner.
       :param predicted_counts: A series of predicted counts for each partner.
       :param k: The number of top partners to consider.
       :return: The precision at k.
       """

    if k>len(predicted_counts):
        print("precision done at {} predicted counts".format(k))
        return np.nan
    
    # if predicted_counts.iloc[k]<=3 and predicted_counts.iloc[k-1]>3:
    #     print("{} is first predicted connection at threshold".format(k))
        
    return sum(np.isin(predicted_counts.index[0:k], actual_counts.index)) / k


def plot_connectivity_metrics(neurons: list[NeuronComparator],
                              k_measures: np.ndarray = np.arange(1, 100, 1),
                              synapse_threshold: int = 5):

    """Plot connectivity metrics for a list of neurons.
    :param neurons: A list of NeuronComparator objects.
    :param k_measures: array of k values to calculate precision and recall at
    :param synapse_threshold: minimum number of synapses to consider a partner
    """

    metrics = defaultdict(list)
    for neuron in neurons:
        # print(k_measures)
        neuron_metrics = calculate_connectivity_metrics(neuron, k_measures, synapse_threshold)

        for metric, values in neuron_metrics.items():
            metrics[metric].append({'neuron': neuron.name,
                                    'values': np.array([k_measures[0:len(values)], values]),
                                    'color': neuron.color})

    for metric, results in metrics.items():
        plt.figure(figsize=[4, 4])
        for result in results:
            # plt.scatter(result['values'][0, :],
            #             result['values'][1, :],
            #             c=result['color'], s=10, alpha=.75, edgecolors=[.2, .2, .2], linewidths=.5)
            plt.plot(result['values'][0, :],
                     result['values'][1, :],
                     c=result['color'])
        plt.xlabel('Top K partners')
        plt.ylabel(f'{metric} @ K')
        plt.ylim([0, 1])
        plt.xlim([1, 50])
        # plt.savefig(DATA_DIR / f"{metric}_at_k.png", dpi=300, bbox_inches='tight')
        plt.savefig(DATA_DIR / f"{metric}_at_k_{synapse_threshold}.svg", transparent=True)


