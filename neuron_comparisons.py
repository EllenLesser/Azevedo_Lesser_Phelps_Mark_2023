from tqdm import tqdm
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import pymaid

import cloudvolume as cv
from caveclient.frameworkclient import CAVEclientFull
from fanc import auth, catmaid, lookup, transforms

from constants import VOXEL_RESOLUTION


class NeuronComparator:
    """Class for comparing a single neuron's manual and automated morphology and synapses."""

    catmaid_neuron: pymaid.CatmaidNeuron
    predicted_synapses: pd.DataFrame
    ground_truth_synapses: pd.DataFrame
    _old_predicted_synapses: pd.DataFrame
    _old_ground_truth_synapses: pd.DataFrame

    def __init__(self,
                 neuron_meta: Dict[str, Any],
                 cave_client: Union[None, CAVEclientFull] = None,
                 cloud_volume: Union[None, cv.frontends.graphene.CloudVolumeGraphene] = None,
                 clear_cache: bool = True,
                 load_synapses: bool = False):

        self.name = neuron_meta['neuron_name']
        self.skid = neuron_meta['skid']
        self.soma_coords_fancv4 = neuron_meta['pt']
        self.neuron_class = neuron_meta['class']
        # self.remote_instance = catmaid.connect(project_id=neuron_meta['catmaid_project_id'], make_global=False)
        self.remote_instance = catmaid.connect(project_id=neuron_meta['catmaid_project_id'])
        if clear_cache:
            self.remote_instance.clear_cache()
        self.color = neuron_meta['color']

        # Get client and cloudvolume if not supplied
        if cave_client is None:
            self.cave_client = auth.get_caveclient()
        else:
            self.cave_client = cave_client
        if cloud_volume is None:
            self.cloud_volume = auth.get_cloudvolume()
        else:
            self.cloud_volume = cloud_volume

        # Get root id
        self.root_id = lookup.segid_from_pt(self.soma_coords_fancv4)

        if load_synapses:
            self.get_post_synapses_ground_truth()
            self.get_post_synapses_predicted()

    def load_synapse_coords(self):
        if not self.ground_truth_synapses.empty:
            self._old_ground_truth_synapses = self.ground_truth_synapses
            self._old_predicted_synapses = self.predicted_synapses
        pickle_fn = './data/synapse_tables/{}_groundtruth_synapses.pkl'.format(self.name)
        self.ground_truth_synapses = pd.read_pickle(pickle_fn)
        self.predicted_synapses = pd.read_pickle('./data/synapse_tables/{}_predicted_synapses.pkl'.format(self.name))

    def get_mesh_neuron(self):
        """Get the mesh for the neuron from the cloudvolume."""
        mesh_manager = auth.get_meshmanager()
        self.mesh = mesh_manager.mesh(seg_id=self.root_id, remove_duplicate_vertices=True)

    def get_catmaid_neuron(self):
        """Get the catmaid neuron object for the neuron."""
        self.catmaid_neuron = pymaid.get_neuron(self.skid, remote_instance=self.remote_instance)

    def get_post_synapses_predicted(self):
        """Get predicted synapses from caveclient.
        :return: A synapse table for the outputs of the neuron.
        """
        self.predicted_synapses = self.cave_client.materialize.synapse_query(post_ids=self.root_id,timestamp='now')

    @staticmethod
    def get_synapse_array(coord_series: pd.Series):
        """Convert a coord column to an array, so it can be handled by transform methods."""
        return np.concatenate(coord_series.values).reshape(-1, 3)

    def get_post_synapses_ground_truth(self):
        """Get ground truth synapses from a catmaid neuron and format them similarly to a CAVE table"""
        ground_truth_synapses = self._get_catmaid_synapse_coords()
        ground_truth_synapses['pre_pt_supervoxel_id'] = lookup.svid_from_pt(
            self.get_synapse_array(ground_truth_synapses.pre_pt_position_v4))
        ground_truth_synapses['post_pt_supervoxel_id'] = lookup.svid_from_pt(
            self.get_synapse_array(ground_truth_synapses.post_pt_position_v4))
        ground_truth_synapses['pre_pt_root_id'] = lookup.segid_from_pt(
            self.get_synapse_array(ground_truth_synapses.pre_pt_position_v4))
        ground_truth_synapses['post_pt_root_id'] = lookup.segid_from_pt(
            self.get_synapse_array(ground_truth_synapses.post_pt_position_v4))
        self.ground_truth_synapses = ground_truth_synapses

    def _get_catmaid_synapse_coords(self, transform_to_v4: bool = True) -> pd.DataFrame:
        """Convert catmaid annotated synapses into pre and post synaptic coordinates in v4 alignment."""
        if not hasattr(self, 'catmaid_neuron'):
            self.get_catmaid_neuron()

        skid = np.int64(self.catmaid_neuron.skeleton_id)
        connectors = pymaid.get_connector_details(self.catmaid_neuron, remote_instance=self.remote_instance)
        synapses = []
        for index, row in tqdm(connectors.iterrows(), total=connectors.shape[0]):
            try:
                synapses.extend([
                    {
                        'pre_skid': np.int64(row['presynaptic_to']),
                        'post_skid': np.int64(post_syn),
                        'pre_node': np.int64(row['presynaptic_to_node']),
                        'post_node': np.int64(post_node),
                    }
                    for post_syn, post_node in zip(row['postsynaptic_to'], row['postsynaptic_to_node']) if post_syn == skid
                ])
            except ValueError(f"Error with {row}") as e:
                raise e
        synapse_table = pd.DataFrame(synapses)
        if synapse_table.shape[0]:
            pre_coords = pymaid.get_node_location(synapse_table.pre_node,
                                                  remote_instance=self.remote_instance).set_index('node_id').drop_duplicates()
            post_coords = pymaid.get_node_location(synapse_table.post_node,
                                                   remote_instance=self.remote_instance).set_index('node_id').drop_duplicates()
            synapse_table['pre_pt_position_v3'] = synapse_table.pre_node.apply(
                lambda x: pre_coords.loc[x, ['x', 'y', 'z']].values / VOXEL_RESOLUTION)
            synapse_table['post_pt_position_v3'] = synapse_table.post_node.apply(
                lambda x: post_coords.loc[x, ['x', 'y', 'z']].values / VOXEL_RESOLUTION)
            if transform_to_v4:
                synapse_table['pre_pt_position_v4'] = transforms.realignment.fanc3_to_4(
                    self.get_synapse_array(synapse_table.pre_pt_position_v3)).tolist()
                synapse_table['post_pt_position_v4'] = transforms.realignment.fanc3_to_4(
                    self.get_synapse_array(synapse_table.post_pt_position_v3)).tolist()
        return synapse_table

