import os
from typing import Union

import numpy as np
from matplotlib import colors, cm

import meshparty
import pymaid
import vtk
from fanc.visualize import read_mesh_stl, scale_bar_actor_2D
from fanc import auth, connectivity
from fanc.transforms import template_alignment
from meshparty import trimesh_vtk, trimesh_io, meshwork
from navis import xform_brain
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLCamera

from constants import TEMPLATE_SPACE_NAVIS, NEURON_PLOT_DIR, NEUROPIL_OUTLINE_PATH
from neuron_comparisons import NeuronComparator

# These are approximate x/y centers of the T1L and T1R neuropils in the template space.
RIGHT_T1_CENTER = (95184.48025184708, 119185.49864607782, 384689.7268014647)
LEFT_T1_CENTER = (174034.11026983315, 119310.89525140286, 336026.3672285372)


def mesh_neuron_plot(neuron: NeuronComparator):
    """Plot the neuron mesh in the template space.
    :param neuron: NeuronComparator
    """
    camera = get_camera(neuron)

    plot_neurons(neuron.root_id,
                 show_outlines=True,
                 camera=camera,
                 cmap=[neuron.color],
                 save=True,
                 save_path=NEURON_PLOT_DIR / f"{neuron.name}_mesh.png")


def get_camera(neuron: NeuronComparator) -> vtkOpenGLCamera:
    """Get camera orientation appropriate for the neuron in template space.
    :param neuron: NeuronComparator
    :return: vtkOpenGLCamera
    """
    if 'T1L' in neuron.neuron_class:
        focus_point = LEFT_T1_CENTER
    else:
        focus_point = RIGHT_T1_CENTER

    camera = trimesh_vtk.oriented_camera(
        focus_point,
        backoff=5,
        backoff_vector=[0, 0, -1],
        up_vector=[0, -1, 0]
    )
    return camera


def get_neuropil_mesh():
    """Get the neuropil mesh in the template space."""
    mesh = read_mesh_stl(NEUROPIL_OUTLINE_PATH)
    mp_mesh = trimesh_io.Mesh(mesh[0], mesh[1])
    return mp_mesh


def catmaid_to_mpskel(catmaid_neuron: pymaid.CatmaidNeuron) -> meshparty.skeleton.Skeleton:
    """Convert a pymaid.CatmaidNeuron to a meshparty.skeleton.Skeleton.
    :param catmaid_neuron: pymaid.CatmaidNeuron
    :return: meshparty.skeleton.Skeleton
    """
    vertices = catmaid_neuron.nodes.loc[:, ['x', 'y', 'z']].values * 1000
    edges = catmaid_neuron.nodes.loc[:, ['node_id', 'parent_id']]
    node_map = dict(zip(edges.node_id, range(0, len(edges))))
    edges.node_id = [node_map[i] for i in edges.node_id]
    edges.parent_id.loc[edges.parent_id != -1] = [node_map[i] for i in edges.parent_id.loc[edges.parent_id != -1]]
    edges = edges.values
    edges = edges[edges[:, 1] != -1]
    return meshparty.skeleton.Skeleton(vertices, edges)


def skel_neuron_plot(neuron: NeuronComparator):
    """Plot the neuron skeleton in the template space.
    :param neuron: NeuronComparator
    """
    if not hasattr(neuron, 'catmaid_neuron'):
        neuron.get_catmaid_neuron()

    camera = get_camera(neuron)

    catmaid_neuron = neuron.catmaid_neuron.copy()
    remote_instance = neuron.remote_instance
    template_aligned_neuron = xform_brain(catmaid_neuron, source='FANC', target=TEMPLATE_SPACE_NAVIS)
    mp_skel = catmaid_to_mpskel(template_aligned_neuron)

    neuropil_mesh = get_neuropil_mesh()
    outlines_inner = meshwork.Meshwork(neuropil_mesh, seg_id=[2], voxel_resolution=[4.3, 4.3, 45])

    to_render = [
        create_soma_actor(template_aligned_neuron.soma_pos[0] * 1000, get_soma_radius(catmaid_neuron, remote_instance), neuron.color)]
    to_render.append(trimesh_vtk.skeleton_actor(mp_skel, color=neuron.color))
    to_render.append(trimesh_vtk.mesh_actor(outlines_inner.mesh, color=(.1, .1, .1), opacity=0.1))

    trimesh_vtk.render_actors(to_render,
                              camera=camera,
                              do_save=True, filename=NEURON_PLOT_DIR / f"{neuron.name}_skeleton.png")


def get_soma_radius(neuron: pymaid.CatmaidNeuron, remote_instance: pymaid.CatmaidInstance) -> float:
    """Look up the soma radius
    :param neuron: pymaid.CatmaidNeuron
    :return: the radis of the soma in nm.
    """
    node_table = pymaid.get_node_table(neuron, remote_instance=remote_instance)
    soma_id = neuron.soma[0]
    return node_table.loc[node_table.node_id == soma_id].radius.values[0]


def create_soma_actor(center: np.ndarray, radius: float, color: list[float]):
    """
    Creates a VTK actor for plotting somas of skeletons.
    :param center: center of the soma
    :param radius: radius of the soma
    :param color: color of the soma
    """

    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(center)
    sphere.SetRadius(radius)
    sphere.SetPhiResolution(30)
    sphere.SetThetaResolution(30)

    # Create a mapper.
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())

    # Create an actor.
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)

    return actor


def plot_neurons(segment_ids: Union[list[int], int],
                 template_space: Union[str, None] ='JRC2018_VNC_FEMALE',
                 cmap: Union[str, list] = 'Blues', opacity=1,
                 plot_type: str ='mesh',
                 resolution: Union[tuple[float], list[float], np.ndarray] = (4.3,4.3,45),
                 camera=None,
                 zoom_factor=300,
                 plot_synapses=False,
                 synapse_type='all',
                 synapse_threshold=3,
                 plot_soma=False,
                 show_outlines=False,
                 scale_bar_origin_3D=None,
                 scale_bar_origin_2D=None,
                 view='X',
                 scale_bar_length=10000,
                 save=False,
                 save_path=None,
                 width=1080,
                 height=720,
                 **kwargs):
    """
    Visualize neurons in 3d meshes, optionally saving high-resolution png images.

    Parameters
    ----------
    segment_ids :  list
        list of segment IDs of neurons
    template_space :  str
        Name of template space to warp neurons into. Must be one of:
          'JRC2018_VNC_FEMALE'
          'JRC2018_VNC_UNISEX'
          'JRC2018_VNC_MALE'
          'FANC'
          None
        Both 'FANC' and None result in neurons being displayed in the
        original FANC-space (i.e. no warping is applied).
    camera :  int
        json state id of neuroglancer scene. required to plot scale bar
    plot_synapses :  bool
        visualize synapses
    plot_soma : bool
        visualize soma
    show_outlines :  bool
        visualize volume outlines
    scale_bar_origin_3D : list
        specify an origin of a 3D scale bar that users want to place in xyz
    scale_bar_origin_2D :  list
        specify an origin of a 2D scale bar that users want to place in xyz
    view : str
        'X', 'Y', or 'Z' to specify which plane you want your 2D scale bar to appear
    scale_bar_length :  int
        length of a scale bar in nm
    save : bool
        write png image to disk, if false will open interactive window (default False)
    save_path : str
        filepath to save png image

    Additional kwargs
    -----------------
    client : caveclient.CAVEclient
        CAVEclient to use instead of the default one

    Returns
    -------
    vtk.vtkRenderer
        renderer when code was finished
    png
        output png image
        (generate two images with/without scale bar if you specify to plot it)
        :param resolution:
    """

    if isinstance(segment_ids, (int, np.integer)):
        segment_ids = [segment_ids]

    if isinstance(cmap, str):
        colormap = cm.get_cmap(cmap, len(segment_ids))
    elif isinstance(cmap, list):
        #Pad the cmap if it is length 1 t
        if len(cmap) < 2:
            cmap.append([0, 0, 0])
        colormap = colors.LinearSegmentedColormap.from_list('custom_colormap', cmap, N=len(cmap))
    else:
        raise ValueError('cmap not a valid colormap or color.')

    if 'client' in kwargs:
        client = kwargs['client']
    else:
        client = auth.get_caveclient()

    if isinstance(camera, (int, np.integer)):
        state = client.state.get_state_json(camera)
        camera = trimesh_vtk.camera_from_ngl_state(state, zoom_factor=zoom_factor)

    meshmanager = auth.get_meshmanager()

    neuron_actors = []
    annotation_actors = []
    # outline_actor = []
    for j in enumerate(segment_ids):
        # Get mesh
        mp_mesh = meshmanager.mesh(seg_id=j[1])
        if template_space and not template_space.startswith('FANC'):
            template_alignment.align_mesh(mp_mesh, target_space=template_space, inplace=True)
            mp_mesh.vertices *= 1000  # TODO delete this after adding nm/um to align_mesh
        neuron = meshwork.Meshwork(mp_mesh, seg_id=j[1], voxel_resolution=[4.3, 4.3, 45])

        if plot_soma == True:
            soma_df = client.materialize.query_table(client.info.get_datastack_info()['soma_table'],
                                                     filter_equal_dict={'pt_root_id': j[1]})
            neuron.add_annotations('soma_pt', soma_df, point_column='pt_position', anchored=False)

        # get synapses
        if plot_synapses is True:
            if synapse_type == 'inputs':
                input_table = connectivity.get_synapses(j[1],
                                                          direction='inputs',
                                                          threshold=synapse_threshold)

                neuron.add_annotations('syn_in', input_table, point_column='post_pt')


            elif synapse_type == 'outputs':
                input_table = None
                output_table = connectivity.get_synapses(j[1],
                                                           direction='outputs',
                                                           threshold=synapse_threshold)
            elif synapse_type == 'all':
                input_table = connectivity.get_synapses(j[1],
                                                          direction='inputs',
                                                          threshold=synapse_threshold)

                output_table = connectivity.get_synapses(j[1],
                                                           direction='outputs',
                                                           threshold=synapse_threshold)

                neuron.add_annotations('syn_in', input_table, point_column='post_pt')
                neuron.add_annotations('syn_out', output_table, point_column='pre_pt')


            else:
                raise Exception('incorrect synapse type, use: "inputs", "outputs", or "all"')

        # Plot

        if 'mesh' in plot_type:
            neuron_actors.append(trimesh_vtk.mesh_actor(neuron.mesh, color=colormap(j[0])[0:3], opacity=opacity))
        elif 'skeleton' in plot_type and plot_soma is not None:
            neuron.skeletonize_mesh(soma_pt=neuron.anno.soma_pt.points[0], invalidation_distance=5000)
            neuron_actors.append(trimesh_vtk.skeleton_actor(neuron.skeleton, line_width=3, color=colormap(j[0])[0:3]))
        elif 'skeleton' in plot_type and plot_soma is None:
            raise Exception('need a soma point to skeletonize')
        else:
            raise Exception('incorrect plot type, use "mesh" or "skeleton"')

        for i in neuron.anno.table_names:
            if 'syn_in' in i:
                annotation_actors.append(
                    trimesh_vtk.point_cloud_actor(neuron.anno.syn_in.points, size=200, color=(0.0, 0.9, 0.9)))
            elif 'syn_out' in i:
                annotation_actors.append(
                    trimesh_vtk.point_cloud_actor(neuron.anno.syn_out.points, size=200, color=(1.0, 0.0, 0.0)))
            else:
                annotation_actors.append(
                    trimesh_vtk.point_cloud_actor(neuron.anno[i].points, size=200, color=(0.0, 0.0, 0.0)))

    all_actors = neuron_actors + annotation_actors

    if show_outlines:
        outlines_actors = []
        base = os.path.join('data/volume_meshes')
        if template_space == 'JRC2018_VNC_FEMALE':
            inner_mesh_filename = os.path.normpath(os.path.join(base, 'JRC2018_VNC_FEMALE', 'VNC_neuropil_Aug2020.stl'))
        elif template_space == 'FANC' or not template_space:
            inner_mesh_filename = os.path.normpath(os.path.join(base, 'JRC2018_VNC_FEMALE_to_FANC', 'VNC_template_Aug2020.stl'))
        else:
            raise NotImplementedError

        mesh_inner = read_mesh_stl(inner_mesh_filename)
        mp_mesh = trimesh_io.Mesh(mesh_inner[0], mesh_inner[1])
        outlines_inner = meshwork.Meshwork(mp_mesh, seg_id=[2], voxel_resolution=[4.3, 4.3, 45])
        outlines_actors.append(trimesh_vtk.mesh_actor(outlines_inner.mesh, color=(.1, .1, .1), opacity=0.1))

        all_actors = all_actors + outlines_actors

    # add actor for scale bar
    if (scale_bar_origin_3D is not None) or (scale_bar_origin_2D is not None):
        if camera is not None:
            if scale_bar_origin_3D is not None:
                scale_bar_ctr = np.array(scale_bar_origin_3D)*np.array(resolution) # - np.array([0,scale_bar_length,0])
                scale_bar_actor = trimesh_vtk.scale_bar_actor(scale_bar_ctr,camera=camera,length=scale_bar_length,linewidth=1)
            else:
                scale_bar_ctr = np.array(scale_bar_origin_2D)*np.array(resolution) - np.array([0,scale_bar_length,0])
                scale_bar_actor = scale_bar_actor_2D(scale_bar_ctr,view=view,camera=camera,length=scale_bar_length,linewidth=1)
        else:
            raise Exception('Need camera to set up scale bar')

    if (scale_bar_origin_3D is None) and (scale_bar_origin_2D is None):
        trimesh_vtk.render_actors(all_actors, camera=camera, do_save=save,
                                  filename=save_path,
                                  scale=4, video_width=width, video_height=height)
    elif save_path is None:
        trimesh_vtk.render_actors((all_actors + [scale_bar_actor]), camera=camera, do_save=save,
                                  filename=save_path,
                                  scale=4, video_width=width, video_height=height)
    else:
        trimesh_vtk.render_actors(all_actors, camera=camera, do_save=save,
                                  filename=save_path,
                                  scale=1, video_width=width, video_height=height)
        trimesh_vtk.render_actors((all_actors + [scale_bar_actor]), camera=camera, do_save=save,
                                    filename=(save_path.rsplit('.', 1)[0] + '_scalebar.' + save_path.rsplit('.', 1)[1]),
                                    scale=1, video_width=width, video_height=height)

