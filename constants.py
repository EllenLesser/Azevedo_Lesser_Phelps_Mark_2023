import os
import site

from pathlib import Path
from fanc.template_spaces import to_navis_name


def get_template_location():
    package_name = 'data'
    template_name = 'volume_meshes/JRC2018_VNC_FEMALE/VNC_neuropil_Aug2020.stl'

    site_packages = site.getsitepackages()[0]
    package_path = os.path.join(site_packages, package_name)
    if os.path.exists(package_path):
        file_path = os.path.join(package_path, template_name)
        if os.path.exists(file_path):
            return file_path
        else:
            raise FileNotFoundError(f"'{template_name}' not found inside '{package_name}'")


DATA_DIR = Path('data/')
NEURON_METADATA_PATH = DATA_DIR / 'neuron_metadata.json'
NEURON_PLOT_DIR = DATA_DIR / 'neuron_plots'

TEMPLATE_SPACE = 'JRC2018_VNC_FEMALE'
TEMPLATE_SPACE_NAVIS = to_navis_name(TEMPLATE_SPACE)
NEUROPIL_OUTLINE_PATH = get_template_location()
VOXEL_RESOLUTION = [4.3, 4.3, 45]
