"""
NOTE
i have no idea what is necessary in this file, it does not seem
like any mujoco building is happening here so who cares and let's move on.
if errors point you here go here, what can i say
"""

from setuptools import setup, find_packages
#import sys, os.path
#
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gym_custom_fetch'))
#from version import VERSION
#
#extras = {
#  'mujoco': ['mujoco_py>=1.50, <2.0', 'imageio'],
#  'robotics': ['mujoco_py>=1.50, <2.0', 'imageio'],
#}
#
#extras['all'] = list(set([item for group in extras.values() for item in group]))

setup(name='gym_custom_fetch', version='0.0.1', install_requires=['gym'])



