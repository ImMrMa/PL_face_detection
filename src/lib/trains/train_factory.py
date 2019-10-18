from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer
from .ddd import DddTrainer
from .exdet import ExdetTrainer
from .multi_pose import MultiPoseTrainer
from .fadet import FadetTrainer
from .fadet2 import FadetTrainer as FadetTrainer2
from .cspdet import CspDetTrainer
train_factory = {
  'fadet':FadetTrainer,
  'exdet': ExdetTrainer, 
  'ddd': DddTrainer,
  'ctdet': CtdetTrainer,
  'multi_pose': MultiPoseTrainer, 
  'fadet2':FadetTrainer2,
  'cspdet':CspDetTrainer
}
