import os
from os.path import dirname

BASE_FOLDER = dirname(os.path.abspath(__file__))
EXT_LIBRARY_FOLDER = "ext"
GIBBSLDA_PATH = os.path.join(BASE_FOLDER, EXT_LIBRARY_FOLDER, "gibbslda", "GibbsLDA++-0.2", "lda")
BIGCLAM_PATH = os.path.join(BASE_FOLDER, EXT_LIBRARY_FOLDER, "agm-package", "bigclam", "bigclam")
COMMMUNITY_DETECTION_METHODS = ['lda', 'louvain', 'bigclam', 'bayesianhmm']