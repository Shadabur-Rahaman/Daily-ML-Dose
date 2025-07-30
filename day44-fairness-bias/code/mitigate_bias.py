from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset

RW = Reweighing(unprivileged_groups, privileged_groups)
dataset_transf = RW.fit_transform(dataset_orig_train)
