import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from .utils import compute_angle


## Contrastive Transforms ##
class ContrastiveTransform(object):
    """Take two transforms of one crystal"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        x1 = self.base_transform1(x)
        x2 = self.base_transform2(x)
        return [x1, x2]



## Structure Transforms ##
class LoadfromStructure(object):
    """Load infos from crystal structure"""
    def __init__(self, infos=['atomic_numbers', 'cart_coords', 'lattice_mat', 'num_atoms']):
        self.infos = infos

    def __call__(self, crystal):
        for info in self.infos:
            crystal[info] = getattr(crystal['structure'], info)
        return crystal


class Padding(object):
    """Padd atoms to a fix number"""
    def __init__(self, max_atoms=300):
        self.max_atoms = max_atoms
        self.mode = {
            1: (1, 1, 1),
            # 2: (1, 1, 2),
            # 4: (1, 2, 2),
            8: (2, 2, 2),
            # 12: (2, 2, 3),
            # 18: (2, 3, 3),
            27: (3, 3, 3),
            # 36: (3, 3, 4),
            # 48: (3, 4, 4),
            64: (4, 4, 4),
            # 80: (4, 4, 5),
            125: (5, 5, 5),
            216: (6, 6, 6),
            343: (7, 7, 7)
        }

    def __call__(self, crystal):
        n = len(crystal['atomic_numbers'])
        for repeat in self.mode:
            if n * repeat > self.max_atoms:
                atomic_numbers = crystal['atomic_numbers']
                coords = crystal['cart_coords']
                lattice = crystal['lattice_mat']
                crystal['atomic_numbers'] = np.tile(atomic_numbers, repeat)
                crystal['cart_coords'] = self.pad_coords(coords, lattice, repeat)
                break
        return crystal

    def pad_coords(self, coords, lattice, n):
        i, j, k = self.mode[n]
        i, j, k = np.meshgrid(np.arange(i), np.arange(j), np.arange(k), indexing='ij')
        ijk = np.stack([i, j, k], 3).reshape(-1, 1, 3)
        offset = ijk @ lattice
        coords = offset + np.expand_dims(coords, 0)
        return coords.reshape(-1, 3)


class Rotating(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, crystal):
        rot_mat = R.random().as_matrix()
        crystal['cart_coords'] = crystal['cart_coords'] @ rot_mat
        crystal['lattice_mat'] = crystal['lattice_mat'] @ rot_mat
        return crystal


class Centering(object):
    def __init__(self):
        pass

    def __call__(self, crystal):
        coords = crystal['cart_coords']
        crystal['cart_coords'] = coords - coords.mean(0)
        return crystal

class CutOff(object):
    def __init__(self, max_atoms=300):
        self.max_atoms = max_atoms

    def __call__(self, crystal):
        coords = crystal['cart_coords']
        idx = np.linalg.norm(coords, axis=-1).argsort()
        idx = idx[:self.max_atoms]
        crystal['atomic_numbers'] = crystal['atomic_numbers'][idx]
        crystal['cart_coords'] = crystal['cart_coords'][idx]
        return crystal

#于从晶体数据中收集指定的输入信息和目标信息
class StructureCollecting(object):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __call__(self, crystal):
        inputs = []
        targets = []
        for i in self.inputs:
            assert i in crystal
            if i == 'atomic_numbers' or i == 'num_atoms':
                inputs.append(torch.tensor(crystal[i], dtype=torch.long))
            else:
                inputs.append(torch.tensor(crystal[i], dtype=torch.float32))
        for i in self.targets:
            if i in crystal:
                targets.append(torch.tensor([crystal[i]], dtype=torch.float32))
            else:
                assert i in crystal['info']
                targets.append(torch.tensor([crystal['info'][i]], dtype=torch.float32))
        return inputs, targets


## Graph Transforms ##
class AddNodeFeature(object):
    def __init__(self, features=['atomic_numbers']):
        self.features = features

    def __call__(self, crystal):
        structure = crystal['structure']
        for feat in self.features:
            assert hasattr(structure, feat)
            crystal['graph'].ndata[feat] = torch.tensor(getattr(structure, feat), dtype=torch.long)
        return crystal


class AddEdgeFeature(object):
    def __init__(self, features=['distance']):
        self.features = features

    def __call__(self, crystal):
        for feat in self.features:
            if feat == 'offset':
                pass
            elif feat == 'distance':
                crystal['graph'].edata[feat] = torch.norm(crystal['graph'].edata.pop('offset'), dim=1).float()
        return crystal


class AddAngleFeature(object):
    def __init__(self, features=['angle']):
        self.features = features
    
    def __call__(self, crystal):
        assert 'line_graph' in crystal
        crystal['line_graph'].apply_edges(compute_angle)
        crystal['line_graph'].ndata.pop('offset')
        return crystal



class GraphCollecting(object):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __call__(self, crystal):
        if self.inputs == ['graph', 'line_graph']:
            inputs = crystal['graph'], crystal['line_graph']
        elif self.inputs == ['graph']:
            inputs = crystal['graph']
        else:
            raise ValueError('Invalid inputs!')
        targets = []
        for i in self.targets:
            if i in crystal:
                targets.append(torch.tensor([crystal[i]], dtype=torch.float32))
            else:
                assert i in crystal['info']
                targets.append(torch.tensor([crystal['info'][i]], dtype=torch.float32))
        return inputs, torch.cat(targets)


## Compose
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
