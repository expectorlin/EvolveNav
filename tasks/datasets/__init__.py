from .base_dataset import MetaDataset

# import the dataset class here
from .r2r import R2RDataset
from .cvdn import CVDNDataset

def load_dataset(name, *args, **kwargs):
    cls = MetaDataset.registry[name]
    return cls(*args, **kwargs)