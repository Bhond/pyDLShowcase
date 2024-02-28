from torch.utils.data import Dataset


class YoloDataset(Dataset):
    def __init__(self, root=None):
        self.root = root

    def __getitem__(self, item):
        print("GetItem method")

    def __len__(self):
        print("Len method")

