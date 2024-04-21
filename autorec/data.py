import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import load_config

cfg = load_config("./config/config.yml")


class MovieLensData(Dataset):
    def __init__(self, mode: str) -> None:
        super().__init__()

        self.mode = mode
        self.data = pd.read_csv(cfg["data_path"], sep="::", header=None,
                                names=["userId", "movieId", "rating", "timestamp"])
        self.mat = self.data.pivot_table(index="userId", columns="movieId", values="rating", fill_value=0.0).to_numpy()

    def __len__(self) -> int:
        if self.mode == "user":
            return self.mat.shape[0]  # num_user
        elif self.mode == "item":
            return self.mat.shape[1]  # num_item
        else:
            raise NotImplementedError

    def get_len_vec(self) -> int:
        if self.mode == "user":
            return self.mat.shape[1]  # num_item
        elif self.mode == "item":
            return self.mat.shape[0]  # num_user
        else:
            raise NotImplementedError

    def __getitem__(self, idx: int) -> torch.Tensor:
        # convert to float avoid double/float type mismatch
        if self.mode == "user":
            return torch.from_numpy(self.mat[idx, :]).float()  # user_idx
        elif self.mode == "item":
            return torch.from_numpy(self.mat[:, idx]).float()  # item_idx
        else:
            raise NotImplementedError
