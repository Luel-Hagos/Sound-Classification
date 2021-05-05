import io
import os
import glob

import torch
from torch import tensor
from torch import Tensor
import torchaudio

from random import randrange

from pathlib import Path
from typing import List, Tuple, Union

import torchaudio.functional as F
import torchaudio.transforms as T

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from IPython.display import Audio, display
from torchaudio.datasets.utils import (download_url, extract_archive, )

_RELEASE_CONFIGS = {
    "release1": {
        "folder_in_archive": "fold1",
        "url": "C:/Users/Dell/Documents/Pytorch/Data_Loader",
        #"checksum": "30301975fd8c5cac4040c261c0852f57cfa8adbbad2ce78e77e4986957445f27",
    }
}

class Dataloader(Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        url: str = _RELEASE_CONFIGS["release1"]["url"],
        folder_in_archive: str = _RELEASE_CONFIGS["release1"]["folder_in_archive"],
        download: bool = False,
        test: bool = False
    ) -> None:
        self._walker = []
        self.test = test
        self._parse_filesystem(root, url, folder_in_archive, download)
        
        n_fft = 1024
        hop_length = 512
        n_mels = 128
        sample_rate = 22050

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate = sample_rate,
            n_fft = n_fft,
            hop_length = hop_length,
            #center = True,
            #pad_mode = "reflect",
            power = 2.0,
            #norm= 'slaney',
            #onesided = True,
            n_mels=n_mels,
        )
        
        #self.Labels = Path(url+'/'+folder_in_archive).glob('*.wav').split['/'][-1].split('-')[-2]
    def _parse_filesystem(self, root: str, url: str, folder_in_archive: str, download: bool) -> None:
        
        for i in root:
            self._walker.extend(sorted(str(p) for p in Path(i).glob('*.wav')))
            
    def _load_item(self, fileid: str):
        waveform, _ = torchaudio.load(fileid)
        waveform = torch.mean(waveform, 0, True)
        label = fileid.split('\\')[-1].split('-')[1]
        
        # Perform transformation
        spec = self.mel_spectrogram(waveform)
        spec = torchaudio.transforms.AmplitudeToDB()(spec)

        tf_shape = spec.shape[-1]
        roll_no = randrange(tf_shape)
        spec = torch.roll(spec, roll_no, 2)
        # perform padding for training data
        pad = torch.zeros((1,128, 128))
        if pad.shape[-1] < tf_shape:
            indices = torch.arange(0, 128)
            spec = torch.index_select(spec, 2, indices)
        else:
            indices = torch.arange(0, tf_shape)
            spec = pad.index_copy_(2, indices, spec)
                
        return spec, int(label)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, List[int]]:
        fileid = self._walker[n]
        item = self._load_item(fileid)
        return item

    def __len__(self) -> int:
        return len(self._walker)