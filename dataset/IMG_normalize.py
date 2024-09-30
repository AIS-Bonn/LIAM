import os
import torch
import sys
sys.path.append("/home/user/wang01/ma_wang")
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm, trange
from config.config import CFG
import numpy as np
from PIL import Image

LOW_ACTION_DICT = {"LookDown": 0, "LookUp": 1, "RotateLeft": 2, "RotateRight": 3, "MoveAhead": 4,
                   "PickupObject": 5, "PutObject": 6,  "SliceObject": 7, "OpenObject": 8,
                   "CloseObject": 9, "ToggleObjectOn": 10, "ToggleObjectOff": 11, "<<stop>>": 12, "<<pad>>": 13}

class IMG_normalize(Dataset):
    def __init__(self, arg, mode):
        super(IMG_normalize, self).__init__()
        assert mode in ["train", "valid_seen", "valid_unseen", "test_seen", "test_unseen"]
        self.mode = mode
        self.arg = arg
        self.root_v = os.path.join(self.arg.ROOT_DIR, mode)
        self.samples = os.listdir(self.root_v)
        self.trials = [os.listdir(os.path.join(self.root_v, f)) for f in self.samples]
        self.samples_trials = [os.path.join(self.samples[idx], self.trials[idx][sidx]) for idx in range(len(self.samples)) for sidx in range(len(self.trials[idx]))]
        self.custom_transform = transforms.Compose([transforms.Resize((224,224)),
                                                   transforms.ToTensor()])

        #self.rawimg_li_ultimate = []
        #for idx in trange(len(self.samples_trials)):
        #    tmp = self.get_img_li(idx)
        #    self.rawimg_li_ultimate.extend(tmp)
    def get_img_li(self, idx):
        img_feat_root = os.path.join("/home/local/ET/data/generated_2.1.0/", self.mode, self.samples_trials[idx],
                                     "raw_images")
        img_li = sorted(os.listdir(img_feat_root))
        img_raw_dir = [self.custom_transform(Image.open(os.path.join(img_feat_root, i))) for i in img_li]
        return img_raw_dir


    def __getitem__(self, idx):
        img_feat_root = os.path.join("/home/local/ET/data/generated_2.1.0/", self.mode, self.samples_trials[idx],
                                     "raw_images")
        img_li = sorted(os.listdir(img_feat_root))
        img_raw_dir = [self.custom_transform(Image.open(os.path.join(img_feat_root, i))) for i in img_li]
        img_batch = torch.stack(img_raw_dir, dim=0)

        numpy_image = img_batch.numpy()
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std0 = np.std(numpy_image, axis=(0, 2, 3))



        return batch_mean, batch_std0

    def __len__(self):
        return len(self.samples_trials)

if __name__ == '__main__':
    arg = CFG()
    train_dataset = IMG_normalize(arg, "valid_unseen")
    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    pop_mean_li = []
    pop_std0_li = []
    for idx, (pop_mean, pop_std0) in enumerate(tqdm(trainloader)):
        pop_mean_li.append(pop_mean)
        pop_std0_li.append(pop_std0)
        #numpy_image = img.numpy()

        # shape (3,)
        #batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        #batch_std0 = np.std(numpy_image, axis=(0, 2, 3))


        #pop_mean.append(batch_mean)
        #pop_std0.append(batch_std0)

    pop_mean = np.array(pop_mean_li).mean(axis=0)
    pop_std0 = np.array(pop_std0_li).mean(axis=0)
    print(pop_mean, pop_std0)
    # Train [[0.49016342 0.41780347 0.33183882]] [[0.16816522 0.16262074 0.15201025]]
    # Valid seen: [0.48394567 0.41719523 0.3350161 ] [0.1985368  0.19323549 0.18687145]
    # Valid unseen: [[0.5145434  0.43689296 0.36367682]] [[0.17401849 0.17177609 0.17165425]]

