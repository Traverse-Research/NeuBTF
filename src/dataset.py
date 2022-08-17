import numpy as np
import torch
from torch.utils.data import Dataset
import utils
from tqdm import tqdm
from btf_extractor import Ubo2014

# TODO: implement abstract BTFDataset class and UBO specific
class BTFDataset(Dataset):
    def __init__(self, path, sample_size=64, train_size=2048, validation_size=512, side_len=512, shuffle=True, max_level=4):
        self.shuffle = shuffle
        self.btf = Ubo2014(path)
        self.sample_size = sample_size
        #self.train_size = train_size
        self.side_len = side_len
        angles = list(self.btf.angles_set)
        self.angles = np.array(angles)
        self.train_size = train_size
        self.validation_size = validation_size
        choice = np.random.choice(len(angles), (self.train_size + self.validation_size,))
        self.train = self.angles[choice[:self.train_size]]
        self.validation = self.angles[choice[self.train_size:]]

        # rgb + uv + wi + wo
        self.data_ch = 3 + 2 + 3 + 3 + 1
        # UBO img width
        self.img_width = 400
        
        self.data = np.empty((self.train_size, self.img_width, self.img_width, self.data_ch), dtype=np.float32)
        
        print("importing btf dataset: {} \n".format(path))
        for n, a in enumerate(tqdm(angles)):
            t_l, p_l, t_v, p_v = a
            image = self.btf.angles_to_image(*a)
            image = image[...,::-1].copy()
            tensor_image = torch.tensor(image).permute(2, 0, 1) 
            
            size = self.img_width
            tot_size = np.sum([(size // (2**l))**2 for l in range(max_level)])
            sample = np.empty((tot_size, 3 + 2 + 1))
           
            coords = utils.create_uvs(size)
            sample = np.empty((size**2, 3 + 2 + 1))
            sample[..., :3] = image.reshape((-1, 3))
            sample[..., 3:5] = coords.reshape((-1, 2))
            sample[..., 5:] = 0.0            

            self.data[n,...,:3] = sample[...,:3].reshape(self.img_width, self.img_width,3)
            ##print(coords.shape)
            self.data[n,...,3:5] = sample[...,3:5].reshape(self.img_width, self.img_width,2)
            self.data[n,...,5:6] = sample[...,5:].reshape(self.img_width, self.img_width,1)
            wi = utils.spherical2dir(t_l, p_l)
            wo = utils.spherical2dir(t_v, p_v)
            dirs = np.array((*wi,*wo))
            dirs = np.tile(dirs, (self.img_width**2, 1))
            self.data[n,...,6:] = dirs.reshape(self.img_width, self.img_width,6) 
        
        self.data = self.data.reshape((-1, self.data_ch ))
        
    def get_shuffled_sample(self,):
        choice = np.random.choice(self.data.shape[0], (self.side_len**2,))

        data = self.data[choice].reshape((self.side_len,self.side_len,self.data_ch))
        image  = data[...,:3]
        coords = data[...,3:5]
        level = data[...,5:6]
        wi = data[...,6:9]
        wo = data[...,9:]
        
        #convert to tensors
        image = torch.tensor(image, dtype=torch.float32)
        coords = torch.tensor(coords, dtype=torch.float32)
        wi = torch.tensor(wi, dtype=torch.float32)
        wo = torch.tensor(wo, dtype=torch.float32)
        level = torch.tensor(level, dtype=torch.float32)

        return image, coords, level, wi, wo

    def get_sample(self, idx):
        a = self.angles[idx]
        t_l, p_l, t_v, p_v = a
        image = self.btf.angles_to_image(*a)
        image = image[...,::-1].copy()
        
        wi = utils.spherical2dir(t_l, p_l)
        wo = utils.spherical2dir(t_v, p_v)
        wi = np.tile(wi, (self.img_width, self.img_width, 1))
        wo = np.tile(wo, (self.img_width, self.img_width, 1))
        level = np.tile(0., (self.img_width, self.img_width, 1))
        coords = utils.create_uvs(self.img_width).reshape((self.img_width, self.img_width,2))

        #convert to tensors
        image = torch.tensor(image, dtype=torch.float32)
        coords = torch.tensor(coords, dtype=torch.float32)
        wi = torch.tensor(wi, dtype=torch.float32)
        wo = torch.tensor(wo, dtype=torch.float32)
        level = torch.tensor(level, dtype=torch.float32)

        return image, coords, level, wi, wo
        
    def get_validation_generator(self,):
        coords = utils.create_uvs(self.img_width).reshape((self.img_width, self.img_width,2))
        coords = torch.tensor(coords, dtype=torch.float32)
        for a in self.validation:
            t_l, p_l, t_v, p_v = a
            image = self.btf.angles_to_image(*a)
            image = image[...,::-1].copy()
        
            wi = utils.spherical2dir(t_l, p_l)
            wo = utils.spherical2dir(t_v, p_v)
            wi = np.tile(wi, (self.img_width, self.img_width, 1))
            wo = np.tile(wo, (self.img_width, self.img_width, 1))
            level = np.tile(0., (self.img_width, self.img_width, 1))
            
            level = torch.tensor(level, dtype=torch.float32)
            image = torch.tensor(image, dtype=torch.float32)
            
            wi = torch.tensor(wi, dtype=torch.float32)
            wo = torch.tensor(wo, dtype=torch.float32)
            yield image, coords, level, wi, wo
    
    def get_ds_generator(self,):
        for i in range(self.train_size):
            yield self.get_sample(i)
    
    def __len__(self):
        return self.train_size

    def __getitem__(self, idx):
        if self.shuffle:
            return self.get_shuffled_sample()
        else:
            return self.get_sample(idx)