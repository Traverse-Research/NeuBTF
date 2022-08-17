import numpy as np
import torch
from torch import nn
import utils
from torchvision.transforms.functional import gaussian_blur


class SineLayer(nn.Module):    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False):
        super().__init__()
        
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(nn.ReLU())

        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features , hidden_features, ))
            self.net.append(nn.ReLU())
        
                
        self.net.append(nn.Linear(hidden_features, out_features))
        if not outermost_linear:
            self.net.append(nn.ReLU())
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        return self.net(x)  

class NeuTex(nn.Module):
    def __init__(self, resolution, embeddings_ch, bound=10e-4, interp_mode='bilinear'):
        super().__init__()
        self.resolution     = resolution
        self.embeddings_ch  = embeddings_ch
        self.interp_mode    = interp_mode
        tex = torch.empty((1, embeddings_ch ,resolution, resolution), dtype=torch.float32, requires_grad=True)
        torch.nn.init.uniform_(tex, a=-bound, b=bound)
        self.tex = nn.Parameter(tex)
    
    def forward(self, uvs):
        coords = self.uvs2coords(uvs)
        return self.grid_sample(coords)
    
    def uvs2coords(self, uvs):
        coords = uvs - torch.trunc(uvs)
        return utils.uvs2coords(coords)
    
    def grid_sample(self, coords):
        batch_size = coords.shape[0]
        tex = self.tile_texture(batch_size)
        return torch.nn.functional.grid_sample(
            tex, 
            coords, 
            mode=self.interp_mode, 
            padding_mode='zeros', 
            align_corners=True)
    
    def tile_texture(self, batch_size):
        return self.tex.repeat((batch_size,1,1,1))
        
    def fuse_blur(self, sigma, kernel_size=17):
        with torch.no_grad():
            blurred = gaussian_blur(self.tex[0], kernel_size, sigma)
            blurred = blurred.unsqueeze(0)
            self.tex.copy_(self.tex * 0.4 + blurred * 0.6)
           
    def set_interp_mode(self, interp_mode='bilinear'):
        self.interp_mode = interp_mode

class NeuMipMap(NeuTex):
    def __init__(self, resolution,  embeddings_ch, n_levels, bound=10e-4, interp_mode='bilinear', concat=False):
        super().__init__(resolution,  embeddings_ch, bound=bound, interp_mode=interp_mode)
        self.concat     = concat
        self.n_levels   = n_levels
        levels = list()
        current_resolution = resolution
        #TODO could calculate the max level
        for _ in range(self.n_levels):
            nex = NeuTex(current_resolution, embeddings_ch, bound=bound)
            levels.append(nex)
            current_resolution//=2
        self.levels = nn.ModuleList(levels)
            
    def forward(self, uvs, level):
        coords = self.uvs2coords(uvs)
        if self.concat:
            result = list()
            level = level
            for n, l in enumerate(self.levels):
                sample = l.grid_sample(coords)
                sample = sample.permute(0, 2, 3, 1)
                z = torch.zeros_like(sample)
                y = torch.where(level<=n, sample, z)
                result.append(y.permute(0, 3, 1, 2))
            return torch.cat(result, 1) 
        else: 
            if len(self.levels) > 1:
                level = level.permute(0, 3, 1, 2)
                base = torch.floor(level)
                alpha = level - base
                alpha = alpha.float()
                base = base.long()
                base = torch.clamp(base, len(self.levels)-1)
                next = base + 1
                next = torch.clamp(next, len(self.levels)-1)

                base = base.float()
                next = next.float()
                result = 0.0
                for n, l in enumerate(self.levels):
                    sample = l.grid_sample(coords) 
                    z = torch.zeros_like(sample[:, :1, :, :])
                    y = torch.where(base==n, sample, z) *(1. - alpha) + torch.where(next==n, sample, z) * (alpha)

                    result += y
                return result
            else:
                return self.levels[0].grid_sample(coords) 
     
    
    def fuse_blur(self, sigma, kernel_size=17):
        with torch.no_grad():
            for n, l in enumerate(self.levels):
                l.fuse_blur(sigma, kernel_size)
                kernel_size = kernel_size//2 + 1
    
    def recreate_pyramid(self):
        with torch.no_grad():
            for n in range(self.n_levels):
                if n != 0:
                    tex = self.levels[n-1].tex
                    tex = torch.nn.functional.interpolate(tex, scale_factor=(.5, .5), mode='area')
                    tex = tex.detach().requires_grad_()
                    self.levels[n].tex.copy_(tex)
    
    def set_interp_mode(self, interp_mode='bilinear'):
        self.interp_mode = interp_mode
        for l in self.levels:
            l.set_interp_mode(self.interp_mode)


class NeuBTF(nn.Module):
    def __init__(self, resolution, embeddings_ch, hidden_features, hidden_layers, out_features, outermost_linear=True, 
                 first_omega_0=30, hidden_omega_0=30., parallax_scale=0.1, concat=False, shared=False, siren=False, n_levels=5):
        super().__init__()
        self.concat         = concat
        self.shared         = shared
        self.siren          = siren
        self.resolution     = resolution
        self.embeddings_ch  = embeddings_ch
        self.n_levels       = n_levels
        self.parallax_scale = parallax_scale
        if concat:
            self.embeddings_ch *= n_levels
            self.tex = NeuMipMap(resolution, embeddings_ch, n_levels, concat=True)
            if not shared:
                self.off_tex = NeuMipMap(resolution, embeddings_ch, n_levels,concat=True)
        else:
            self.tex = NeuMipMap(resolution, embeddings_ch, 1)
            if not shared:
                self.off_tex = NeuMipMap(resolution, embeddings_ch, 1)
        
        self.sigma = 8.0
        
        net_in_ch = self.embeddings_ch + 6 
        off_net_in_ch = self.embeddings_ch + 3 
        
        if siren:
            self.net = Siren(
                net_in_ch , hidden_features, hidden_layers, 
                out_features, outermost_linear=outermost_linear,
                first_omega_0=first_omega_0,hidden_omega_0=hidden_omega_0)
            self.off_net = Siren( 
                off_net_in_ch, hidden_features, hidden_layers, 1, 
                outermost_linear=outermost_linear,
                first_omega_0=first_omega_0,hidden_omega_0=hidden_omega_0)
        else: 
            self.net = MLP(
                net_in_ch , hidden_features, 
                hidden_layers, out_features, 
                outermost_linear=outermost_linear)
            self.off_net = MLP(
                off_net_in_ch , hidden_features, 
                hidden_layers, 1, 
                outermost_linear=outermost_linear)

    # todo: swap wi, wo with mitsuba    
    def btf_sample(self, wi, wo, uv, side_len=512) -> np.ndarray:  
        sample_size = len(wi)
        step = side_len**2
        
        result = list()

        for start in range(0, sample_size, step):
            end = min((start + step, sample_size))
            s_uv = torch.tensor(uv[start:end], dtype=torch.float32)
            s_wi = torch.tensor(wi[start:end], dtype=torch.float32)
            s_wo = torch.tensor(wo[start:end], dtype=torch.float32)

            if end == sample_size:
                s_uv = s_uv.reshape(( 1, 1, -1, 2))
                s_wi = s_wi.reshape((1, 1, -1, 3))
                s_wo = s_wo.reshape((1, 1, -1, 3))

            else: 
                s_uv = s_uv.reshape(( 1, side_len, side_len, 2 ))
                s_wi = s_wi.reshape((1, side_len, side_len, 3 ))
                s_wo = s_wo.reshape((1, side_len, side_len, 3 ))

            level = torch.zeros_like(s_uv[..., :1])
            level = level.cuda()
            s_uv =  s_uv.cuda()
            s_wi =  s_wi.cuda()
            s_wo =  s_wo.cuda()

            y, _, _, _ = self.forward(
                s_uv, 
                level,
                s_wi,
                s_wo,
            )

            y = y.detach().cpu().reshape((-1,3)).numpy()
            result.append(y)

        result = np.concatenate(result, axis=0)

        return result
    
    def forward(self, uvs, level, wi, wo, skip_offset=False):
        offset, neu_depth = self.calculate_offset(uvs, wo, level)
        new_uvs = uvs + offset
        net_in = self.tex(new_uvs, level)
        net_in = torch.cat(( net_in.permute(0,2,3,1), wi, wo,),-1)
        result = self.net(net_in)
        
        return result, net_in[...,:self.embeddings_ch], new_uvs, neu_depth
    
    def calculate_offset(self, uvs, wo, level):
        if self.shared:
            off_in = self.tex(uvs, level)
        else:
            off_in = self.off_tex(uvs, level)
        off_in = torch.cat((off_in.permute(0, 2, 3, 1), wo, ),-1)
        
        neu_depth = self.off_net(off_in)

        offset = self.parallax_mapping(neu_depth, wo, self.parallax_scale)
        offset[...,1] = - offset[...,1]
        return offset, neu_depth
    
    def freeze_offset(self):
        for param in self.off_net.parameters():
            param.requires_grad = False
        for param in self.off_tex.parameters():
            param.requires_grad = False
    
    def fuse_blur(self, step, max_step):
        half_step = max_step//2
        
        if step <  half_step:
            sigma = self.sigma * (2.**(-step/(half_step//2)))
            kernel_size = 17 
            self.tex.fuse_blur(sigma, kernel_size)
            if not self.shared:
                self.off_tex.fuse_blur(sigma, kernel_size)
                
    def set_interp_mode(self, interp_mode='bilinear'):
        self.tex.set_interp_mode(interp_mode)
        if not self.shared:
            self.off_tex.set_interp_mode(interp_mode)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        
    def parallax_mapping(self, depth, w_o, scale=1.0):
        p = w_o[..., :2] / torch.clamp(w_o[..., 2:], min=0.6)
        p *= (depth*scale)
        return p