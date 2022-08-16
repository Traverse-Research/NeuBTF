import numpy as np
import torch

# ds related
# converts spherical coords to 3D directions
def spherical2dir(theta: float, phi: float, r=1.) -> np.ndarray:
    """ 
    expects theta and phi in degrees, 
    theta = inclination and phi = azimuth.
    converts spherical coords to 3D directions
    """
    p = np.deg2rad(phi)
    t = np.deg2rad(theta)
    x = r * np.cos(p) * np.sin(t)
    y = r * np.sin(p) * np.sin(t)
    z = r * np.cos(t)
    return np.stack((x,y,z), axis=-1)

def gamma_correction(image, gamma=2.2):
    return image**(1./gamma)

def create_coords(side_len) -> np.ndarray:
    """ 
    creates a 2d meshgrid to be used as input coordinates [-1,1] , [-1,1]
    """
    x = np.linspace(-1, 1, side_len)
    y = x.copy()
    mgrid = np.stack(np.meshgrid(x, y, indexing='ij'), -1)
    return mgrid

def create_uvs(side_len) -> np.ndarray:
    """ 
    creates a 2d meshgrid to be used as input uvs [0,1], [1,0]
    """
    u = np.linspace(0., 1., side_len)
    v = np.linspace(1., 0., side_len)
    mgrid = np.stack(np.meshgrid(u, v, indexing='xy'), -1)
    return mgrid

def uvs2coords(uvs: torch.Tensor) -> torch.Tensor:
    """ 
    uvs to coords conversion on torch tensors
    """
    coords = uvs.clone()
    # invert v(y)
    coords[..., 1] = 1. - coords[..., 1]
    coords[..., 0] = coords[..., 1]
    coords[..., 1] = uvs[..., 0]
    # normalize to [-1., 1.]
    coords = coords * 2. - 1.
    spherical2dir
    return coords