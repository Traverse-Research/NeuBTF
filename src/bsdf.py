import numpy as np

import enoki as ek
# from mitsuba.core import Bitmap, Struct, Thread, math, Properties, Frame3f, Float, Vector3f, warp, Transform3f
# from mitsuba.render import BSDF, BSDFContext, BSDFFlags, BSDFSample3f, SurfaceInteraction3f, register_bsdf, Texture

from mitsuba.core import math,  Frame3f, Float, Vector3f, warp, Transform3f
from mitsuba.render import BSDF, BSDFFlags, BSDFSample3f

from models import NeuBTF

# Custom BSDF plugin for Mitsuba2
# https://mitsuba2.readthedocs.io/en/latest/src/advanced_topics/custom_plugins.html
class NeuBSDF(BSDF):
    def __init__(self, props, btf: NeuBTF):
        BSDF.__init__(self, props)
        self.m_transform = Transform3f(props["to_uv"].extract())

        self.m_flags = BSDFFlags.DiffuseReflection | BSDFFlags.FrontSide
        self.m_components = [self.m_flags]
        self.btf = btf
    
    def init_neumip(self, neumip_btf):
        self.btf = neumip_btf
    
    
    def btf_sample(self, wi, wo, uv):       
        uv = self.m_transform.transform_point(uv)
        rgb = self.btf.btf_sample(np.array(wi), np.array(wo), np.array(uv))
        rgb = np.clip(rgb, 0., 1.)
        return Vector3f(rgb)

    def sample(self, ctx, si, sample1, sample2, active):
        cos_theta_i = Frame3f.cos_theta(si.wi)

        active &= cos_theta_i > 0

        bs = BSDFSample3f()
        bs.wo  = warp.square_to_cosine_hemisphere(sample2)
        bs.pdf = warp.square_to_cosine_hemisphere_pdf(bs.wo)
        bs.eta = 1.0
        bs.sampled_type = +BSDFFlags.DiffuseReflection
        bs.sampled_component = 0

        
        value = self.btf_sample(si.wi, bs.wo, si.uv)

        return ( bs, ek.select(active & (bs.pdf > 0.0), value, Vector3f(0)) )

    def eval(self, ctx, si, wo, active):
        if not ctx.is_enabled(BSDFFlags.DiffuseReflection):
            return Vector3f(0)

        cos_theta_i = Frame3f.cos_theta(si.wi)
        cos_theta_o = Frame3f.cos_theta(wo)

        
        value = self.btf_sample(si.wi, wo, si.uv) * math.InvPi * cos_theta_o

        return ek.select((cos_theta_i > 0.0) & (cos_theta_o > 0.0), value, Vector3f(0))

    def pdf(self, ctx, si, wo, active):
        if not ctx.is_enabled(BSDFFlags.DiffuseReflection):
            return Vector3f(0)

        cos_theta_i = Frame3f.cos_theta(si.wi)
        cos_theta_o = Frame3f.cos_theta(wo)

        pdf = warp.square_to_cosine_hemisphere_pdf(wo)

        return ek.select((cos_theta_i > 0.0) & (cos_theta_o > 0.0), pdf, 0.0)


