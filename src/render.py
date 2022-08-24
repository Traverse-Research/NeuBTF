import argparse


# importing module
import sys
from tqdm import tqdm
import os

# import argparse
# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument("-i", "--input", type=str, default="btf-rendering/scenes/cloth/cloth_ngpu_ubo_2.xml", help="Input scene filepath (.xml)")
# parser.add_argument("-o", "--output", type=str, default="rendered.jpg", help="Output image filepath (.jpg, .png)")
# parser.add_argument("-m", "--mode", type=str, default="gpu_rgb", help="Rendering mode (scalar_rgb or gpu_rgb)")
# args = parser.parse_args()

from train import init_model, train_config, ablation_config, ablation_materials

scenes = {
    "cloth": "../scenes/cloth_neubtf.xml",
    "matpreview": "../scenes/matpreview_neubtf.xml"
}

def render(render_path, model, scene_path, mode="gpu_rgb"):
    import mitsuba
    mitsuba.set_variant(mode)
    from mitsuba.core import Bitmap, Struct, Thread
    from mitsuba.core.xml import load_file
    from mitsuba.render import register_bsdf
    from bsdf import NeuBSDF

    register_bsdf('neubtf', lambda props: NeuBSDF(props, model))

    Thread.thread().file_resolver().append(os.path.dirname(scene_path))
    scene = load_file(scene_path)

    scene.integrator().render(scene, scene.sensors()[0])

    film = scene.sensors()[0].film()
    bmp = film.bitmap(raw=True)
    img = bmp.convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, srgb_gamma=True)
    img.write(render_path)
    return img
    

if __name__ == "__main__":
    out_dir = "rendering"
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    out_dir = os.path.join(out_dir, 'ubo2014')
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    model_dir = os.path.join("models", 'ubo2014')
    for mat in tqdm(ablation_materials):
        save_path = os.path.join(out_dir, mat)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)


        model_path = os.path.join(model_dir, mat)

        for n, (siren, shared, concat) in enumerate(tqdm(combs)):
            if True:
                config["siren"] = siren
                config["shared"] = shared
                config["concat"] = concat
                if concat:
                    config["embeddings_ch"] = 4
                else:
                    config["embeddings_ch"] = 7
                print('running configuration {}: {} {} {}'.format(n, siren, shared, concat))



                import torch
                model = init_model(config=config, cuda=True)
              

                model_name = os.path.join(model_path,'{}.pth'.format(1 + n))
                model.load(model_name)
                model = model.cuda()
                
                # Register MeasuredBTF
                register_bsdf('neubtf', lambda props: NeuBTF(props, model))

                # Filename
                filename_src = args.input
                
                filename_dst = os.path.join(save_path,'{}.jpg'.format(1 + n))
                print(filename_dst)
                
                #filename_dst = mat+'{}.jpg'.format(1 + n)
                # Load an XML file
                Thread.thread().file_resolver().append(os.path.dirname(filename_src))
                scene = load_file(filename_src)

                # Rendering
                scene.integrator().render(scene, scene.sensors()[0])

                # Save image
                film = scene.sensors()[0].film()
                bmp = film.bitmap(raw=True)
                bmp.convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, srgb_gamma=True).write(filename_dst)
                del model
