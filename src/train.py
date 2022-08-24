import torch 
from torch.utils.data import DataLoader
import numpy as np
from dataset import BTFDataset
import utils
from models import NeuBTF
import os
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt


ablation_materials = [
    "leather11",
    "leather08",
    "carpet11",
    "carpet07",
    "fabric01",
]

ablation_config = [
    [False, False, False],
    [True, False, False],
    [False, True, False],
    [True, True, False],
    [False, False, True],
    [True, False, True],
    [False, True, True],
    [True, True, True],
]

train_config={
    "ds_name": "leather11",
    "total_steps": 15,
    "batch_size":1,
    "siren": True,
    "shared": False,
    "concat": True,
    "hidden_layers": 2,
    "hidden_ch": 32,
    "embeddings_ch": 7,
    "levels": 4,
    "steps_till_summary": 1024,
}

def get_sample(model: NeuBTF, ds: BTFDataset, idx: int,):
    with torch.no_grad():
        gt, co, l, wi, wo = ds.get_sample(idx)

        co, l, wi, wo = (co.cuda(), l.cuda(), wi.cuda(), wo.cuda())
        
        model_output, neu_tex, offset, neu_depth = model(co.unsqueeze(0),l.unsqueeze(0), wi.unsqueeze(0), wo.unsqueeze(0), False)
 
        neu_tex = neu_tex.cpu().detach().squeeze().numpy()
        
        neu_tex = neu_tex[...,:4]
        
        offset_vis = np.zeros((400,400,3), dtype=np.float32) 
        offset_vis[...,:2] = offset.detach().squeeze().cpu().numpy()
        neu_depth = neu_depth.detach().squeeze().cpu().numpy()
    
        gt = utils.gamma_correction(gt)
        model_output = utils.gamma_correction(model_output.cpu().detach().numpy()[0])
        
   
    return gt, model_output, neu_tex, offset_vis, neu_depth,

def show_sample(model: NeuBTF, ds: BTFDataset, idx: int,):
    gt, model_output, neu_tex, offset_vis, neu_depth, = get_sample(model, ds, idx,)
    ax1 = plt.subplot(1, 5, 1)
    ax1.imshow(gt)
    ax2 = plt.subplot(1, 5, 2)
    ax2.imshow(model_output)
    ax3 = plt.subplot(1, 5, 3)
    ax3.imshow(neu_tex)
    ax4 = plt.subplot(1, 5, 4)
    ax4.imshow(offset_vis)
    ax5 = plt.subplot(1, 5, 5)
    ax5.imshow(neu_depth)
    plt.show()


def init_model(train_config, cuda=True):
    model = NeuBTF(
        400, 
        train_config["embeddings_ch"], 
        train_config["hidden_ch"], 
        train_config["hidden_layers"], 
        3, 
        outermost_linear=True, 
        siren=train_config["siren"], 
        shared=train_config["shared"], 
        concat=train_config["concat"] , 
        n_levels=train_config["levels"]) # in, hidden,#hidden, out
    if cuda:
        model.cuda()
    
    print(model)
    return model

def train(config, model=None, train_dataloader=None, btf_ds=None, save=False):
    if btf_ds == None:
        ds_dir = os.path.join("..","dataset","UBO2014")
        ds_path = os.path.join(ds_dir,".".join((config["ds_name"], "btf")) )
        btf_ds = BTFDataset(ds_path, train_size=1024)
        train_dataloader = DataLoader(btf_ds, batch_size=config["batch_size"], num_workers=0)
    
    if train_dataloader == None:
        train_dataloader = DataLoader(btf_ds, batch_size=config["batch_size"], num_workers=0)

    if model == None:
        model = init_model(config)
        
    total_steps = config["total_steps"]
    optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
    count = 0
    loss_history = list()

    for step in tqdm(range(total_steps)):
        batch_bar = tqdm(train_dataloader)
        epoch_history = list()
        for n, (gt, co, l, wi, wo) in enumerate(batch_bar):
            gt, co, l, wi, wo = (gt.cuda(), co.cuda(), l.cuda(), wi.cuda(), wo.cuda())

            model_output, _, _, _,  = model(co, l, wi, wo, False)
            
            gt = gt.cuda()
            
            l1_l = torch.nn.functional.l1_loss(model_output, gt)
            mse_l = torch.nn.functional.mse_loss(model_output, gt)
            
            total_loss = l1_l + 10. * mse_l

            optim.zero_grad()
            total_loss.backward()
            optim.step()

           
                
            if (count + 1) % config["steps_till_summary"] == 0:
                show_sample(model, btf_ds, 1)
            
            total_loss = total_loss.detach().cpu().numpy()
            epoch_history.append(total_loss)
            batch_bar.set_description("Loss: {:.4f}".format(total_loss))
    
            count += 1
        loss_history.append(epoch_history)
        #model.fuse_blur(step, total_steps)
    if save:
        torch.save(model.state_dict(), config["out_path"])
    return model, np.array(loss_history)

if __name__ == "__main__":
    
    models_path = "ubo2014"
    render_dir = os.path.join("rendering_final", models_path)
    models_path = os.path.join("models_test", models_path)
    
    if not os.path.isdir(models_path):
        os.mkdir(models_path)

    

    for mat in tqdm(ablation_materials):
        batch_size = train_config["batch_size"]
        train_config["ds_name"] = mat
        ds_dir = os.path.join("dataset","UBO2014")
        ds_path = os.path.join(ds_dir,".".join((train_config["ds_name"], "btf")) )
        render_path = os.path.join(render_dir, mat)
        if not os.path.isdir(render_path):
            os.mkdir(render_path)

        btf_ds = BTFDataset(ds_path,data_size=1024)
        train_dataloader = DataLoader(btf_ds, batch_size=batch_size, num_workers=8)

        
        
        save_path = os.path.join(models_path, mat)
            
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
            
        for n, (siren, shared, concat) in enumerate(tqdm(ablation_config)):
            train_config["siren"] = siren
            train_config["shared"] = shared
            train_config["concat"] = concat

            if concat:
                train_config["embeddings_ch"] = 4
            else:
                train_config["embeddings_ch"] = 7
            print('running configuration {}: {} {} {}'.format(n, siren, shared, concat))

            train_config["out_path"] = os.path.join(save_path,'{}.pth'.format(1 + n))
            train_config["render_path"] = os.path.join(render_path,'{}.jpg'.format(1 + n))
            out_model, history = train(train_config, train_dataloader, btf_ds, True)
            del out_model, history
            
        del btf_ds, train_dataloader