import sys
sys.path.append(".")


from omegaconf import OmegaConf
config_path = "logs/2020-11-09T13-31-51_sflckr/configs/2020-11-09T13-31-51-project.yaml"
config = OmegaConf.load(config_path)
import yaml
print(yaml.dump(OmegaConf.to_container(config)))

from taming.models.cond_transformer import Net2NetTransformer
model = Net2NetTransformer(**config.model.params)



import torch
ckpt_path = "logs/2020-11-09T13-31-51_sflckr/checkpoints/last.ckpt"
sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
missing, unexpected = model.load_state_dict(sd, strict=False)



model.cuda().eval()
torch.set_grad_enabled(False)




from PIL import Image
import numpy as np
segmentation_path = "data/sflckr_segmentations/norway/25735082181_999927fe5a_b.png"
segmentation = Image.open(segmentation_path)
segmentation = np.array(segmentation)
segmentation = np.eye(182)[segmentation]
segmentation = torch.tensor(segmentation.transpose(2,0,1)[None]).to(dtype=torch.float32, device=model.device)





def show_segmentation(s):
    s = s.detach().cpu().numpy().transpose(0,2,3,1)[0,:,:,None,:]
    colorize = np.random.RandomState(1).randn(1,1,s.shape[-1],3)
    colorize = colorize / colorize.sum(axis=2, keepdims=True)
    s = s@colorize
    s = s[...,0,:]
    s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
    s = Image.fromarray(s)
    display(s)

show_segmentation(segmentation)