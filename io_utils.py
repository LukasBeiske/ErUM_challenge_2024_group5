import torch 
import torchvision as tv
import matplotlib.pyplot as plt

def get_convert_mikly_way():
    !wget https://d2pn8kiwq2w21t.cloudfront.net/original_images/jpegPIA10748.jpg

    milky_way = tv.io.read_image("./jpegPIA10748.jpg")
    milky_way = tv.transforms.Resize(size=64)(milky_way)    
    milky_way = milky_way.permute(1,2,0).to(torch.float32)
    return milky_way / 255

plt.imshow(milky_way.detach().cpu().numpy())

def jax_to_torch(jax_tensor):
    return torch.from_numpy(np.asarray(np.copy(jax_tensor)))