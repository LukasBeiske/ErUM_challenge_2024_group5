import torch 
import torchvision as tv


def get_convert_mikly_way():
    !wget https://d2pn8kiwq2w21t.cloudfront.net/original_images/jpegPIA10748.jpg

    milky_way = tv.io.read_image("./jpegPIA10748.jpg").permute(1,2,0).to(torch.float32)
    return milky_way / 255

def jax_to_torch(jax_tensor):
    return torch.from_numpy(np.asarray(jax_tensor.copy()))