import torch
import numpy as np
from PIL import Image
import math

def pad_and_resize(im, new_width=768, new_height=1024, pad_color=(255, 255, 255), mode=Image.LANCZOS):
    """
    Pad and resize image while maintaining aspect ratio
    
    Args:
        im (PIL.Image): Input image
        new_width (int): Target width
        new_height (int): Target height
        pad_color (tuple): Padding color (R,G,B)
        mode: Resampling mode
    
    Returns:
        PIL.Image: Processed image
        int: Horizontal padding amount
        int: Vertical padding amount
    """
    old_width, old_height = im.size
    
    ratio_w = new_width / old_width
    ratio_h = new_height / old_height
    if ratio_w < ratio_h:
        new_size = (new_width, round(old_height * ratio_w))
    else:
        new_size = (round(old_width * ratio_h), new_height)
    
    im_resized = im.resize(new_size, mode)

    pad_w = math.ceil((new_width - im_resized.width) / 2)
    pad_h = math.ceil((new_height - im_resized.height) / 2)

    new_im = Image.new('RGB', (new_width, new_height), pad_color)
    new_im.paste(im_resized, (pad_w, pad_h))

    return new_im, pad_w, pad_h

def unpad_and_resize(padded_im, pad_w, pad_h, original_width, original_height):
    """
    Remove padding and restore image to original size
    
    Args:
        padded_im (PIL.Image): Padded image
        pad_w (int): Horizontal padding amount
        pad_h (int): Vertical padding amount
        original_width (int): Original width
        original_height (int): Original height
    
    Returns:
        PIL.Image: Processed image
    """
    width, height = padded_im.size
    
    left = pad_w
    top = pad_h
    right = width - pad_w
    bottom = height - pad_h
    
    cropped_im = padded_im.crop((left, top, right, bottom))
    resized_im = cropped_im.resize((original_width, original_height), Image.LANCZOS)

    return resized_im

def resize_image(img, target_size=768):
    """
    Resize image while maintaining aspect ratio
    
    Args:
        img (PIL.Image): Input image
        target_size (int): Target size
    
    Returns:
        PIL.Image: Resized image
    """
    width, height = img.size
    
    if width < height:
        scale = target_size / width
    else:
        scale = target_size / height
    
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_img

def tensor_to_pil(img_tensor, batch_index=0):
    # Convert tensor of shape [batch_size, channels, height, width] at the batch_index to PIL Image
    img_tensor = img_tensor[batch_index].unsqueeze(0)
    i = 255. * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img

def batch_tensor_to_pil(img_tensor):
    # Convert tensor of shape [batch_size, channels, height, width] to a list of PIL Images
    return [tensor_to_pil(img_tensor, i) for i in range(img_tensor.shape[0])]


def pil_to_tensor(image):
    # Takes a PIL image and returns a tensor of shape [1, height, width, channels]
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    if len(image.shape) == 3:  # If the image is grayscale, add a channel dimension
        image = image.unsqueeze(-1)
    return image


def batched_pil_to_tensor(images):
    # Takes a list of PIL images and returns a tensor of shape [batch_size, height, width, channels]
    return torch.cat([pil_to_tensor(image) for image in images], dim=0)
