import argparse
import os
import torch
import numpy as np
from PIL import Image
from itertools import islice
from torch import autocast
from contextlib import nullcontext
import torchvision
import matplotlib.pyplot as plt
from PIL import ImageOps
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import Resize

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)


def compose_images(img_p, ref_p, mask_pil):
    '''
        The background image is img_p
        The logo image that is going to be composited on top
        The mask showing the location of this compositing method
    '''
    # Find the bounding box of the non-zero regions in the mask image
    bbox = mask_pil.getbbox()

    # Calculate the coordinates of the bounding box
    left = bbox[0]
    top = bbox[1]
    right = bbox[2]
    bottom = bbox[3]

    # Calculate the center of the bounding box
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2

    # Calculate the height and width of the bounding box
    width = right - left
    height = bottom - top

    padding = 25

    ref_p_comp = ImageOps.contain(ref_p, (width+padding, height+padding))
    img_p_comp = img_p.copy()
    img_masked_comp = np.array(img_p.copy())

    # Create a new image with RGBA mode (transparency)
    ref_p_comp_rgba = Image.new("RGBA", (width+padding, height+padding), (255, 255, 255, 0))

    # Paste the resized reference image onto the RGBA image
    ref_p_comp_rgba.paste(ref_p_comp, (0, 0), ref_p_comp)

    # Overwrite part of the base image with the image to paste at the specified position
    img_p_comp.paste(ref_p_comp_rgba, (left - (width // 20), top + (height // 20)), ref_p_comp_rgba)

    img_masked_comp[top:bottom, left:right] = (0, 0, 0, 255)  # Set to black (RGB: 0, 0, 0)

    # Convert the NumPy array back to a Pillow image
    masked_img_p_comp = Image.fromarray(img_masked_comp)

    return img_p_comp, masked_img_p_comp


def main():
    parser = argparse.ArgumentParser(prefix_chars='--')

    parser.add_argument(
        "--outdir",
        type=str,
        # nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        default=False,
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--n_imgs",
        type=int,
        default=100,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given reference image. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="evaluate at this precision",
        default="./PaintbyExample/Test_Bench_Logo/background_image/000000000003.jpeg"
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        help="evaluate at this precision",
        default="./PaintbyExample/Test_Bench_Logo/masks/000000000003_mask.jpeg"
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        help="evaluate at this precision",
        default="./PaintbyExample/Test_Bench_Logo/logo/000000000003_ref.png"
    )

    # seed_everything(opt.seed)

    opt, unknown = parser.parse_known_args()

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # batch_size = opt.n_samples
    # n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    sample_path = os.path.join(outpath, "source")
    result_path = os.path.join(outpath, "results")
    grid_path = os.path.join(outpath, "grid")
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(grid_path, exist_ok=True)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with precision_scope("cpu"):

        filename = os.path.basename(opt.image_path)
        logo_filename = os.path.basename(opt.reference_path)
        img_p = Image.open(opt.image_path).convert("RGBA")
        image_tensor = get_tensor()(img_p.convert("RGB"))
        image_tensor = image_tensor.unsqueeze(0)
        _, h, w = image_tensor.squeeze().shape
        ref_p = Image.open(opt.reference_path).convert("RGBA")
        ref_tensor = get_tensor_clip()(ref_p.convert("RGB"))
        ref_tensor = ref_tensor.unsqueeze(0)
        mask_pil = Image.open(opt.mask_path).convert("L")
        mask = np.array(mask_pil)[None, None]
        mask = 1 - mask.astype(np.float32)/255.0
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask_tensor = torch.from_numpy(mask)

        img_p_comp, masked_p_comp = compose_images(img_p, ref_p, mask_pil)

        inpaint_image = get_tensor()(img_p_comp.convert("RGB")).to(device)
        masked_image = get_tensor()(masked_p_comp.convert("RGB")).to(device)

        test_model_kwargs = {}
        test_model_kwargs['inpaint_mask'] = mask_tensor.to(device)
        test_model_kwargs['inpaint_image'] = inpaint_image.to(device)
        ref_tensor = ref_tensor.to(device)

        inpaint_mask = test_model_kwargs['inpaint_mask']

        def un_norm(x):
            return (x+1.0)/2.0

        def un_norm_clip(x):
            x[0, :, :] = x[0, :, :] * 0.26862954 + 0.48145466
            x[1, :, :] = x[1, :, :] * 0.26130258 + 0.4578275
            x[2, :, :] = x[2, :, :] * 0.27577711 + 0.40821073
            return x

        if not opt.skip_save:
            all_img = []
            all_img.append(un_norm(image_tensor.squeeze()).cpu())
            all_img.append(un_norm(masked_image).cpu())
            ref_img = ref_tensor
            ref_img = Resize([opt.H, opt.W])(ref_img)
            all_img.append(un_norm_clip(ref_img.squeeze()).cpu())
            all_img.append(un_norm(inpaint_image.cpu()))
            grid = torch.stack(all_img, 0)
            grid = make_grid(grid)
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            img = Image.fromarray(grid.astype(np.uint8))
            img.save(os.path.join(grid_path, 'grid-'+filename[:-4]+'_'+str(opt.seed)+f'_{logo_filename[:-4]}.png'))

            print("Raw File: %s" % filename)
            print("SAVING FILE TO: %s" % os.path.join(result_path, filename[:-4]+"_"+opt.reference_path.split('/')[-1]))
            img_p_comp.save(os.path.join(result_path, filename[:-4]+"_"+opt.reference_path.split('/')[-1]))

            mask_pil.save(os.path.join(sample_path, filename[:-4]+'_'+str(opt.seed)+f"_mask_{logo_filename[:-4]}.png"))

            GT_img = 255. * rearrange(all_img[0], 'c h w -> h w c').numpy()
            GT_img = Image.fromarray(GT_img.astype(np.uint8))
            GT_img.save(os.path.join(sample_path, filename[:-4]+'_'+str(opt.seed)+f"_GT_{logo_filename[:-4]}.png"))
            inpaint_img = 255.*rearrange(all_img[1], 'c h w -> h w c').numpy()
            inpaint_img = Image.fromarray(inpaint_img.astype(np.uint8))
            inpaint_img.save(os.path.join(sample_path, filename[:-4]+'_'+str(opt.seed)+f"_inpaint_{logo_filename[:-4]}.png"))
            ref_img = 255.*rearrange(all_img[2], 'c h w -> h w c').numpy()
            ref_img = Image.fromarray(ref_img.astype(np.uint8))
            ref_img.save(os.path.join(sample_path, filename[:-4]+'_'+str(opt.seed)+f"_ref_{logo_filename[:-4]}.png"))

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n \nEnjoy.")


if __name__ == "__main__":
    main()
