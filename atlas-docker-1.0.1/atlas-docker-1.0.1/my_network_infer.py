import argparse
import nibabel as nib
import numpy as np
from unet import UNet3D
import torch
from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser(description='Image segmentation')
    parser.add_argument('-i', '--input', type=str, help='input image filename (NIfTI)', required=True)
    parser.add_argument('-o', '--output', type=str, help='output image filename (NIfTI)', required=True)
    parser.add_argument('-s', '--state_dict', help='model state_dict filename to load', required=True)
    parser.add_argument('-v', '--verbose', action='store_true')  # on/off flag
    args = parser.parse_args()

    model = UNet3D(in_channel=1, out_channel=2)
    # If DataParallel is used during the training, the model is stored in 'module'
    state_dict = torch.load(args.state_dict, map_location=torch.device('cpu'))
    # Let's check if 'module' exists and if it is the case create a new OrderedDict that does not contain 'module'.
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]  # remove 'module.'
            new_state_dict[name] = v
        else:
            name = k
            new_state_dict[name] = v
    # Load params
    model.load_state_dict(new_state_dict)
    model.eval()
    if args.verbose:
        print(model)
    # Load image
    image = nib.load(args.input)
    np_img = image.get_fdata().astype(np.float32)
    # Because a batch size is needed !
    np_img = np_img.reshape(1, 1, np_img.shape[0], np_img.shape[1], np_img.shape[2])
    torch_image = torch.from_numpy(np_img).float()
    torch_image = torch_image.permute(0, 1, 4, 3, 2)
    # Compute output
    pred = model(torch_image)
    current_pred = torch.argmax(pred[0, :, :, :, :], dim=0).float()
    current_pred = current_pred.permute(2, 1, 0)
    current_pred = torch.squeeze(current_pred)
    current_pred = current_pred.cpu().detach().numpy()
    current_pred = nib.Nifti1Image(current_pred, affine=image.affine)
    nib.save(current_pred, args.output)


if __name__ == '__main__':
    main()
