import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from lib.model import CFANet
from utils.dataloader import get_loader,test_dataset
#from utils.eva_funcs import eval_Smeasure,eval_mae,numpy2tensor
from model import call_model_package
import cv2
from PIL import Image
from utils import prepare_dtype
from utils.accelerator_utils import prepare_accelerator
from model.segmentation_unet import SemanticSeg_2
def main(args):

    print(f' step 0. accelerator')
    accelerator = prepare_accelerator(args)

    print(f' step 1. make model')
    print(f' (1) stable diffusion model')
    weight_dtype, save_dtype = prepare_dtype(args)
    condition_model, vae, unet, network = call_model_package(args, weight_dtype, accelerator)
    network.load_state_dict(torch.load(args.network_weights))
    print(f' (2) segmentation head')
    segmentation_head = SemanticSeg_2(n_classes=args.n_classes, mask_res=args.mask_res, )
    segmentation_head.load_state_dict(torch.load(args.segmentation_head_weights))

    print(f' step 2. check data path')
    for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:

        # [1] data_path here
        data_path = os.path.join(args.base_path, _data_name)

        # [2] save_path
        save_path = os.path.join(args.save_base, _data_name)  # './result_map/PolypPVT/{}/'.format()
        os.makedirs(save_path, exist_ok=True)

        os.makedirs(save_path, exist_ok=True)

        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)

        test_loader = test_dataset(image_root, gt_root, args.testsize)


        """
        for i in range(test_loader.size):


            image, gt, name, org_img = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            _, _, _, res = model(image)

            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            h, w = res.shape
            # expand rgb_image to [500,574,3]
            res = np.expand_dims(res, axis=2).repeat(3, axis=2)
            res_pil = Image.fromarray((res * 255).astype(np.uint8))

            # [3] saving
            org_img = org_img.resize((w, h), Image.BILINEAR)

            # [3.2] gt
            gt_np = np.array(gt) * 255
            gt_pil = Image.fromarray(gt_np.astype(np.uint8)).resize((w, h))

            # [3.4]
            merged_image = Image.blend(org_img, res_pil, 0.4)

            # [8] merging all image
            total_img = Image.new('RGB', (w*4, h))
            total_img.paste(org_img, (0,0))
            total_img.paste(gt_pil, (w,0))
            total_img.paste(res_pil, (w*2,0))
            total_img.paste(merged_image, (w*3,0))
            total_img.save(os.path.join(save_path, f'{name}'))

        print(_data_name, 'Finish!')
        """


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352,
                        help='testing size')
    parser.add_argument('--pth_path', type=str,
                        default='./checkpoint/CFANet.pth')
    parser.add_argument('--base_path', type=str,
                        default=r'/home/dreamyou070/MyData/anomaly_detection/medical/leader_polyp/Pranet/test')
    parser.add_argument('--save_base', type=str, default='./result_sy')
    args = parser.parse_args()
    main(args)