import torch

features = torch.randn(1,160,256,256)
gt = torch.randn(1,2,256,256)
gt[:,0,:,:] = 0
gt[:,1,:,:] = 1
gt[:,1,:,0] = 0

# extracting class 1 samples
# if gt == 1 -> save features
gt_map = gt[0,1,:,:].squeeze()
gt_map = torch.where(gt_map == 1, True, False) * 1 # 256,256
pix_num = gt_map.sum()
gt_map = gt_map.repeat(160,1,1).unsqueeze(0) # 1,160,256,256
features = features * gt_map # average pooling
features = (features.sum(dim=(2,3)) / pix_num).squeeze()
print(features.shape)

# get index where gt_map = True

# where gt_map True, get feature
# get index
#gt_map = gt_map.nonzero()

"""
        # extracting class 1 samples
        h,w = features.shape[2], features.shape[3]
        for h_index in range(h) :
            for w_index in range(w) :
                feat = features[0,:,h_index,w_index].squeeze()
                label = gt[0,1,h_index,w_index].squeeze().item()
                if label == 1 :
                    if accelerator.is_main_process:
                        torch.save(feat,
                                   os.path.join(feature_save_dir, f'feature_{step_i}_{h_index}_{w_index}.pt'))
"""