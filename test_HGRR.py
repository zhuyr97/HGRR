from os.path import join
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import torch,os
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.image as img
from models.arch.default3 import DRNet3
import models.losses as losses
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datadir ='/gdata1/zhuyr/Deref/training_data/' #'/gdata1/zhuyr/Deref/training_data/'
SAVE_PATH = '/gdata1/zhuyr/Deref/checkpoints/errnet_DB_boosting2/errnet_063_00499716.pt'
eval_dataset_nature20 = datasets.CEILTestDataset(join(datadir, 'nature20'))
eval_dataloader_nature20 = datasets.DataLoader(eval_dataset_nature20, batch_size=1, shuffle=False,
                                                    num_workers=0, pin_memory=True)
# trans_eval = transforms.Compose(
#         [transforms.ToTensor()
#         ])
# eval_dataset_nature20_ =Deref_dataset_test(root_in=datadir +'/nature20/blended/'
#                                            ,root_label=datadir +'/nature20/transmission_layer/',transform=trans_eval)  #
# eval_dataloader_nature20_ = datasets.DataLoader(eval_dataset_nature20_, batch_size=1, shuffle=False,
#                                                     num_workers=0, pin_memory=True)

for i, data in enumerate(eval_dataloader_nature20, 0):
    #print('name', name, name[0], '-' * 30, i)
    #{'input': M, 'target_t': B, 'fn': fn, 'real': True, 'target_r': B}
    if i == 0:
        print(data['input'],data['input'].size(), data['fn'][0])


if __name__ == '__main__':
    vgg = losses.Vgg19(requires_grad=False).to(device)

    net =DRNet3(in_channels=1475, out_channels=3, n_feats= 128, n_resblocks=4, se_reduction=8, bottom_kernel_size=1,
                norm=None,res_scale=0.1, pyramid=True,last_sigmoid=False).to(device)

    print('#generator parameters:',sum(param.numel() for param in net.parameters()))
    #training
    net.load_state_dict(torch.load(SAVE_PATH))

    net.eval()
    with torch.no_grad():
        psnr = 0
        rmse = 0
        shadowfree_rmse =0
        shadow_rmse =0
        for i, data in enumerate(eval_dataloader_nature20, 0):
            # print('name', name, name[0], '-' * 30, i)
            # {'input': M, 'target_t': B, 'fn': fn, 'real': True, 'target_r': B}
            print(data['input'], data['input'].size(), data['fn'][0])
            print('name',data['fn'][0],'-'*30,i)
            inputs = Variable(data['input']).to(device)
            hypercolumn = vgg(inputs)
            _, C, H, W = inputs.shape
            hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for
                           feature in
                           hypercolumn]
            input_i = [inputs]
            input_i.extend(hypercolumn)
            input_i = torch.cat(input_i, dim=1)

            out_eval,out_H = net(input_i)

            out_eval = torch.clamp(out_eval, 0., 1.)
            out_eval_np = np.squeeze(out_eval.cpu().numpy())
            out_eval_np_ = out_eval_np.transpose((1,2,0))

            out_H_eval = torch.clamp(out_H, 0., 1.)
            out_H_eval_np = np.squeeze(out_H_eval.cpu().numpy())
            out_H_eval_np_ = out_H_eval_np.transpose((1,2,0))

            #img.imsave(test_results_path + name[0],np.uint8(out_eval_np_ * 255.))

# class Deref_dataset_test(Dataset):
#     def __init__(self,root_in,root_label,transform =None):
#         super(Deref_dataset_test,self).__init__()
#         #in_imgs
#         in_files = os.listdir(root_in)
#         self.imgs_in = [os.path.join(root_in, k) for k in in_files]
#         #gt_imgs
#         gt_files = os.listdir(root_label)
#         self.imgs_gt = [os.path.join(root_label, k) for k in gt_files]
#
#         self.transform = transform
#     def __getitem__(self, index):
#         in_img_path = self.imgs_in[index]
#         img_name =in_img_path.split('/')[-1]
#         in_img = Image.open(in_img_path)
#         gt_img_path = self.imgs_gt[index]
#         gt_img = Image.open(gt_img_path)
#         if self.transform:
#             data_IN = self.transform(in_img)
#             data_GT = self.transform(gt_img)
#         else:
#             data_IN =np.asarray(in_img)
#             data_IN = torch.from_numpy(data_IN)
#             data_GT = np.asarray(gt_img)
#             data_GT = torch.from_numpy(data_GT)
#         return data_IN,data_GT,img_name
#     def __len__(self):
#         return len(self.imgs_in)
