from os.path import join
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import data,time,os

opt = TrainOptions().parse()

cudnn.benchmark = True

opt.display_freq = 10

if opt.debug:
    opt.display_id = 1
    opt.display_freq = 20
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 100
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True

# modify the following code to
datadir = '/gdata1/zhuyr/Deref/training_data/'

datadir_syn = join(datadir, 'JPEGImages')
datadir_real = join(datadir, 'real_train')

train_dataset = datasets.CEILDataset(
    datadir_syn, read_fns('/ghome/zhuyr/ADeref_two1/VOC2012_224_train_png.txt'), size=opt.max_dataset_size, enable_transforms=True,
    low_sigma=opt.low_sigma, high_sigma=opt.high_sigma,
    low_gamma=opt.low_gamma, high_gamma=opt.high_gamma)

train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True)

train_dataset_fusion = datasets.FusionDataset([train_dataset, train_dataset_real], [0.7, 0.3])

train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataset_nature20 = datasets.CEILTestDataset(join(datadir, 'nature20'))
eval_dataset_real20 = datasets.CEILTestDataset(join(datadir, 'real20'))


eval_dataloader_nature20 = datasets.DataLoader(eval_dataset_nature20, batch_size=1, shuffle=False,
                                                    num_workers=opt.nThreads, pin_memory=True)
eval_dataloader_real20 = datasets.DataLoader(eval_dataset_real20, batch_size=1, shuffle=False,
                                                    num_workers=opt.nThreads, pin_memory=True)
eval_dataset_wild55 = datasets.CEILTestDataset(join(datadir, 'wild55'))
eval_dataloader_wild55 = datasets.DataLoader(eval_dataset_wild55, batch_size=1, shuffle=False,
                                               num_workers=0, pin_memory=True)
eval_dataset_solid200 = datasets.CEILTestDataset(join(datadir, 'solid200'))
eval_dataloader_solid200 = datasets.DataLoader(eval_dataset_solid200, batch_size=1, shuffle=False,
                                               num_workers=0, pin_memory=True)
eval_dataset_postcard199 = datasets.CEILTestDataset(join(datadir, 'postcard199'))
eval_dataloader_postcard199 = datasets.DataLoader(eval_dataset_postcard199, batch_size=1, shuffle=False,
                                                  num_workers=0, pin_memory=True)

engine = Engine(opt)

"""Main Loop"""
result_dir  = '/ghome/zhuyr/ADeref/results/'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
st = time.time()
engine.eval(eval_dataloader_nature20, dataset_name='testdata_nature20',savedir=join(result_dir, 'testdata_nature20'))
# engine.eval(eval_dataloader_real20, dataset_name='testdata_real20')
# engine.eval(eval_dataloader_wild55, dataset_name='testdata_wild55')
# engine.eval(eval_dataloader_postcard199, dataset_name='testdata_postcard199')
# engine.eval(eval_dataloader_solid200, dataset_name='testdata_solid200')