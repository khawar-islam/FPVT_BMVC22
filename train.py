import argparse
import gc
import os
import time
import torchvision.transforms as T
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from torch.nn.parallel import DistributedDataParallel as DDP
from matplotlib import pyplot as plt
from torchvision.utils import save_image

import wandb
# from flopth import flopth
# from nystrom_attention import Nystromformer
from tensorboardX import SummaryWriter
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch.utils.data import DataLoader

from config import get_config
# from get_flops import mha_flops
from util.utils import get_val_data, perform_val, get_time, buffer_val, AverageMeter, train_accuracy
from vit_pytorch import ViT_face
from vit_pytorch import ViTs_face
# from torchsummary import summary
from vit_pytorch.Pit import PiT
# from vit_pytorch.pvt_v2 import ResNetFeatures
from vit_pytorch.focal_transformer import FocalTransformer

# from apex.parallel import DistributedDataParallel as DDP
from vit_pytorch.ours import Ours_FPVT

wandb.init(project='bmvc2022', entity='khawar512')

gc.collect()
torch.cuda.empty_cache()

# Khawar
from vit_pytorch.cait import CaiT
from vit_pytorch.deepvit import DeepViT
# from x_transformers import Encoder

# from vit_pytorch.CvT.cvt import CvT

from vit_pytorch.t2t import T2TViT

# DEIT -FACEBOOK AI RESEARCH
from torchvision.models import resnet50
from vit_pytorch.nest import NesT

# from CrossViT.crossvit import CrossViT
teacher = resnet50(pretrained=True)

from vit_pytorch.CvT.cvt import CvT

from vit_pytorch.pt2 import PyramidPoolingTransformer

# Not working

# Hirarchical
from vit_pytorch.CeiT.ceit import CeiT
#from vit_pytorch.ours import PyramidVisionTransformerV2
from vit_pytorch.pvt_v2 import PyramidVisionTransformerV2
from vit_pytorch.pvt import PyramidVisionTransformer
# Calaculate flops
from ptflops import get_model_complexity_info


# if torch.cuda.is_available():
#     print("dfd")


def need_save(acc, highest_acc):
    do_save = False
    save_cnt = 0
    if acc[0] > 0.98:
        do_save = True
    for i, accuracy in enumerate(acc):
        if accuracy > highest_acc[i]:
            highest_acc[i] = accuracy
            do_save = True
        if i > 0 and accuracy >= highest_acc[i] - 0.002:
            save_cnt += 1
    if save_cnt >= len(acc) * 3 / 4 and acc[0] > 0.99:
        do_save = True
    print("highest_acc:", highest_acc)
    return do_save


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-w", "--workers_id", help="gpu ids or cpu", default='cpu', type=str)
    parser.add_argument("-e", "--epochs", help="training epochs", default=50, type=int)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=1, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [casia, vgg, ms1m, retina, ms1mr, CelebA]",
                        default='casia', type=str)
    parser.add_argument("-n", "--net", help="which network, ['VIT','VITs', 'VITs_Eff','DeepViT','PiT','Levitt', "
                                            "'Comb_ViT', 'Dino_VIT', 'CvT','CrossViT', 'Swin','DEIT','CAIT','T2TViT',"
                                            "'Ours', 'NesT','CCT', 'RVT','CeiT','PVT','P2T']",
                        default='VITs_Eff',
                        type=str)
    parser.add_argument("-head", "--head", help="head type, ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss', "
                                                "'ArcMarginProduct']",
                        default='ArcMarginProduct', type=str, required=False)
    parser.add_argument("-t", "--target", help="verification targets", default='agedb',
                        type=str)
    parser.add_argument("-r", "--resume", help="resume model", default='', type=str)
    parser.add_argument('--outdir', help="output dir", default='./results/VITs_Eff_cosface_s1', type=str)

    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    args = parser.parse_args()

    # ======= hyperparameters & data loaders =======#
    cfg = get_config(args)

    SEED = cfg['SEED']  # random seed for reproduce results
    torch.manual_seed(SEED)

    print(torch.__version__)

    DATA_ROOT = cfg['DATA_ROOT']  # the parent root where your train/val/test data are stored
    EVAL_PATH = cfg['EVAL_PATH']
    WORK_PATH = cfg['WORK_PATH']  # the root to buffer your checkpoints and to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME']
    HEAD_NAME = cfg['HEAD_NAME']  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']

    INPUT_SIZE = cfg['INPUT_SIZE']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE']  # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    NUM_EPOCH = cfg['NUM_EPOCH']

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU']  # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID']  # specify your GPU ids
    print(GPU_ID)
    print('GPU_ID', GPU_ID)
    TARGET = cfg['TARGET']
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    with open(os.path.join(WORK_PATH, 'config.txt'), 'w') as f:
        f.write(str(cfg))
    print("=" * 60)

    writer = SummaryWriter(WORK_PATH)  # writer for buffering intermedium results
    torch.backends.cudnn.benchmark = True
    NUM_CLASS = 526
    # with open(os.path.join(DATA_ROOT, 'property'), 'r') as f:
    #    NUM_CLASS, h, w = [int(i) for i in f.read().split(',')]
    # assert h == INPUT_SIZE[0] and w == INPUT_SIZE[1]

    transform = transforms.Compose([
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]),  # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    # /media/khawar/HDD_Khawar/face_datasets/facescrub_images_112x112/112x112/imgs
    # /data1/khawar/khawar/datasets/facescrub_images_112x112/112x112/imgs
    # /raid/khawar/dataset/CASIA-maxpy-clean
    dataset = datasets.ImageFolder(root='/home/cvpr/Documents/facescrub_images_112x112/112x112', transform=transform)
    trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count(), drop_last=False)
    print(len(trainloader))

    print("Number of Training Classes: {}".format(NUM_CLASS))

    lfw, cfp_ff, cfp_fp, agedb, calfw, cplfw, vgg2_fp, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_issame, calfw_issame, cplfw_issame, vgg2_fp_issame = get_val_data(
        './eval')
    # highest_acc = [0.0 for t in TARGET]

    BACKBONE_DICT = {"PVTV2": Ours_FPVT(
        loss_type=HEAD_NAME,
        GPU_ID=GPU_ID,
        img_size=112,
        depths=[3, 4, 18, 3],
        patch_size=8,
        num_classes=NUM_CLASS,
        in_chans=3
    ), "P2T": PyramidPoolingTransformer(
        loss_type=HEAD_NAME,
        GPU_ID=GPU_ID,
        # patch_size=8,
        # embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        # qkv_bias=True,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        img_size=112,
        depths=[3, 4, 18, 3],
        patch_size=8,
        num_classes=NUM_CLASS,
        in_chans=3
    ), 'VIT': ViT_face(
        loss_type=HEAD_NAME,
        GPU_ID=GPU_ID,
        num_class=NUM_CLASS,
        image_size=112,
        patch_size=8,
        dim=512,
        depth=6,
        heads=6,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1
    ), 'VITs': ViTs_face(
        loss_type=HEAD_NAME,
        GPU_ID=GPU_ID,
        num_class=NUM_CLASS,
        image_size=112,
        patch_size=8,
        ac_patch_size=12,
        pad=4,
        dim=512,
        depth=6,
        heads=6,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1
    ), 'CAiT': CaiT(
        loss_type=HEAD_NAME,
        GPU_ID=GPU_ID,
        num_classes=NUM_CLASS,
        image_size=112,
        patch_size=8,
        ac_patch_size=12,
        cls_depth=2,
        pad=4,
        dim=512,
        depth=3,
        heads=3,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1
    ), 'PiT': PiT(
        loss_type=HEAD_NAME,
        GPU_ID=GPU_ID,
        image_size=112,
        ac_patch_size=12,
        num_classes=NUM_CLASS,
        patch_size=8,
        pad=4,
        dim=64,
        depth=(3, 6, 4),  # list of depths, indicating the number of rounds of each stage before a downsample
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    ),
        "CeiT": CeiT(
            loss_type=HEAD_NAME,
            GPU_ID=GPU_ID,
            image_size=112,
            patch_size=4,
            dim=256,
            depth=20,
            num_classes=NUM_CLASS,
            heads=8,
            dropout=0.1,
            emb_dropout=0.1
        ), 'DeepViT': DeepViT(
            loss_type=HEAD_NAME,
            GPU_ID=GPU_ID,
            image_size=112,
            patch_size=8,
            num_classes=NUM_CLASS,
            ac_patch_size=12,
            pad=4,
            dim=512,
            heads=6,
            depth=6,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1
        ), "NesT": NesT(
            loss_type='CosFace',
            GPU_ID=GPU_ID,
            image_size=112,
            patch_size=14,
            dim=96,
            heads=3,
            num_hierarchies=3,  # number of hierarchies
            block_repeats=(2, 2, 8),  # the number of transformer blocks at each heirarchy, starting from the bottom
            num_classes=NUM_CLASS
        ), "CvT": CvT(
            loss_type='CosFace',
            patch_size=8,
            ac_patch_size=12,
            pad=4,
            GPU_ID=GPU_ID,
            image_size=112,
            in_channels=3,
            num_classes=NUM_CLASS
        ), "T2TViT": T2TViT(
            loss_type=HEAD_NAME,
            GPU_ID=GPU_ID,
            num_classes=NUM_CLASS,
            image_size=224,
            # patch_size=8,
            # ac_patch_size=12,
            # pad=4,
            dim=512,
            depth=5,
            heads=8,
            mlp_dim=512,
            # dropout=0.1,
            # emb_dropout=0.1,
            t2t_layers=((7, 4), (3, 2), (3, 2))
            # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
        ), "FocalTransformer": FocalTransformer(
            img_size=112,
            patch_size=2,
            num_classes=NUM_CLASS,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            drop_path_rate=0.2,
            loss_type=HEAD_NAME,
            GPU_ID=GPU_ID,
            focal_levels=[0, 1, 2, 3],
            expand_sizes=[3, 3, 3, 3],
            expand_layer="all",
            num_heads=[3, 6, 12, 24],
            focal_windows=[7, 5, 3, 1],
            window_size=7,
            # use_conv_embed=False,
            # use_shift=False,
        )
    }
    # print(BACKBONE_DICT[BACKBONE_NAME])
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]

    with torch.cuda.device(0):
        # net = BACKBONE()
        macs, params = get_model_complexity_info(BACKBONE, (3, 112, 112), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # BACKBONE = Recorder(BACKBONE)
    print("=" * 60)
    # print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)

    LOSS = nn.CrossEntropyLoss()

    # embed()
    OPTIMIZER = create_optimizer(args, BACKBONE)
    print("=" * 60)
    print(OPTIMIZER)
    print("Optimizer Generated")
    print("=" * 60)
    lr_scheduler, _ = create_scheduler(args, OPTIMIZER)

    # optionally resume from a checkpoint
    if BACKBONE_RESUME_ROOT:
        print("=" * 60)
        print(BACKBONE_RESUME_ROOT)
        if os.path.isfile(BACKBONE_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT), strict=False)
        else:
            print("No Checkpoint Found at '{}' . Please Have a Check or Continue to Train from Scratch".format(
                BACKBONE_RESUME_ROOT))
        print("=" * 60)

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids=GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:
        # single-GPU setting
        print("Using Single GPU")
        print(DEVICE)
        BACKBONE = BACKBONE.to(DEVICE)

    config = wandb.config
    config.learning_rate = 3e-4

    # ======= train & validation & save checkpoint =======#
    DISP_FREQ = 10  # frequency to display training loss & acc
    VER_FREQ = 20

    batch = 0  # batch index

    losses = AverageMeter()
    top1 = AverageMeter()

    # resnet_features = ResNetFeatures().to(DEVICE)

    # BACKBONE = ResNet34(
    #     input_channels=3,
    #     classes=classes).to(device)

    BACKBONE.train()  # set to training mode

    wandb.watch(BACKBONE)

    for epoch in range(NUM_EPOCH):  # start training process

        lr_scheduler.step(epoch)

        last_time = time.time()

        for inputs, labels in iter(trainloader):

            # compute output
            inputs = inputs.to(DEVICE)
            #print(inputs.size)

            labels = labels.to(DEVICE).long()

            outputs, emb = BACKBONE(inputs.float(), labels)
            loss = LOSS(outputs, labels)

            #print("outputs", outputs.data.size())
            # measure accuracy and record loss

            prec1 = train_accuracy(outputs.data, labels, topk=(1,))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))

            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

            wandb.log({"loss": losses.avg, "epoch": NUM_EPOCH, 'acc': top1.avg, 'batch': batch})

            # dispaly training loss & acc every DISP_FREQ (buffer for visualization)
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                epoch_loss = losses.avg
                epoch_acc = top1.avg
                writer.add_scalar("Training/Training_Loss", epoch_loss, batch + 1)
                writer.add_scalar("Training/Training_Accuracy", epoch_acc, batch + 1)

                batch_time = time.time() - last_time
                last_time = time.time()

                print('Epoch {} Batch {}\t'
                      'Speed: {speed:.2f} samples/s\t'
                      'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch + 1, batch + 1, speed=inputs.size(0) * DISP_FREQ / float(batch_time),
                    loss=losses, top1=top1))

                # print("=" * 60)
                losses = AverageMeter()
                top1 = AverageMeter()

            if ((
                        batch + 1) % VER_FREQ == 0) and batch != 0:  # perform validation & save checkpoints (buffer for visualization)
                for params in OPTIMIZER.param_groups:
                    lr = params['lr']
                    break
                print("Learning rate %f" % lr)
                print("Perform Evaluation on", TARGET, ", and Save Checkpoints...")
                acc = []
                accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
                                                                              BATCH_SIZE,
                                                                              BACKBONE, lfw, lfw_issame)
                buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, epoch + 1)

                accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(MULTI_GPU, DEVICE,
                                                                                       EMBEDDING_SIZE,
                                                                                       BATCH_SIZE, BACKBONE, cfp_ff,
                                                                                       cfp_ff_issame)
                buffer_val(writer, "CFP_FF", accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff, epoch + 1)
                accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(MULTI_GPU, DEVICE,
                                                                                       EMBEDDING_SIZE,
                                                                                       BATCH_SIZE, BACKBONE, cfp_fp,
                                                                                       cfp_fp_issame)
                buffer_val(writer, "CFP_FP", accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp, epoch + 1)

                accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
                                                                                    BATCH_SIZE, BACKBONE, agedb,
                                                                                    agedb_issame)
                buffer_val(writer, "AgeDB", accuracy_agedb, best_threshold_agedb, roc_curve_agedb, epoch + 1)

                accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
                                                                                    BATCH_SIZE, BACKBONE, calfw,
                                                                                    calfw_issame)
                buffer_val(writer, "CALFW", accuracy_calfw, best_threshold_calfw, roc_curve_calfw, epoch + 1)
                accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
                                                                                    BATCH_SIZE, BACKBONE, cplfw,
                                                                                    cplfw_issame)
                buffer_val(writer, "CPLFW", accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw, epoch + 1)
                accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val(MULTI_GPU, DEVICE,
                                                                                          EMBEDDING_SIZE,
                                                                                          BATCH_SIZE, BACKBONE, vgg2_fp,
                                                                                          vgg2_fp_issame)
                buffer_val(writer, "VGGFace2_FP", accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp,
                           epoch + 1)
                print(
                    "Epoch {}/{}, Evaluation: LFW Acc: {}, CFP_FF Acc: {}, CFP_FP Acc: {}, AgeDB Acc: {}, CALFW Acc: {}, "
                    "CPLFW Acc: {}, VGG2_FP Acc: {}".format(epoch + 1, NUM_EPOCH, accuracy_lfw, accuracy_cfp_ff,
                                                            accuracy_cfp_fp, accuracy_agedb, accuracy_calfw,
                                                            accuracy_cplfw, accuracy_vgg2_fp))

                acc.append(accuracy_agedb)
                # wandb.log({"loss": loss.item(), "epoch": NUM_EPOCH, 'acc': accuracy, 'batch': batch})
                # save checkpoints per epoch

                # if need_save(acc):
                if MULTI_GPU:
                    torch.save(BACKBONE.module.state_dict(), os.path.join(WORK_PATH,
                                                                          "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                                                                              BACKBONE_NAME, epoch + 1, batch + 1,
                                                                              get_time())))
                else:
                    torch.save(BACKBONE.state_dict(), os.path.join(WORK_PATH,
                                                                   "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                                                                       BACKBONE_NAME, epoch + 1, batch + 1,
                                                                       get_time())))
                BACKBONE.train()  # set to training mode

            batch += 1  # batch index

