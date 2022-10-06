import torch, os
import yaml
from IPython import embed


def get_config(args):
    configuration = dict(
        SEED=1337,  # random seed for reproduce results
        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        EMBEDDING_SIZE=512,  # feature dimension #2048
    )

    if args.workers_id == 'cpu' or not torch.cuda.is_available():
        configuration['GPU_ID'] = []
        print("check", args.workers_id, torch.cuda.is_available())
    else:
        configuration['GPU_ID'] = [int(i) for i in args.workers_id.split(',')]
    if len(configuration['GPU_ID']) == 0:
        configuration['DEVICE'] = torch.device('cpu')
        configuration['MULTI_GPU'] = False
    else:
        configuration['DEVICE'] = torch.device('cuda:%d' % configuration['GPU_ID'][0])
        if len(configuration['GPU_ID']) == 1:
            configuration['MULTI_GPU'] = False
        else:
            configuration['MULTI_GPU'] = True

    configuration['NUM_EPOCH'] = args.epochs
    configuration['BATCH_SIZE'] = args.batch_size

    if args.data_mode == 'casia':
        configuration['DATA_ROOT'] = '/home/cvpr/Documents/facescrub_images_112x112/112x112'
    elif args.data_mode == "CelebA":
        configuration['DATA_ROOT'] = '/media/khawar/HDD_Khawar/face_datasets/CelebA'
    elif args.data_mode == "faces":
        configuration['DATA_ROOT'] = '/raid/khawar/dataset/faces/'
    else:
        print('fff')
        # raise Exception(args.data_mode)

    configuration['EVAL_PATH'] = './eval/'
    assert args.net in ['VIT', 'VITs', 'VITs_Eff', 'CAiT', 'DeepViT', 'CAiT', 'CrossViT', 'DEIT', 'PiT', 'Levitt',
                        'Comb_ViT', 'Dino_VIT', 'RVT', 'CeiT','FocalTransformer','PVTV2',
                        'CvT', 'Swin', 'T2TViT', 'Ours', 'NesT', 'CCT','PVT','P2T']
    configuration['BACKBONE_NAME'] = args.net
    # assert args.head in ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']
    configuration['HEAD_NAME'] = args.head
    configuration['TARGET'] = [i for i in args.target.split(',')]

    if args.resume:
        configuration['BACKBONE_RESUME_ROOT'] = args.resume
    else:
        configuration['BACKBONE_RESUME_ROOT'] = ''  # the root to resume training from a saved checkpoint
    configuration['WORK_PATH'] = args.outdir  # the root to buffer your checkpoints
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    return configuration


def model_summary(model):
    print("model_summary")
    print()
    print("Layer_name" + "\t" * 7 + "Number of Parameters")
    print("=" * 100)
    model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
    layer_name = [child for child in model.children()]
    j = 0
    total_params = 0
    print("\t" * 10)
    for i in layer_name:
        print()
        param = 0
        try:
            bias = (i.bias is not None)
        except:
            bias = False
        if not bias:
            param = model_parameters[j].numel() + model_parameters[j + 1].numel()
            j = j + 2
        else:
            param = model_parameters[j].numel()
            j = j + 1
        print(str(i) + "\t" * 3 + str(param))
        total_params += param
    print("=" * 100)
    print(f"Total Params:{total_params}")
