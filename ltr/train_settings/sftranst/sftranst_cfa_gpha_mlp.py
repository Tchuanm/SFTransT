import torch
from ltr.dataset import Lasot, MSCOCOSeq, Got10k, TrackingNet
from ltr.data import processing, sampler, LTRLoader
import ltr.models.tracking.sftranst_cfa_gpha_mlp as sftranst_network
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU


def run(settings):
    # Most common settings are assigned in the settings struct
    settings.device = 'cuda'
    settings.description = 'new try training settings.'
    settings.resume = True  # training begin with last epcoh or not;
    settings.train_all = False
    settings.multi_decay_step = False
    settings.batch_size = 4
    settings.iter = 1000
    settings.samples_per_epoch = settings.iter * settings.batch_size
    settings.backbone_LR = 1.e-5     # 1e-5;1e-6
    settings.others_LR = 1.e-4       # 1e-4;1e-5
    if settings.train_all:  # for train all
        settings.total_epoch = 500
        settings.decay_epoch = 400
        settings.decay_LR = 0.1
    else:  # got10k only
        settings.total_epoch = 100
        settings.decay_epoch = 80
        settings.decay_LR = 0.1
    if settings.multi_decay_step:
        settings.total_epoch = 500
        settings.decay_epoch = 300
        settings.decay_epoch2 = 400
        settings.decay_LR = 0.1
    # loss weight
    settings.loss_cls_weight = 10
    settings.loss_l1_weight = 5
    settings.loss_iou_weight = 2

    # neck model param
    settings.iteration_gpha = 6
    settings.iteration = 2
    settings.position_embedding = 'sine'
    settings.hidden_dim = 256
    settings.dropout = 0.1
    settings.nheads = 8
    settings.dim_feedforward = 2048  # 512
    settings.activation = 'gelu'  # gelu
    # base init
    settings.num_workers = 8
    settings.multi_gpu = True
    settings.print_interval = 50
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 4.0
    settings.template_area_factor = 2.0
    settings.search_feature_sz = 32
    settings.template_feature_sz = 16
    settings.search_sz = settings.search_feature_sz * 8     # 256
    settings.temp_sz = settings.template_feature_sz * 8     # 128
    settings.center_jitter_factor = {'search': 3, 'template': 0}
    settings.scale_jitter_factor = {'search': 0.25, 'template': 0}

    if settings.train_all:      # train whole datasets
        lasot_train = Lasot(settings.env.lasot_dir, split='train')
        trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(12)))
        coco_train = MSCOCOSeq(settings.env.coco_dir, version='2017')
    got10k_train = Got10k(settings.env.got10k_dir, split='all')

    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05), tfm.RandomHorizontalFlip(probability=0.5))

    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2), tfm.RandomHorizontalFlip(probability=0.5),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # Data processing to do on the training pairs
    data_processing_train = processing.trackingProcessing(search_area_factor=settings.search_area_factor,
                                                          template_area_factor=settings.template_area_factor,
                                                          search_sz=settings.search_sz,
                                                          temp_sz=settings.temp_sz,
                                                          center_jitter_factor=settings.center_jitter_factor,
                                                          scale_jitter_factor=settings.scale_jitter_factor,
                                                          mode='sequence',
                                                          transform=transform_train,
                                                          joint_transform=transform_joint)
    # The sampler for training  4 datasets    4580*2.5=11450  11440
    if settings.train_all:
        dataset_train = sampler.Sampler([lasot_train, got10k_train, coco_train, trackingnet_train], [1, 1, 1, 1],
                                        samples_per_epoch=settings.samples_per_epoch, max_gap=200,
                                        processing=data_processing_train)
    else:
        dataset_train = sampler.Sampler([got10k_train], [1],
                                        samples_per_epoch=settings.samples_per_epoch, max_gap=200,
                                        processing=data_processing_train)
    # The loader for training
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size,
                             num_workers=settings.num_workers, shuffle=True, drop_last=True, stack_dim=0)
    # Create network and actor
    model = sftranst_network.sftranst_network(settings)
    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        model = MultiGPU(model, dim=0)

    objective = sftranst_network.losses_combination(settings)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('---------------------------------------------model:--------------------------------\n', model)
    print('----------------------number of params: %d,  %.3f M -----------------' % (n_parameters, n_parameters/1e6))
    actor = actors.sftranstActor(net=model, objective=objective)

    # Optimizer
    param_dicts = [{"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
                   {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": settings.backbone_LR, }, ]        # backbone:e-5; others e-4
    optimizer = torch.optim.AdamW(param_dicts, lr=settings.others_LR, weight_decay=1.e-4)

    if settings.multi_decay_step:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=settings.decay_LR,
                       milestones=[settings.decay_epoch, settings.decay_epoch2])
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, settings.decay_epoch, gamma=settings.decay_LR)

    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler)
    trainer.train(settings.total_epoch, load_latest=settings.resume, fail_safe=True)
