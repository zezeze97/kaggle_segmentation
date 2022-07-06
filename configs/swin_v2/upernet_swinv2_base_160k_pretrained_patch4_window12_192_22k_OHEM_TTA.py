norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='Multi_Label_EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SwinTransformerV2',
        in_chans=3,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=24,
        pretrained_window_sizes=[12, 12, 12, 6],
        use_checkpoint=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12_192_22k.pth'
        )),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000, multi_label=True),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000, multi_label=True),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole',multi_label=True))
dataset_type = 'Kaggle_Dataset'
data_root = 'data/mmseg_train_25d_carno/'
classes = ['large_bowel', 'small_bowel', 'stomach']
palette = [[64,64,64],[128,128,128],[255,255,255]]
img_norm_cfg = dict(mean=[0,0,0], std=[1,1,1], to_rgb=True)
crop_size = (384, 384)
img_scale = (384, 384)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='unchanged', force_uint8=True, force_3channel=False),
    dict(type='LoadAnnotations',reduce_zero_label=False),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True, ratio_range=(0.75, 1.25)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=0.5, degree=(-30, 30), pad_val=0, seg_pad_val=0, center=None, auto_bound=False),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle_Multilabel'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='unchanged', force_uint8=True, force_3channel=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        img_ratios=[0.75, 1.0, 1.25],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
   train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images_25d',
        ann_dir='labels',
        img_suffix=".png",
        seg_map_suffix='.png',
        # split="splits/fold_0.txt",
        classes=classes,
        palette=palette,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images_25d',
        ann_dir='labels',
        img_suffix=".png",
        seg_map_suffix='.png',
        split="splits/holdout_0.txt",
        classes=classes,
        palette=palette,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        img_dir='images_25d',
        img_suffix=".png",
        seg_map_suffix='.png',
        split="splits/holdout_0.txt",
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))

log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000, max_keep_ckpts=1)
evaluation = dict(interval=16000, metric='mDice', pre_eval=True, save_best='mDice')
auto_resume = False