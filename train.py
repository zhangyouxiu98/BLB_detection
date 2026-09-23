import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import (seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    Cuda            = True
    seed            = 11
    distributed     = False
    sync_bn         = False
    fp16            = False

    num_classes     = 2
    #---------------------------------#
    #   Backbone network used:
    #   mobilenet
    #   xception
    #---------------------------------#
    backbone        = "mobilenet"
    model_path      = "model_data/deeplab_mobilenetv2.pth"
    #---------------------------------------------------------#
    #   downsample_factor   downsampling factor, 8 or 16.
    #                       8 downsamples less and is better in
    #                       theory, but needs more memory.
    #---------------------------------------------------------#
    downsample_factor   = 16
    #------------------------------#
    #   Input image size
    #------------------------------#
    input_shape         = [512, 512]
    Init_Epoch          = 0
    Freeze_Epoch        = 0
    Freeze_batch_size   = 0
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 16
    Freeze_Train        = False
    Init_lr             = 7e-3
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "sgd"
    momentum            = 0.9
    weight_decay        = 1e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   learning-rate decay: 'step' or 'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    save_period         = 5
    save_dir            = 'logs_cycleGAN_mask'
    eval_flag           = True
    eval_period         = 5

    #------------------------------------------------------------------#
    #   VOCdevkit_path  dataset path
    #------------------------------------------------------------------#
    VOCdevkit_path  = r'home\VOCdevkit_byk_CycleGAN'
    dice_loss       = False
    focal_loss      = False
    cls_weights     = np.ones([num_classes], np.float32)
    num_workers         = 4
    seed_everything(seed)
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    model   = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor)
    weights_init(model)
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        #------------------------------------------------------#
        #   Load weights by matching keys between the
        #   checkpoint and the model
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   Print keys that did not match
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "...\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "...\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mNote: it is normal for the head part not to load; it is an error for the backbone part not to load.\033[0m")

    #----------------------#
    #   Log the loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None

    #------------------------------------------------------------------#
    #   torch 1.2 does not support amp; use torch 1.7.1+ for fp16.
    #   Hence "could not be resolved" may appear with torch 1.2.
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   Sync BN across GPUs
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   Multi-GPU parallel
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    #---------------------------#
    #   Read the txt files for the dataset
    #---------------------------#
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        #---------------------------------------------------------#
        #   Total epochs = number of full passes over the data.
        #   Total steps  = number of gradient descents.
        #   Each epoch contains several steps; each step performs
        #   one gradient descent.
        #   This only suggests a minimum epoch count; only the
        #   unfreeze phase is considered.
        #----------------------------------------------------------#
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('Dataset is too small to train; please expand the dataset.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] With the %s optimizer, it is recommended to set the total training steps to at least %d.\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] This run has %d training samples, Unfreeze_batch_size=%d, %d epochs, so total steps=%d.\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] Since total steps=%d is less than the suggested %d, it is recommended to set the total epochs to %d.\033[0m"%(total_step, wanted_step, wanted_epoch))

    #------------------------------------------------------#
    #   The backbone features are general; freezing speeds up
    #   training and protects the weights early on.
    #   Init_Epoch: starting epoch
    #   Interval_Epoch: epochs of frozen training
    #   Epoch: total training epochs
    #   On OOM / out-of-memory, reduce the batch size.
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   Freeze part of the model for training
        #------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        #-------------------------------------------------------------------#
        #   If not freezing, use Unfreeze_batch_size directly
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   Adapt the learning rate to the current batch size
        #-------------------------------------------------------------------#
        nbs             = 16
        lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        if backbone == "xception":
            lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
            lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   Select the optimizer by type
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #---------------------------------------#
        #   Get the learning-rate decay schedule
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        #---------------------------------------#
        #   Compute the length of each epoch
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset is too small to continue training; please expand the dataset.")

        train_dataset   = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = deeplab_dataset_collate, sampler=train_sampler,
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = deeplab_dataset_collate, sampler=val_sampler,
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        #----------------------#
        #   Log the eval metric curve
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None

        #---------------------------------------#
        #   Start training
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   If the model has a frozen part,
            #   unfreeze it and reset the parameters
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   Adapt the learning rate to the current batch size
                #-------------------------------------------------------------------#
                nbs             = 16
                lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                if backbone == "xception":
                    lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   Get the learning-rate decay schedule
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset is too small to continue training; please expand the dataset.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = deeplab_dataset_collate, sampler=train_sampler,
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = deeplab_dataset_collate, sampler=val_sampler,
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
