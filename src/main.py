from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from obj_spec import multi_data
from utils.widerface import WIDERDetection, detection_collate

def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    for k,v in model.named_parameters():
        print(k,v.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
    # for k,v in model.named_parameters():
    #     if 'bn' in k:
    #         v.requires_grad=False
    #     print(k,v.requires_grad)
    # params=torch.load('/home/mayx/project/github/CenterNet/exp/ctdet/pascal_resnet18_rgb_ori_20/model_best.pth',map_location='cpu')['state_dict']
    # model.load_state_dict(params)
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('Setting up data...')
    def default_collate(batches):
        batch_hm=[]
        batch_wh=[]
        batch_offset=[]
        batch_mask=[]
        for i in range(4):
            batch_hm.append([])
            batch_offset.append([])
            batch_wh.append([])
            batch_mask.append([])
        for batch in batches:
            for i in range(4):
                batch_hm[i].append(batch['hm'][i])
                batch_wh[i].append(batch['wh'][i])
                batch_offset[i].append(batch['offset'][i])
                batch_mask[i].append(batch['mask'][i])
        batch=dict()
        batch['hm']=[torch.tensor(i) for i in batch_hm]
        batch['wh']=[torch.tensor(i) for i in batch_wh]
        batch['offset']=[torch.tensor(i) for i in batch_offset]
        batch['mask']=[torch.tensor(i) for i in batch_mask]
        data=[i['input'] for i in batches]
        batch['input']=torch.stack([i['input'] for i in batches],0)
        return batch
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        # collate_fn=default_collate
    )

    if opt.test:
        _, preds = trainer.val(0, val_loader)
        val_loader.dataset.run_eval(preds, opt.save_dir)
        return
    if opt.multi_res:
        print('yes')
        train_loader=multi_data()
    else:
        train_loader = torch.utils.data.DataLoader(
            Dataset(opt, 'train'),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True,
            # collate_fn=default_collate
        )
    print('Starting training...')
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
