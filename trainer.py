import torch
import clip
import os
import numpy as np
from torch import nn
from tqdm import tqdm
from core.BIT_CD.misc.metric_tool import ConfuseMatrixMeter
from core.BIT_CD.models.losses import cross_entropy
from evaluator import inference
from evaluator import inference_source
from datetime import datetime
from PIL import Image, ImageDraw

device = 'cuda'
    

def trainer(model, train_dataset, val_dataset, cfg, logger):
    ####
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_pth = f'output/{cfg.MODEL.ADAPTATION}/{current_time}'
    
    os.makedirs(save_pth, exist_ok=True)
    max_ech = cfg.MAX_EPOCH
    update_acc = 0
    
    for epoch in range(1, (max_ech+1)):
        model.train()
        logger.info(f'Is_training: {epoch}/{max_ech}')
        for id, batch in enumerate(tqdm(val_dataset)):
            imgA = batch['A'].to(device)
            imgB = batch['B'].to(device)

            sam_mask = batch['sam_mask'].to(device)
                
            loss = model(imgA, imgB, sam_mask, epoch=epoch, name=batch['name'])    

            # 状态
            if id%100==2:
                logger.info(f'[{epoch}/{max_ech}]  loss: {loss.item()}')
        ############################################################################
            
        # start eval
        logger.info("=====> Start val")
        scores_dict = inference(model, val_dataset)
        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        logger.info('=' *10)
        logger.info('%s\n' % message)  # save the message
        ############################################################################

        # save model
        if scores_dict['acc'] > update_acc:
            update_acc = scores_dict['acc']
            torch.save({
            'epoch_id': epoch,
            'best_val_acc': update_acc,
            'model_state_dict': model.model.state_dict()}, 
            os.path.join(save_pth, f'ep_{epoch}.pt'))

        ############################################################################

