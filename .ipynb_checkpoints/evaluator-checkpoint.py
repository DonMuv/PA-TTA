import os
import torch
import clip
import numpy as np
from torch import nn
from PIL import Image
from core.BIT_CD.misc.metric_tool import ConfuseMatrixMeter
import torch.nn.functional as F
device = 'cuda'


def inference_source(model, dataset):
    model.eval()
    running_metric = ConfuseMatrixMeter(n_class=2)
    running_metric.clear()
    for id, batch in enumerate(dataset):
        img_in1 = batch['A'].to(device)
        img_in2 = batch['B'].to(device)
        outs = model(img_in1, img_in2).detach()
        # outs = model(img_in1, img_in2)[-1]

        outs = torch.argmax(outs, dim=1)
        target = batch['L'].to(device).detach()
        current_score = running_metric.update_cm(pr=outs.cpu().numpy(), gt=target.cpu().numpy())
        # log
        if id % 50 == 0:
            print(f"---> {id} done.")

    scores_dict = running_metric.get_scores()
    return scores_dict
        
    
def inference(model, dataset):
    model.eval()
    device = 'cuda'
    running_metric = ConfuseMatrixMeter(n_class=2)
    running_metric.clear()
    for id, batch in enumerate(dataset):
        imgA = batch['A'].to(device)
        imgB = batch['B'].to(device)
        sam_mask = batch['sam_mask'].to(device)
    
        outs = model(imgA, imgB, sam_mask, epoch=0, is_train=False).detach()
        outs = outs.argmax(dim=1)
        
        target = batch['L'].to(device).detach()

        current_score = running_metric.update_cm(pr=outs.cpu().numpy(), gt=target.cpu().numpy())
        # log
        if id % 50 == 0:
            print(f"---> {id} done.")

    scores_dict = running_metric.get_scores()

    return scores_dict



