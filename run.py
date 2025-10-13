import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from core.BIT_CD.models.network_split import BASE_Transformer
from core.cca_module import CCA
from evaluator import inference_source
from trainer import trainer

# import tent
# import cotta
import pa_ucd

from conf import cfg, load_cfg_fom_args
from data.PVPDataLoader import CDDataset


logger = logging.getLogger(__name__)


def load_model_PV():
    # Initialize weights from source model
    model_path = 'best_ckpt.pt'
    model = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_G_state_dict'], strict=True)

    CCA_model = CCA(model)
    return CCA_model

    
def evaluate(description):
    load_cfg_fom_args(description)
    # configure model
    base_model = load_model_PV().cuda()
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    if cfg.MODEL.ADAPTATION == "cotta":
        logger.info("test-time adaptation: CoTTA")
        model = setup_cotta(base_model)
    if cfg.MODEL.ADAPTATION == "rotta":
        logger.info("test-time adaptation: RoTTA")
        model = setup_rotta(base_model)
    if cfg.MODEL.ADAPTATION == "mgtta":
        logger.info("test-time adaptation: mgtta")
        # MGTTA requires some unlabeled training data
        train_set = CDDataset(root_dir='../PVPanel-CD-India', split='train', img_size=256, is_train=True)
        train_dataset = DataLoader(train_set, batch_size=cfg.TEST.BATCH_SIZE, shuffle=True, num_workers=4)
        model = setup_mgtta(base_model, train_dataset)
    if cfg.MODEL.ADAPTATION == "ours":
        logger.info("=====> test-time adaptation: ours")
        model = setup_ours(base_model)
    try:
        model.reset()
        logger.info("resetting model!!")
    except:
        logger.warning("not resetting model1")

    # root_dir = 'PVPanel-CD-India'
    root_dir = 'PVP-Germany'
    # 加载测试集    
    val_set = CDDataset(root_dir=root_dir, split='val', img_size=256, is_train=False)
    val_dataset = DataLoader(val_set, batch_size=cfg.TEST.BATCH_SIZE, shuffle=True, num_workers=4)
    # 进行训练或推理
    train_flag = True
    if train_flag:
        logger.info("=====> Start training.....")
        trainer(model, None, val_dataset, cfg, logger)
    else:
        scores_dict = inference_source(model, val_dataset)
        # #####
        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        logger.info('=' *10)
        logger.info('%s\n' % message)  # save the message


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    # logger.info(f"model for evaluation: %s", model)
    return model.cuda()


def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics, 配置模型进行训练 + 通过批量统计进行特征调制，
    collect the parameters for feature modulation by gradient optimization,  收集通过梯度优化进行特征调制的参数
    set up the optimizer, and then tent the model.  设置优化器，然后对模型进行帐篷自适应。
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    # logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def setup_ours(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics, 配置模型进行训练 + 通过批量统计进行特征调制，
    collect the parameters for feature modulation by gradient optimization,  收集通过梯度优化进行特征调制的参数
    set up the optimizer, and then tent the model.  设置优化器，然后对模型进行帐篷自适应。
    """
    # model = our_model.configure_model(model)
    params, param_names = pa_ucd.collect_params(model)
    optimizer = setup_optimizer(params)
    paucd_model = pa_ucd.ViLUCD(model, optimizer,
                             steps=cfg.OPTIM.STEPS,
                             episodic=cfg.MODEL.EPISODIC)
    # logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return paucd_model.cuda()


def setup_cotta(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model)   # 让模型train()状态，并不可梯度求导
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta.CoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC, 
                           mt_alpha=cfg.OPTIM.MT, 
                           rst_m=cfg.OPTIM.RST, 
                           ap=cfg.OPTIM.AP)
    # logger.info(f"model for adaptation: %s", model)
    # logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model.cuda()


def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    evaluate('cross-domains evaluation.')
