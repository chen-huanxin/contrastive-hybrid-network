import os
import torch
import logging
import datetime
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from params import parse_args
from tools.model_maker import make_model
from train import start_up

logger = logging.getLogger("CONTRASTIVE")


def load_weight(model, path):
    weights_dict = torch.load(path, map_location="cpu")
    new_state_dict = {}
    for k, v in weights_dict.items():
        # Transfer TC classification into Regression
        if "linear" in k :
            continue

        new_state_dict[k[7:]] = v

    model.load_state_dict(new_state_dict, strict=False)

    return model

if __name__ == "__main__":
    args = parse_args()

    ## 配置GPU
    if args.multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_device
    else:
        torch.cuda.set_device(args.gpu)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    ## 配置日志
    if not os.path.isdir(args.log_root):
        os.makedirs(args.log_root)

    if not os.path.isdir(args.ckpt_root):
        os.makedirs(args.ckpt_root)

    cur_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args.cur_time = cur_time

    suffix = cur_time + args.mode

    args.log_dir = os.path.join(args.log_root, cur_time + args.mode)
    os.makedirs(args.log_dir)

    args.ckpt_dir = os.path.join(args.ckpt_root, args.cur_time)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    logger.setLevel(logging.DEBUG) # 记得要加这一句，否则info不输出
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s -  %(message)s")

    fh = logging.FileHandler(os.path.join(args.log_dir, "log.txt"), encoding="utf8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.info(str(args))

    tb_writer = SummaryWriter(args.log_dir)
    args.tb_writer = tb_writer

    args.best_acc = 0.0
    args.best_rmse = 100.0

    ## 生成模型
    model = make_model(args)

    ## 启动
    start_up(args, model)

    tb_writer.close()