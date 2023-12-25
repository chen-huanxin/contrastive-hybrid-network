import os
import torch

from models.resnet_contrastive import get_resnet_contrastive
from models.origin_resnet import get_resnet_ms
from models.dcnn import DCNN
from models.phurie_model import PHURIE
from models.cnn_base import CNNBase


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

def make_model(args):
    if args.model == "OriginRes34" or args.model == "OriginRes50" or args.model == "OriginRes101":
        model = get_resnet_ms(args.model, num_classes=8)
    # elif args.model == "resnet32" or args.model == "resnet56" or args.model == "resnet110":
    # elif args.model == "dcnn":
    #     model = 
    elif args.model == "dcnn":
        args.reshape = 232
        model = DCNN()
    elif args.model == "phurie":
        model = PHURIE()
    elif args.model == "cnn_base":
        args.reshape = 259
        args.img_dim = 4
        model = CNNBase()
    else:
        model = get_resnet_contrastive(args.model, num_classes=8)

    if os.path.exists(args.weights):
        load_weight(model, args.weights)
        print("Successful using pretrain-weights.")
    else:
        print("not using pretrain-weights.")

    if torch.cuda.device_count() > 1 and args.multi_gpu:
        # 如果不用os.environ的话，GPU的可见数量仍然是包括所有GPU的数量
        # 但是使用的还是只用指定的device_ids的GPU

        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        # model = nn.DataParallel(model, device_ids=[0,1,2,3])
        model = torch.nn.DataParallel(model, device_ids=args.visible_device.split(','))
    
    model = model.to(args.device)

    return model

def train_model(args, model, inputs):
    if args.model == "GoogleNet":
        outputs = model(inputs).logits
    elif args.model == "resnet32" or args.model == "resnet56" or args.model == "resnet110":
        _, outputs = model(inputs)
    else:
        outputs = model(inputs)

    return outputs

def test_model(args, model, inputs):
    if args.model == "resnet32" or args.model == "resnet56" or args.model == "resnet110":
        outputs = model(inputs, Flag="Test")
    else:
        outputs = model(inputs)

    return outputs


if __name__ == "__main__":

    pass