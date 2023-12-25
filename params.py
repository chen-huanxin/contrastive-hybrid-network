import argparse

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--model",
    default="resnet32",
    choices=[
      "resnet32",
      "resnet56",
      "resnet110",
      "OriginRes34",
      "OriginRes50",
      "OriginRes101",
      "vgg16",
      "DenseNet121",
      "GoogleNet",
      "dcnn",
      "phurie",
      "cnn_base",
    ],
    help="Model to use",
  )

  parser.add_argument(
    "--dataset",
    default="DeepTI",
    choices=[
      "TCIR",
      "DeepTI",
    ],
    help="Dataset to use",
  )

  parser.add_argument(
    "--dataset_root",
    type=str,
    # default="I:\\TCIR-SPLT",
    default="I:\\NASA-tropical-storm",
    help="the path of dataset",
  )

  parser.add_argument(
    "--mode",
    default="cross-entropy",
    choices=[
      "cross-entropy",
      "contrastive",
      "ce-contrastive",
      "performance", # for test performance 
      "status",
    ],
    help="Type of loss function"
  )

  parser.add_argument(
    "--focal_alpha",
    default=None,
    type=float,
    help="On the contrastive step this will be multiplied by two.",
  )

  parser.add_argument(
    "--reshape",
    type=int,
    default=224,
    help="reshape size",
  )

  parser.add_argument(
    "--img_dim",
    type=int,
    default=3,
    help="dim of img"
  )

    # 0, 0.1 , 5
  parser.add_argument(
    "--focal_gamma",
    default=0.25,
    type=float,
    help="On the contrastive step this will be multiplied by two.",
  )

  parser.add_argument(
    "--batch_size",
    default=32,
    type=int,
    help="On the contrastive step this will be multiplied by two.",
  )

  parser.add_argument(
    "--temperature", 
    default=0.1, 
    type=float, 
    help="Constant for loss no thorough "
  )

  parser.add_argument(
    "--auto-augment", 
    default=False, 
    type=bool
  )

  # focal contrasive alpha
  parser.add_argument(
    "--alpha", 
    default=1, 
    type=float
  )

  parser.add_argument(
    "--n_epochs_contrastive", 
    default=75, 
    type=int
  )

  parser.add_argument(
    "--n_epochs_cross_entropy", 
    default=100, 
    type=int
  )

  # Train From Pretrained
  parser.add_argument(
    "--lr_contrastive", 
    default=0.01, 
    type=float
  )

  parser.add_argument(
    "--lr_cross_entropy", 
    default=0.001, 
    type=float
  )

  parser.add_argument(
    "--weights", 
    default="I:\\Models\\resnet34-333f7ec4.pth", 
    type=str, 
    help='initial weights path'
  )

  parser.add_argument(
    "--cosine", 
    default=False, 
    type=bool, 
    help="Check this to use cosine annealing instead of "
  )

  parser.add_argument(
    "--step", 
    default=True, 
    type=bool, 
    help="Check this to use step"
  )

  parser.add_argument(
    "--lr_decay_rate", 
    type=float, 
    default=0.1, 
    help="Lr decay rate when cosine is false"
  )

  parser.add_argument(
      "--lr_decay_epochs",
      type=list,
      default=[50, 75],
      help="If cosine false at what epoch to decay lr with lr_decay_rate",
  )

  parser.add_argument(
    "--weight_decay", 
    type=float, 
    default=1e-4, 
    help="Weight decay for SGD"
  )

  parser.add_argument(
    "--momentum", 
    default=0.9, 
    type=float, 
    help="Momentum for SGD"
  )

  parser.add_argument(
    "--num_workers", 
    default=4, 
    type=int, 
    help="number of workers for Dataloader"
  )

  parser.add_argument(
    "--gpu", 
    type=int, 
    default=0, 
    help="using gpu"
  )

  parser.add_argument(
    "--multi_gpu", 
    action="store_true", 
    default=False,
    help="use multi gpu",
  )

  parser.add_argument(
    "--visible_device",
    type=str,
    default="0,1,2,3",
    help=""
  )

  parser.add_argument(
    "--use_test", 
    action="store_true", 
    default=False
  )

  parser.add_argument(
    "--multi_mode", 
    action="store_true", 
    default=False,
  )

  parser.add_argument(
    "--log_root",
    type=str,
    default="logs",
    help="path of log file",
  )

  parser.add_argument(
    "--ckpt_root",
    type=str,
    default="checkpoint",
    help="path of checkpoint file",
  )

  parser.add_argument(
    "--use_focal",
    action="store_true",
    default=False,
    help="use focal loss",
  )

  args = parser.parse_args()

  return args