import os
import math
import torch
import logging
import numpy as np
from torch import nn, optim
from torchvision import transforms
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score, recall_score

from tools.data_maker import make_dataset
from tools.model_maker import train_model, test_model
from loss.supervised_contrastive_loss import SupervisedContrastiveLoss
from loss.focal_loss import FocalLoss
from augmentation.duplicate_sample_transform import DuplicateSampleTransform
from tools.utils import progress_bar
from perf import perf_test, stat_test

logger = logging.getLogger("CONTRASTIVE")
  

def adjust_learning_rate(args, optimizer, epoch):
    """

    :param optimizer: torch.optim
    :param epoch: int
    :param mode: str
    :param args: argparse.Namespace
    :return: None
    """
    if args.mode == "contrastive":
        lr = args.lr_contrastive
        n_epochs = args.n_epochs_contrastive
    elif args.mode == "cross_entropy":
        lr = args.lr_cross_entropy
        n_epochs = args.n_epochs_cross_entropy
    else:
        lr = args.lr_cross_entropy
        n_epochs = args.n_epochs_cross_entropy
        #raise ValueError("Mode %s unknown" % mode)

    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / n_epochs)) / 2

    if args.step:
        n_steps_passed = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if n_steps_passed > 0:
            lr = lr * (args.lr_decay_rate ** n_steps_passed)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    logger.info("Adjusting Lr = " + str(lr))


def start_up(args, model):
    if args.mode == "contrastive":
        train_contrastive(args, model)
    elif args.mode == "ce-contrastive":
        train_ce_contrasive(args, model)
    elif args.mode == "cross-entropy":
        train_cross_entropy(args, model)
    elif args.mode == "performance":
        perf_test(args, model)
    elif args.mode == "status":
        stat_test(args, model)


def train_contrastive(args, model):
    train_trans = transforms.Compose([
        transforms.Resize(args.reshape),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    contrastive_train_trans = DuplicateSampleTransform(train_trans)

    train_loader, _ = make_dataset(args, contrastive_train_trans, class_flag=True)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr_contrastive,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    criterion = SupervisedContrastiveLoss(device=args.device, temperature=args.temperature)
    criterion.to(args.device)

    ## train
    model.train()
    best_loss = float("inf")

    for epoch in range(args.n_epochs_contrastive):
        print(f"Epoch [{epoch + 1}/{args.n_epochs_contrastive}]")

        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = torch.cat(inputs)
            targets = targets.repeat(2)

            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()

            projections = model.forward_constrative(inputs)
            loss = criterion(projections, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            args.tb_writer.add_scalar(
                "Train Loss | Supervised Contrastive",
                loss.item(),
                epoch * len(train_loader) + batch_idx,
            )

            progress_bar(
                batch_idx,
                len(train_loader),
                "Loss: %.3f" % (train_loss / (batch_idx + 1)),
            )

        avg_loss = train_loss / (batch_idx + 1)
        logger.info("Epoch avg_loss= " + str(avg_loss))

        if avg_loss < best_loss:
            # print("Saving...")
            # save_name = os.path.join(args.ckpt_dir, "ckpt_contrastive_" + str(avg_loss) + ".pth")
            # torch.save(model.state_dict(), save_name)
            logger.info(f"Epoch: {epoch}, Better Loss: {avg_loss}")
            best_loss = avg_loss

        # validation(args, model, epoch, test_loader)
        
        adjust_learning_rate(args, optimizer, epoch)

    args.best_acc = 0.0
    args.best_rmse = 100.0
    model.freeze_projection()
    train_cross_entropy(args, model)


rmse = lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred))


def train_cross_entropy(args, model):
    train_loader, test_loader = make_dataset(args, class_flag=True)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr_cross_entropy,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.use_focal:
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    else:
        criterion = nn.CrossEntropyLoss()

    criterion.to(args.device)

    for epoch in range(args.n_epochs_cross_entropy):
        print(f"Epoch [{epoch + 1}/{args.n_epochs_cross_entropy}]")

        model.train()
        total_loss = 0
        total_num = 0
        total_correct_num = 0
        ps_sum = 0
        rs_sum = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            
            optimizer.zero_grad()

            outputs = train_model(args, model, inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)

            batch_size = targets.size(0)
            correct_num = predicted.eq(targets).sum().item()

            rs = recall_score(predicted.cpu(), targets.cpu(), average='macro', zero_division=True)
            ps = precision_score(predicted.cpu(), targets.cpu(), average='macro', zero_division=True)

            ps_sum += ps
            rs_sum += rs
            # f1s = f1_score(predicted.cpu(), targets.cpu(), average='weighted')

            total_num += batch_size
            total_correct_num += correct_num

            args.tb_writer.add_scalar(
                "Loss train | Cross Entropy",
                loss.item(),
                epoch * len(train_loader) + batch_idx,
            )

            args.tb_writer.add_scalar(
                "Accuracy train | Cross Entropy",
                correct_num / batch_size,
                epoch * len(train_loader) + batch_idx,
            )

            progress_bar(
                batch_idx,
                len(train_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d) | Precision: %.3f%% | Recall: %.3f%%"
                % (
                    total_loss / (batch_idx + 1),
                    100.0 * total_correct_num / total_num,
                    total_correct_num,
                    total_num,
                    100.0 * ps,
                    100.0 * rs,
                ),
            )

        logger.info(
            "CE Training Epoch = " + str(epoch) + 
            " Loss train = " + str(loss.item()) + 
            " Accuracy train = " + str(correct_num / batch_size) +
            " Precision = " + str(ps_sum / len(train_loader)) +
            " Recall = " + str(rs_sum / len(train_loader))
        )
        
        validation(args, model, epoch, test_loader)
        adjust_learning_rate(args, optimizer, epoch)
    
    print("Finish Training...")


def train_ce_contrasive(args, model):
    train_trans = transforms.Compose([
        transforms.Resize(args.reshape),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    contrastive_train_trans = DuplicateSampleTransform(train_trans)

    train_loader, test_loader = make_dataset(args, contrastive_train_trans, class_flag=True)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr_cross_entropy,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    criterion_contrastive = SupervisedContrastiveLoss(device=args.device, temperature=args.temperature)
    criterion_contrastive.to(args.device)

    criterion_focal = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    criterion_focal.to(args.device)

    ## train
    model.train()
    best_loss = float("inf")

    train_losses = []
    test_acc = []
    
    for epoch in range(args.n_epochs_cross_entropy):
        alpha = 1 - (epoch / args.n_epochs_cross_entropy)

        print(f"Epoch [{epoch + 1}/{args.n_epochs_cross_entropy}]")

        model.train()
        total_loss = 0
        total_num = 0
        total_correct_num = 0
        ps_sum = 0
        rs_sum = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = torch.cat(inputs)
            targets = targets.repeat(2)
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            
            optimizer.zero_grad()

            # _, outputs = model(inputs)
            outputs_contrasive, outputs_classify = model(inputs)

            loss_contrasive = criterion_contrastive(outputs_contrasive, targets)
            loss_focal = criterion_focal(outputs_classify, targets)

            loss_all = (1 - alpha) * loss_focal + alpha * loss_contrasive

            train_losses.append(loss_all.item())
            # loss = criterion(outputs, targets)
            loss_all.backward()
            optimizer.step()

            total_loss += loss_focal.item()
            _, predicted = outputs_classify.max(1)

            batch_size = targets.size(0)
            correct_num = predicted.eq(targets).sum().item()

            rs = recall_score(predicted.cpu(), targets.cpu(), average='macro', zero_division=True)
            ps = precision_score(predicted.cpu(), targets.cpu(), average='macro', zero_division=True)

            ps_sum += ps
            rs_sum += rs
            # f1s = f1_score(predicted.cpu(), targets.cpu(), average='weighted')

            total_num += batch_size
            total_correct_num += correct_num

            args.tb_writer.add_scalar(
                "Loss train | Cross Entropy",
                loss_focal.item(),
                epoch * len(train_loader) + batch_idx,
            )

            args.tb_writer.add_scalar(
                "Accuracy train | Cross Entropy",
                correct_num / batch_size,
                epoch * len(train_loader) + batch_idx,
            )

            progress_bar(
                batch_idx,
                len(train_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d) | Precision: %.3f%% | Recall: %.3f%%"
                % (
                    total_loss / (batch_idx + 1),
                    100.0 * total_correct_num / total_num,
                    total_correct_num,
                    total_num,
                    100.0 * ps,
                    100.0 * rs,
                ),
            )

            if loss_focal < best_loss:
                # print("Saving...")
                # save_name = os.path.join(args.ckpt_dir, "ckpt_contrastive_" + str(avg_loss) + ".pth")
                # torch.save(model.state_dict(), save_name)
                logger.info(f"Epoch: {epoch}, Better Loss: {loss_focal}")
                best_loss = loss_focal

        logger.info(
            "CE Training Epoch = " + str(epoch) + 
            " Loss train = " + str(loss_focal.item()) + 
            " Accuracy train = " + str(correct_num / batch_size) +
            " Precision = " + str(ps_sum / len(train_loader)) +
            " Recall = " + str(rs_sum / len(train_loader))
        )
        
        acc = validation(args, model, epoch, test_loader)
        test_acc.append(acc)
        adjust_learning_rate(args, optimizer, epoch)

    np.save(os.path.join(args.log_dir, "train_loss.npy"), np.array(train_losses))
    np.save(os.path.join(args.log_dir, "test_acc.npy"), np.array(test_acc))
    
    print("Finish Training...")


def validation(args, model, cur_epoch, data_loader):
    model.eval()

    test_loss = 0
    correct_num = 0
    total_num = 0
    ps_sum = 0
    rs_sum = 0
    all_rmse = []
    acc_arr = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            outputs = test_model(args, model, inputs)

            _, predicted = outputs.max(1)

            total_num += targets.size(0)
            correct_num += predicted.eq(targets).sum().item()

            rs = recall_score(predicted.cpu(), targets.cpu(), average='macro', zero_division=True)
            ps = precision_score(predicted.cpu(), targets.cpu(), average='macro', zero_division=True)

            rs_sum += rs
            ps_sum += ps
            # f1s = f1_score(predicted.cpu(), targets.cpu(), average='weighted')

            label_true = targets.data.cpu().numpy()
            pred_true = predicted.data.cpu().numpy()

            cal_rmse = rmse(label_true, pred_true)
            all_rmse.append(cal_rmse)

            progress_bar(
                batch_idx,
                len(data_loader),
                "Loss: %.3f | Acc: %.3f%% | RMSE: %.3f(%d/%d) | Precision: %.3f%% | Recall: %.3f%%"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct_num / total_num,
                    np.mean(all_rmse),
                    correct_num,
                    total_num,
                    100.0 * ps,
                    100.0 * rs,
                ),
            )
    
    # save checkpoint
    acc = 100.0 * correct_num / total_num
    mean_rmse = np.mean(all_rmse)
    logger.info(
        "Epoch" + str(cur_epoch) + 
        " Testing Accuracy: " + str(acc) + 
        " Testing rmse: " + str(mean_rmse) +
        " Precision: " + str(ps_sum / len(data_loader)) +
        " Recall: " + str(rs_sum / len(data_loader))
    )
    logger.info("Best Accuracy: " + str(args.best_acc) + " Best RMSE: " + str(args.best_rmse))

    print("[Epich {}], Accuracy: {}, RMSE: {}".format(cur_epoch, acc, mean_rmse))

    args.tb_writer.add_scalar("Accuracy validation | Cross Entropy", acc, cur_epoch)

    if mean_rmse < args.best_rmse:
        print("Saving...")
        args.best_rmse = mean_rmse

        state = model.state_dict()

        save_name = os.path.join(args.ckpt_dir, "ckpt_" + str(mean_rmse) + ".pth")
        torch.save(state, save_name)

    if acc > args.best_acc:
        print("Saving...")
        args.best_acc = acc

        state = model.state_dict()

        save_name = os.path.join(args.ckpt_dir, "ckpt_" + str(acc) + ".pth")
        torch.save(state, save_name)

    return acc
        


