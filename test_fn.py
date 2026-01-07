import torch
from torch.utils.data import DataLoader
from utils.generic import freeze_model, unfreeze_model
from copy import deepcopy
from utils.viz import plt_test_confmat, plt_confmats, plt_test_confmat_task
from parser import config_parameters
import pickle as pkl
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math

def get_stat_exp(y, y_hats, exp_idx, task_id, task_predictions):
    """ Compute accuracy and task accuracy for each experience."""
    conf_mat = torch.zeros((exp_idx+1, exp_idx+1))
    for i in range(exp_idx+1):
        ybuff= y[task_id==i]
        y_hats_buff=y_hats[task_id==i]
        acc = (ybuff==y_hats_buff).sum()/y_hats_buff.shape[0]

        for j in range(exp_idx+1):
            conf_mat[i,j] = ((task_id==i)&(task_predictions==j)).sum()/(task_id==i).sum()

        print(f"EXP:{i}, acc:{acc:.3f}, task:{conf_mat[i,i]:.3f}, distrib:{[round(conf_mat[i,j].item(), 3) for j in range(exp_idx+1)]}")


def entropy(vec):
    return -torch.sum(vec * torch.log(vec + 1e-7), dim=1)


def test(strategy, test_set, plot=True):
    strategy.model.eval()
    dataloader = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=8)

    s = config_parameters["n_classes"] + int(config_parameters["extra_classes"]*config_parameters["n_experiences"])
    confusion_mat = torch.zeros((s, s))
    confusion_mat_taw = torch.zeros((s, s))

    y_hats = []
    y_taw = []
    ys = []
    task_predictions = []
    task_ids = []
    for i, (x, y, task_id) in enumerate(dataloader):
        frag_preds = []
        for j in range(strategy.experience_idx+1):
            # create a temporary model copy
            model = freeze_model(deepcopy(strategy.model))

            strategy.pruner.set_gating_masks(model, j, weight_sharing=config_parameters["weight_sharing"], distillation=True)
            model.load_bn_params(j)
            model.exp_idx = j

            pred = model(x.to(config_parameters["device"]))

            #in class augmentation we have additional classes, this part of the code is made to ignore them
            pred = pred[:, j * (config_parameters["classes_per_exp"] + config_parameters["extra_classes"]): (j + 1) * (config_parameters["classes_per_exp"] + config_parameters["extra_classes"])]
            sp = torch.softmax(pred/config_parameters["temperature"], dim=1)
            sp = sp[:, :config_parameters["classes_per_exp"]]
            frag_preds.append(sp)

        frag_preds = torch.stack(frag_preds)  # [n_frag, bsize, n_classes]

        batch_size = frag_preds.shape[1]

        ### select across the top 2 of likelihood the head  with the lowest entropy
        # buff -> batch_size  x 2, 0-99 val
        buff = frag_preds.max(dim=-1)[0].argsort(dim=0)[-2:] # [2, bsize]

        # buff_entropy ->  2 x batch_size, entropy values
        indices = torch.arange(batch_size)

        task_predictions.append(buff[-1])
        y_hats.append(frag_preds[buff[-1], indices].argmax(dim=1) + (config_parameters["classes_per_exp"] + config_parameters["extra_classes"])*buff[-1])
        y_taw.append(frag_preds[task_id.to(torch.int32), indices].argmax(dim=-1) + ((config_parameters["classes_per_exp"] + config_parameters["extra_classes"])*task_id.to(config_parameters["cuda"])).to(torch.int32))

        task_ids.append(task_id)
        ys.append(y)

    # concat labels and preds
    y_hats = torch.cat(y_hats, dim=0).to('cpu')

    y = torch.cat(ys, dim=0).to('cpu')
    y_taw = torch.cat(y_taw, dim=0).to('cpu')
    task_predictions = torch.cat(task_predictions, dim=0).to('cpu')
    task_ids = torch.cat(task_ids, dim=0).to('cpu')

    #this piece of code is made to remove element of extra classes in the train set
    a = y % (config_parameters["classes_per_exp"] + config_parameters["extra_classes"])
    y = y[a < config_parameters["classes_per_exp"]]
    y_hats = y_hats[a < config_parameters["classes_per_exp"]]
    y_taw = y_taw[a < config_parameters["classes_per_exp"]]
    task_predictions = task_predictions[a < config_parameters["classes_per_exp"]]
    task_ids = task_ids[a < config_parameters["classes_per_exp"]]


    # assign +1 to the confusion matrix for each prediction that matches the label
    for i in range(y.shape[0]):
        confusion_mat[y[i], y_hats[i]] += 1
        confusion_mat_taw[y[i], y_taw[i]] += 1

    
    #task confusion matrix and forgetting mat
    for j in range(strategy.experience_idx + 1):
        i = strategy.experience_idx
        acc_conf_mat_task = confusion_mat[j * (config_parameters["classes_per_exp"] + config_parameters["extra_classes"]):(j + 1) * (config_parameters["classes_per_exp"] + config_parameters["extra_classes"]),j * (config_parameters["classes_per_exp"] + config_parameters["extra_classes"]):(j + 1) * (config_parameters["classes_per_exp"] + config_parameters["extra_classes"])].diag().sum() / confusion_mat[i * (config_parameters["classes_per_exp"] + config_parameters["extra_classes"]):(i + 1) * (config_parameters["classes_per_exp"] + config_parameters["extra_classes"]),:].sum()
        strategy.confusion_mat_task[i][j] = acc_conf_mat_task
        strategy.forgetting_mat[i][j] = strategy.confusion_mat_task[:, j].max() - acc_conf_mat_task


    
        
    # compute accuracy
    accuracy = confusion_mat.diag().sum() / confusion_mat.sum()
    accuracy_taw = confusion_mat_taw.diag().sum() / confusion_mat_taw.sum()

    task_accuracy = (task_predictions==task_ids).sum()/y_hats.shape[0]
    print(f"Test Accuracy: {accuracy:.4f}, Test Accuracy taw: {accuracy_taw:.4f}, Task accuracy: {task_accuracy:.4f}")
    get_stat_exp(y, y_hats, strategy.experience_idx, task_ids,task_predictions)

    if plot:
        plt_test_confmat(config_parameters["run_name"], confusion_mat, strategy.experience_idx)
        if strategy.experience_idx == config_parameters["n_experiences"]-1:
            plt_test_confmat_task(config_parameters["run_name"], strategy.confusion_mat_task)
            torch.save(strategy.forgetting_mat, f'./logs/{config_parameters["run_name"]}/forgetting_mat.pt')
            torch.save(strategy.confusion_mat_task, f'./logs/{config_parameters["run_name"]}/confusion_mat_task.pt')


    if strategy.experience_idx == config_parameters["n_experiences"]-1:
        res = {}
        res['y'] = y.cpu().numpy()
        res['y_hats'] = y_hats.cpu().numpy()
        res['frag_preds'] = frag_preds.cpu().numpy()
        res['y_taw'] = y_taw.cpu().numpy()

        # write to file
        with open(f'./logs/{config_parameters["run_name"]}/res.pkl', 'wb') as f:
            pkl.dump(res, f)

    return accuracy, task_accuracy, accuracy_taw


#################### MIND TESTS ####################

def test_single_exp(pruner, tested_model, loader, exp_idx, distillation):
    confusion_mat = torch.zeros((config_parameters["n_classes"]+config_parameters["extra_classes"]*config_parameters["n_experiences"], config_parameters["n_classes"]+config_parameters["extra_classes"]*config_parameters["n_experiences"]))
    y_hats = []
    ys = []
    for i, (x, y, _) in enumerate(loader):
        model = freeze_model(deepcopy(tested_model))
        pred = model(x.to(config_parameters["device"]))
        preds = torch.softmax(pred, dim=1)

        #frag preds size = (hid,bsize,100) and I want to reshape (bsize, hid, 100)
        y_hats.append(preds.argmax(dim=1))
        ys.append(y)

    # concat labels and preds
    y_hats = torch.cat(y_hats, dim=0).to('cpu')
    y = torch.cat(ys, dim=0).to('cpu')

    #to filter out external elements
    a = y % (config_parameters["classes_per_exp"] + config_parameters["extra_classes"])
    y = y[a < config_parameters["classes_per_exp"]]
    y_hats = y_hats[a < config_parameters["classes_per_exp"]]

    # assign +1 to the confusion matrix for each prediction that matches the label
    for i in range(y.shape[0]):
        confusion_mat[y[i], y_hats[i]] += 1

    return confusion_mat


def test_during_training(pruner, train_dloader, test_dloader, model, fresh_model, scheduler, epoch, exp_idx, distillation, plot=True):
    if distillation:
        model = model.eval()
    else:
        model = fresh_model.eval()

    with torch.no_grad():
        train_conf_mat = test_single_exp(pruner, model, train_dloader, exp_idx, distillation)
       # test_conf_mat = test_single_exp(pruner, model, test_dloader, exp_idx, distillation)
        # compute accuracy
        train_acc = train_conf_mat.diag().sum() / train_conf_mat.sum()
      #  test_acc = test_conf_mat.diag().sum() / test_conf_mat.sum()
        print(f"    e:{epoch:03}, tr_acc:{train_acc:.4f}, lr:{scheduler.get_last_lr()[0]:.5f}")

        # if plot:
        #     plt_confmats(config_parametersrun_name, train_conf_mat, test_conf_mat, distillation, exp_idx)

    model.train()

    return train_acc


def confidence(frag_preds, task_id):
    on_shell_probs = []
    elsewhere_probs = []
    for i, frag in enumerate(frag_preds):
        on_shell_probs.append(torch.softmax(frag[task_id==i], dim = -1))
        elsewhere_probs.append(torch.softmax(frag[task_id!=i], dim = -1))


    max_on_shell_probs = torch.max(torch.stack(on_shell_probs), dim = -1)[0]
    on_shell_confidence = [(1./(1.-p + 1e-6).mean()) for p in max_on_shell_probs]

    max_elsewhere_probs = torch.max(torch.stack(elsewhere_probs), dim = -1)[0]
    elsewhere_confidence = [(1./(1.-p + 1e-6).mean()) for p in max_elsewhere_probs]

    return on_shell_confidence, elsewhere_confidence
