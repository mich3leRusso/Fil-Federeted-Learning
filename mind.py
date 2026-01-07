import torch
from copy import deepcopy
from pruner import Pruner
from utils.generic import freeze_model, unfreeze_model
from test_fn import test_during_training
from parser import config_parameters
from torch.nn import CrossEntropyLoss
from utils.viz import plt_masks_grad_weight
import torch.nn.functional as F



class MIND():

    def __init__(self, model):

        # default values        
        self.scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.experience_idx = -1
        self.optimizer = None
        self.train_epochs = None

        # model + loss
        self.model = model
        self.criterion = CrossEntropyLoss(label_smoothing=0.)

        # params pruner
        self.pruner = Pruner(self.model, train_bias=False, train_bn=False)
        self.distillation = False

        #param distill
        self.distill_model = None

        # logging stuff
        self.log_every = config_parameters["log_every"]
        self.plot_gradients = config_parameters["plot_gradients"]
        self.plot_gradients_of_layer = config_parameters["plot_gradients_of_layer"]

        self.confusion_mat_task = torch.zeros((config_parameters["n_experiences"], config_parameters["n_experiences"]))
        self.forgetting_mat = torch.zeros((config_parameters["n_experiences"], config_parameters["n_experiences"]))

    def get_ce_loss(self):
        """ Cross entropy loss. """
        return self.criterion(self.mb_output, self.mb_y.to(config_parameters["device"]))



    def get_distill_loss_JS(self):
        """ Distillation loss. (jensen-shannon) """
        with torch.no_grad():
            old_y = self.distill_model.forward(self.mb_x)

        new_y = self.mb_output
        soft_log_old = torch.nn.functional.log_softmax(old_y+10e-5, dim=1)
        soft_log_new = torch.nn.functional.log_softmax(new_y+10e-5, dim=1)

        soft_old = torch.nn.functional.softmax(old_y+10e-5, dim=1)
        soft_new = torch.nn.functional.softmax(new_y+10e-5, dim=1)

        dist1 = torch.nn.functional.kl_div(soft_log_new+10e-5, soft_old+10e-5, reduction='batchmean')
        dist2 = torch.nn.functional.kl_div(soft_log_old+10e-5, soft_new+10e-5, reduction='batchmean')

        dist = ((dist1+dist2)/2).mean()

        return dist


    def train(self):

        for epoch in range(self.train_epochs):
            self.epoch = epoch
            if not self.distillation:
                loss_ce, loss_distill = self.training_epoch_fresh_model()
            else:
                loss_ce, loss_distill = self.training_epoch()


            if self.scheduler:
                self.scheduler.step()
            
            if (epoch) % self.log_every == 0 or epoch+1 == self.train_epochs:
                with open(f"logs/client_{config_parameters['client_id']}/{config_parameters['run_name']}/results/loss.csv", "a") as f:
                    f.write(f"{self.experience_idx},{self.distillation},{epoch},{loss_ce:.4f},{loss_distill:.4f}\n")
                with open(f"logs/client_{config_parameters['client_id']}/{config_parameters['run_name']}/results/acc.csv", "a") as f:
                    acc_train = test_during_training(self.pruner,
                                                               self.train_dataloader,
                                                               self.val_dataloader,
                                                               self.model,
                                                               self.fresh_model,
                                                               self.scheduler,
                                                               epoch,
                                                               self.experience_idx,
                                                               self.distillation,
                                                               plot=True)

                    f.write(f"{self.experience_idx},{self.distillation},{epoch},{acc_train:.4f}, loss: {loss_ce:.4f} loss_distill {loss_distill:.4f}  \n")
                #print loss
                #print(f"loss_ce: {loss_ce:.4f}, loss_distill: {loss_distill:.4f}")

        if self.distillation:
            self.model.save_bn_params(self.experience_idx)


    def training_epoch_fresh_model(self):
        base_model = deepcopy(self.fresh_model)
        # freeze the model
        freeze_model(base_model)
        self.fresh_model.train()

        for i, (self.mb_x, self.mb_y, self.mb_t) in enumerate(self.train_dataloader):
            self.mb_x = self.mb_x.to(config_parameters["device"])
            self.loss = torch.tensor(0.).to(config_parameters["device"])
            self.loss_ce = torch.tensor(0.).to(config_parameters["device"])
            self.loss_distill = torch.tensor(0.).to(config_parameters["device"])

            self.mb_output = self.fresh_model.forward(self.mb_x.to(config_parameters["device"]))

            self.loss_ce = self.get_ce_loss()
            self.loss += self.loss_ce + self.loss_distill
            self.loss.backward()

            if self.plot_gradients and (len(self.train_dataloader)-2)==i and self.epoch == self.train_epochs-1:
                plt_masks_grad_weight(config_parameters["run_name"], self.fresh_model, self.pruner, self.experience_idx, distillation=False, layer_idx=self.plot_gradients_of_layer)
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            if config_parameters["self_distillation"]:
                freeze_model(self.fresh_model)
                self.pruner.ripristinate_weights(self.fresh_model, base_model, self.experience_idx, self.distillation)
                unfreeze_model(self.fresh_model)

        return self.loss_ce, self.loss_distill


    def training_epoch(self):

        # to ripristinate the model
        base_model = deepcopy(self.model)
        
        # freeze the model
        freeze_model(base_model)

        self.model.train()

        for i, (self.mb_x, self.mb_y, self.mb_t) in enumerate(self.train_dataloader):
            self.mb_x = self.mb_x.to(config_parameters["device"])
            self.loss = torch.tensor(0.).to(config_parameters["device"])
            self.loss_ce = torch.tensor(0.).to(config_parameters["device"])
            self.loss_distill = torch.tensor(0.).to(config_parameters["device"])
            
            self.mb_output = self.model.forward(self.mb_x.to(config_parameters["device"]))


            if config_parameters["distill_beta"] > 0:
                self.loss_distill = config_parameters["distill_beta"]*self.get_distill_loss_JS()

            self.loss_ce = self.get_ce_loss()
            self.loss += self.loss_ce + self.loss_distill
            self.loss.backward()
            
            if self.plot_gradients and (len(self.train_dataloader)-2)==i and self.epoch == self.train_epochs-1:
                plt_masks_grad_weight(config_parameters["run_name"], self.model, self.pruner, self.experience_idx, distillation=True, layer_idx=self.plot_gradients_of_layer)
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            
            freeze_model(self.model)
            self.pruner.ripristinate_weights(self.model, base_model, self.experience_idx, self.distillation)
            unfreeze_model(self.model)

        return self.loss_ce, self.loss_distill


