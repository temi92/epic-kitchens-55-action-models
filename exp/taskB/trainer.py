import torch
import torch.nn as nn
from earlystopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import accuracy
from utils.plot_results import create_acc_loss_curve

class Trainer():
    def __init__(self, config, model, trainloader, valloader):
        self.config = config
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training in device: ", self.device)
        self.model = self.model.to(self.device)

        #define loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        #define optimizer 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr =self.config["lr"])

        self.model_results = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
        self.writer = SummaryWriter(log_dir="results")
  



    def train (self):

        """
        Training loop for the network
        """
        max_epochs = self.config["max_epochs"]
        best_loss = float("inf")
        early_stopping = EarlyStopping(patience=7, verbose=True, path='checkpoint.pt')
        for epoch in range(1, max_epochs):
            
            train_loss, train_acc = self.train_epoch()
            self.model_results["train_loss"].append(train_loss)
            self.model_results["train_acc"].append(train_acc)

            val_loss, val_acc = self.val_epoch()
            self.model_results["val_loss"].append(val_loss)
            self.model_results["val_acc"].append(val_acc)
          
            print ("Epoch [{}/{}], TrainLoss: {:.4f}, TrainAcc: {:.4f}, ValLoss: {:.4f}, ValAccuracy: {:.4f}"
            .format(epoch, max_epochs, train_loss, train_acc, val_loss, val_acc)
            )
   
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print ("Early stopping!!!")
                break


            #writing to tensorboard
            self.writer.add_scalar("Train/Loss", train_loss, epoch)   
            self.writer.add_scalar("Train/Accuracy", train_acc/100.0, epoch)

            self.writer.add_scalar("Val/Loss", val_loss, epoch)   
            self.writer.add_scalar("Val/Accuracy", val_acc/100.0, epoch)
            self.writer.flush()
            
    
        self.writer.close()
    
    def train_epoch(self):
        """
        Trains the model for one epoch on the dataset
        """
        losses = AverageMeter()
        accuracies = AverageMeter()

        self.model.train()
        for imgs, labels in self.trainloader:
          
            labels = labels["idx"]
            #transfer to GPU
            imgs = imgs.to(device=self.device)
            labels = labels.to(device=self.device)

            #compute model
            output,_ = self.model(imgs)
            loss = self.criterion(output, labels)

            #record loss ..
            losses.update(loss.item(), imgs.size(0))
            #measure and record accuracy
            acc1= accuracy(output, labels, ks=(1, ))
            accuracies.update(acc1[0].item(), imgs.size(0))

            #compute gradient and do step..
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return losses.avg, accuracies.avg

    def val_epoch(self):
        losses = AverageMeter()
        accuracies = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for imgs, labels in self.valloader:
                labels = labels["idx"]
                imgs = imgs.to(device=self.device)
                labels = labels.to(device=self.device)

                #compute model
                output,_ = self.model(imgs)
                loss = self.criterion(output, labels)

                #record loss ..
                losses.update(loss.item(), imgs.size(0))
                #measure and record accuracy
                acc1= accuracy(output, labels, ks=(1, ))
                accuracies.update(acc1[0].item(), imgs.size(0))
        return losses.avg,  accuracies.avg

    def plot_results(self):
        create_acc_loss_curve(self.model_results)

    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
