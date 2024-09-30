from tqdm import tqdm
import wandb
import numpy as np
import os
import torch
@torch.no_grad()
def eval_model(model, test_loader, criterion, device):
    """ Computing model accuracy """
    correct = 0
    total = 0
    loss_list = []

    for idx, (img, label) in enumerate(tqdm(test_loader)):
        img, label = img.to(device), label.to(device)
        outputs = model(img)

        action_loss = criterion(outputs, label)
        preds = torch.argmax(outputs, dim=1)
        correct += len(torch.where(preds == label)[0])
        total += len(label)

        loss_list.append(action_loss.item())

    # Total correct predictions and loss
    accuracy = correct / total * 100
    loss = np.mean(loss_list)
    return accuracy, loss

def train_epoch(model, train_loader, optimizer, criterion, device):
    """ Training a model for one epoch """
    loss_list = []
    correct = 0
    total = 0

    for idx, (img, label) in enumerate(tqdm(train_loader)):
        img, label = img.to(device), label.to(device)
        outputs = model(img)

        action_loss = criterion(outputs, label)
        preds = torch.argmax(outputs, dim=1)
        correct += len(torch.where(preds == label)[0])
        total += len(label)
        loss_list.append(action_loss.item())

        action_loss.backward()

        # Updating parameters
        optimizer.step()

    mean_loss = np.mean(loss_list)
    return mean_loss, loss_list, correct / total * 100

def train_model(model, optimizer, scheduler, train_loader, valid_loader, test_loader, criterion, num_epochs, device):
    """ Training a model for a given number of epochs"""
    train_loss = []


    for epoch in tqdm(range(num_epochs)):
        if ((epoch + 1) % 5 == 0):
            # validation epoch
            model.eval()  # important for dropout and batch norms
            valid_acc, val_loss = eval_model(model, valid_loader, criterion, device)
            test_acc, _ = eval_model(model, test_loader, criterion, device)
            wandb.log({"Valid loss": val_loss, "Valid Seen Accuracy": valid_acc,
                       "Valid Unseen Accuracy": test_acc})

        # training epoch
        model.train()  # important for dropout and batch norms
        mean_loss, loss_li, train_acc = train_epoch(model=model, train_loader=train_loader, optimizer=optimizer, criterion=criterion, device=device)
        train_loss.append(mean_loss)
        scheduler.step()

        wandb.log({'Train loss': mean_loss, "Train Accuracy": train_acc})

        if (epoch % 3 == 0):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"    Train loss: {round(mean_loss, 5)}")
            print(f"    Valid Accuracy: {valid_acc}%")
            print(f"    Test Accuracy: {test_acc}%")

            print("\n")
            save_model(model, optimizer, epoch)
            #torch.save(model, savepath)

    print(f"Training completed")
    return train_loss

def save_model(model, optimizer, epoch):
    """ Saving model checkpoint """

    if (not os.path.exists("saved_models")):
        os.makedirs("saved_models")
    savepath = f"saved_models/resnet_contrastiveKL__{epoch}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        #'stats': stats
    }, savepath)
    return


def load_model(model, optimizer, savepath):
    """ Loading pretrained checkpoint """

    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    #stats = checkpoint["stats"]

    return model, optimizer, epoch#, stats

