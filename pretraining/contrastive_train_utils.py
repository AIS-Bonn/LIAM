from tqdm import tqdm
import wandb
import numpy as np
import os
import torch
@torch.no_grad()
def eval_model(model, test_loader, device):
    """ Computing model accuracy """
    loss_list = []
    acc_li = []

    #for idx, (img, label) in enumerate(tqdm(test_loader)):
    for idx, batch in enumerate(tqdm(test_loader)):
        #_, img, label, *_ = batch
        img, label = batch
        img, label = img.to(device), label.to(device)

        loss, logits, targets = model(img, label)
        loss_list.append(loss.item())
        accuracy = compute_accuracy(logits, targets)
        acc_li.append(accuracy)

    return np.mean(acc_li), np.mean(loss_list)

@torch.no_grad()
def eval_model_triple(model, test_loader, device):
    """ Computing model accuracy """
    loss_list = []
    acc_li_txt = []
    acc_li_act = []

    for idx, batch in enumerate(tqdm(test_loader)):
        loss, logits_txt_vis, logits_vis_act, targets_txt_vis, targets_vis_act = model(batch)
        loss_list.append(loss.item())
        acc_txt_vis = compute_accuracy(logits_txt_vis, targets_txt_vis)
        acc_vis_act = compute_accuracy(logits_vis_act, targets_vis_act)
        acc_li_txt.append(acc_txt_vis)
        acc_li_act.append(acc_vis_act)

    return np.mean(acc_li_txt), np.mean(acc_li_act), np.mean(loss_list)

def train_epoch(model, train_loader, optimizer, scheduler, device):
    """ Training a model for one epoch """
    loss_list = []
    acc_list = []

    for idx, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        #_, img, label, *_ = batch
        img, label = batch
        img, label = img.to(device), label.to(device)


        loss, logits, targets = model(img, label)
        loss_list.append(loss.item())

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # First remove gradient cliping

        acc_list.append(compute_accuracy(logits, targets))
        optimizer.step()

        scheduler.step()
        with torch.no_grad():
            model.temperature.clamp_(min=0.01, max=100)

    mean_loss = np.mean(loss_list)
    return mean_loss, loss_list, np.mean(acc_list)

def train_epoch_triple(model, train_loader, optimizer, scheduler, device):
    """ Training a model for one epoch """
    loss_list = []
    acc_li_txt = []
    acc_li_act = []

    for idx, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        loss, logits_txt_vis, logits_vis_act, targets_txt_vis, targets_vis_act = model(batch)
        loss_list.append(loss.item())

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # First remove gradient cliping
        acc_txt_vis = compute_accuracy(logits_txt_vis, targets_txt_vis)
        acc_vis_act = compute_accuracy(logits_vis_act, targets_vis_act)
        acc_li_txt.append(acc_txt_vis)
        acc_li_act.append(acc_vis_act)

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            model.temperature1.clamp_(min=0.01, max=100)
            model.temperature2.clamp_(min=0.01, max=100)


    mean_loss = np.mean(loss_list)
    return mean_loss, loss_list, np.mean(acc_li_txt), np.mean(acc_li_act)

def train_model_contrast(model, optimizer, scheduler, train_loader, valid_loader, test_loader, num_epochs, device, output_dir, triple=True):
    """ Training a model for a given number of epochs"""
    train_loss = []


    for epoch in tqdm(range(num_epochs)):
        model.eval()  # important for dropout and batch norms
        if not triple:
            valid_acc_act, val_loss = eval_model(model, valid_loader, device)
            wandb.log({"Valid loss": float(val_loss), "Valid accuracy": valid_acc_act})
        else:
            valid_acc_txt, valid_acc_act, val_loss = eval_model_triple(model, valid_loader, device)
            wandb.log({"Valid loss": float(val_loss), "Valid accuracy": valid_acc_act, "Valid accuracy text": valid_acc_txt})

        # training epoch
        model.train()  # important for dropout and batch norms

        if not triple:
            mean_loss, loss_li, train_acc = train_epoch(model=model, train_loader=train_loader, optimizer=optimizer,scheduler=scheduler, device=device)
            wandb.log({'Train loss': mean_loss, 'Train accuracy': train_acc})

        else:
            mean_loss, loss_li, train_acc_txt, train_acc_act = train_epoch_triple(model=model, train_loader=train_loader, optimizer=optimizer,scheduler=scheduler, device=device)
            wandb.log({'Train loss': mean_loss, 'Train accuracy': train_acc_txt, 'Train accuracy text': train_acc_txt})

        train_loss.append(mean_loss)


        if (epoch % 2 == 0):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"    Train loss: {round(mean_loss, 5)}")
            print(f"    Valid loss: {round(val_loss,5)}")
            print(f"    Valid Accuracy: {valid_acc_act}%")
            print("\n")

            save_model(model, optimizer, epoch, output_dir)

    if not triple:
        test_acc, test_loss = eval_model(model, test_loader, device)
        wandb.log({"Test loss": float(test_loss), "Test accuracy": test_acc})
    else:
        test_acc_txt, test_acc = eval_model_triple(model, test_loader, device)
    print(f"    Test Accuracy: {test_acc}%")
    print(f"    Test loss: {round(test_loss, 5)}")

    print(f"Training completed")
    return 0

def compute_accuracy(logits, target):
    # logits: [BSZ, number_actions]
    # target: [BSZ, number_actions]
    predicted_indices = torch.argmax(logits, dim=1)
    true_indices = torch.argmax(target, dim=1)
    # Compute the number of correct predictions
    correct_predictions = (predicted_indices == true_indices).sum().item()
    # Calculate the accuracy
    accuracy = correct_predictions / logits.size(0)

    return accuracy
def save_model(model, optimizer, epoch, output_dir):
    """ Saving model checkpoint """

    if (not os.path.exists("saved_models")):
        os.makedirs("saved_models")
    savepath = f"saved_models/{output_dir}_{epoch}.pth"

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

