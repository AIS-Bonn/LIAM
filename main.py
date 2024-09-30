import os
import torch
from torch.utils.data import Dataset, DataLoader
from config.config import CFG
from utils.train_util import train_model, eval_model, load_model
import torch.nn as nn
import wandb
from tqdm import tqdm
#!Always check if the dataset and model is loaded correctly
from dataset.MyAlfred import Alfred, AlfredPreFeat
#from model.ET_openCLIP import ET_openclip
#from model.ET_Baseline import ET_Baseline
import numpy as np
import argparse

if __name__ == '__main__':
    torch.cuda.empty_cache()
    #wandb.init(name="db_aux only", project="MA")   #
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--root_dir', type=str, default="/home/nfs/inf6/data/datasets/alfred/generated_2.1.0")
    # "/home/local/ET/data/generated_2.1.0/"
    # /home/nfs/inf6/data/datasets/alfred/full_2.1.0/z/
    # /home/user/wang01/alfred/data/full_2.1.0/z/

    # often adjustable argument
    parser.add_argument('--BSZ', default=16, type=int)
    parser.add_argument('-e','--epochs', default=21)
    parser.add_argument('--nth_max_frame', default=-1)
    parser.add_argument('--acc_step', default=1)

    parser.add_argument('--use_et_generated', default=True)
    parser.add_argument('--clip_root', default='/home/nfs/inf6/data/datasets/alfred/mobile_clip_checkpoints/mobileclip_s0.pt')
    #/home/user/wang01/ml-mobileclip-main/checkpoints/mobileclip_s0.pt'
    parser.add_argument('--self_pretrain', type=bool, default=True)   # Use pre-aligned clip model, if False load normal embedding layer
    parser.add_argument('--pre_backbone_dir', type=str, default="MA/pre_vitb32_triple_10.pth")   # MA/3mobile_contrast_checkpoint_epoch_9.pth
    #MA/pre_vitb32_triple_10.pth        # MA/pre_vitb32_35_2con.pth
    parser.add_argument('--FEAT_precomputed', type=bool, default=True)
    parser.add_argument('--sem_map_module', type=bool, default=False)
    parser.add_argument('--ETencoder', type=bool, default=False)
    parser.add_argument('--aux', type=bool, default=False)
    parser.add_argument('--output_dir', type=str,default="dbaux")



    parser.add_argument('--device', default="cuda:0" if torch.cuda.is_available() else "cpu")

    arg = parser.parse_args()

    num_modals = 4 if arg.sem_map_module else 3

    if arg.ETencoder:
        print("Using ET baseline instead of CLIP")
        data = AlfredPreFeat(arg, mode="train", root_dir=arg.root_dir, LP_module=False, maps=arg.sem_map_module)
        # Drop last since cls token is initialized with the batch size
        trainloader = DataLoader(data, batch_size=arg.BSZ, shuffle=True, collate_fn=data.my_collate_fn, drop_last=False)
        print("Train data loaded")
        valid_data = AlfredPreFeat(arg, mode="valid_seen", root_dir=arg.root_dir, LP_module=False, maps=arg.sem_map_module)
        validloader = DataLoader(valid_data, batch_size=arg.BSZ, shuffle=True, collate_fn=valid_data.my_collate_fn,
                                 drop_last=False)
        print("Valid data loaded")
        test_data = AlfredPreFeat(arg, mode="valid_unseen", root_dir=arg.root_dir, LP_module=False, maps=arg.sem_map_module)
        testloader = DataLoader(test_data, batch_size=arg.BSZ, shuffle=False, collate_fn=test_data.my_collate_fn,
                                drop_last=False)
        print("Test data loaded")
        from model.MyETBase import ET_openclip
        model_e2e = ET_openclip(arg, num_modals=num_modals).to(arg.device)
    else:
        #from dataset.MyAlfredE2E import Alfred
        from dataset.MyAlfred import Alfred
        # !TODO change only for triple

        #arg = CFG()
        data = Alfred(arg, mode="train", root_dir=arg.root_dir, LP_module=False, maps=arg.sem_map_module)
        # Drop last since cls token is initialized with the batch size
        trainloader = DataLoader(data, batch_size=arg.BSZ, shuffle=True, collate_fn=data.my_collate_fn, drop_last=False)
        print("Train data loaded")
        valid_data = Alfred(arg, mode="valid_seen", root_dir=arg.root_dir, LP_module=False, maps=arg.sem_map_module)
        validloader = DataLoader(valid_data, batch_size=arg.BSZ, shuffle=True, collate_fn=valid_data.my_collate_fn, drop_last=False)
        print("Valid data loaded")
        test_data = Alfred(arg, mode="valid_unseen", root_dir=arg.root_dir, LP_module=False, maps=arg.sem_map_module)
        testloader = DataLoader(test_data, batch_size=arg.BSZ, shuffle=False, collate_fn=test_data.my_collate_fn, drop_last=False)
        print("Test data loaded")
        from model.MyET08 import ET_openclip
        print("Loading new model and dataset")

        model_e2e = ET_openclip(arg, num_modals=num_modals, aux=arg.aux).to(arg.device)

        for param in model_e2e.clip_model.parameters():
            param.requires_grad = False


    #import optuna
    # TODO write optuna coda
    optimizer = torch.optim.AdamW(model_e2e.parameters(), lr=1e-4, eps=1e-8)
    #model_e2e, optimizer, _ = load_model(model_e2e, optimizer, "/home/user/wang01/ma_wang/saved_models/ETbaseline/1807_0.pth")

    # ET base line step up
    #model_baseline = ET_Baseline(arg).to(arg.device)
    #optimizer = torch.optim.AdamW(model_baseline.parameters(), lr=1e-4, weight_decay=0.33)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=arg.epochs, eta_min=1e-5)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(trainloader), epochs=arg.epochs)

    #weights = [0.5, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0]
    #sum_w = sum(weights)
    #weights = [w / sum_w for w in weights]

    #class_weights = torch.FloatTensor(weights).to(arg.device)
    #criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=13)
    #criterion = nn.CrossEntropyLoss(ignore_index=13)
    


    train_model(model=model_e2e, optimizer=optimizer, train_loader=trainloader, valid_loader=validloader,test_loader=testloader,scheduler=scheduler, num_epochs=arg.epochs, accumulation_step=arg.acc_step,output_dir=arg.output_dir, aux=arg.aux)
    model, optimizer, _ = load_model(model_e2e, optimizer, "/home/local/saved_models_ma_wang/e2e_tri_baseline_8h_20.pth")
    model.eval()
    print("Load successful")
    # txt_acc, act_acc, loss = eval_model_triple(model, test_loader, arg.device)
    metrics_dict, loss = eval_model(model, testloader, arg.device)

    # print(txt_acc)
    # print(f"    Test Accuracy (img-txt): {txt_acc}%")
    print("===" * 40)
    print("Test")
    for keys, values in metrics_dict.items():
        #print(keys, np.mean(values))
        if keys == "acc": continue
        print(keys)
        values_li = torch.mean(torch.stack(values), dim=0)
        print(values_li)
    print("===" * 40)
    # val_acc, val_act_acc, loss = eval_model_triple(model, valid_loader, arg.device)

    metrics_dict, loss = eval_model(model, validloader, arg.device)

    # print(txt_acc)
    # print(f"    Valid Accuracy (img-txt): {val_acc}%")
    for keys, values in metrics_dict.items():
        #print(keys, np.mean(values))

        if keys == "acc": continue
        print(keys)
        values_li = torch.mean(torch.stack(values), dim=0)
        print(values_li)

    


    '''
    if eval_test:
        #torch.set_printoptions(profile="full")

        model = ET_openclip(arg, use_pretrain=True).to(arg.device)
        model_path = "/home/user/wang01/ma_wang/saved_models/ETbaseline/1306_weighted_checkpoint_epoch_15.pth"
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model.eval()
        acc, loss = eval_model(model, testloader, criterion)
        print("Test accuracy: ", acc)
        print("Test Loss: ", loss)
    '''

'''
def objective(trial, model, arg):
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    scheduler_name = trial.suggest_categorical("scheduler", ["StepLR", "OneCycleLR"])
    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.1)
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-2, steps_per_epoch=len(trainloader), epochs=arg.epochs)

    else:
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(optimizer)
'''

