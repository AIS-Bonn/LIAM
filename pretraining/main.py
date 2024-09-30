import torch.cuda
import torch.nn as nn
from Dataset import ActionMapping, ActionMappingRaw
import sys
sys.path.append("/home/user/wang01/MA")
from dataset.Alfred import Alfred
from torch.utils.data import DataLoader
from contrastive_train_utils import train_model_contrast, load_model, eval_model_triple, eval_model
from InferenceModel import ContrastiveModel, SuperContrastiveModel
import wandb
import argparse

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    #wandb.init(name="double mobile", project="MA")
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--root_dir', type=str, default="/home/nfs/inf6/data/datasets/alfred/generated_2.1.0")
    # "/home/local/ET/data/generated_2.1.0/"
    # /home/nfs/inf6/data/datasets/alfred/full_2.1.0/z/
    # /home/user/wang01/alfred/data/full_2.1.0/z/

    # often adjustable argument
    parser.add_argument('--BSZ', default=64, type=int)    # mobile clip 128 / CLIP RN50 64

    parser.add_argument('-e', '--epochs', default=11)
    parser.add_argument('--nth_max_frame', default=-1)
    parser.add_argument('--acc_step', default=1)

    parser.add_argument('--use_et_generated', default=True)
    parser.add_argument('--clip_root',
                        default='/home/nfs/inf6/data/datasets/alfred/mobile_clip_checkpoints/mobileclip_s0.pt')
    # /home/user/wang01/ml-mobileclip-main/checkpoints/mobileclip_s0.pt'
    parser.add_argument('--self_pretrain', type=bool,
                        default=False)  # Use pre-aligned clip model, if False load normal embedding layer
    parser.add_argument('--pre_backbone_dir', type=str,
                        default="MA/triple_contrast_checkpoint_epoch_18.pth")  # MA/3mobile_contrast_checkpoint_epoch_9.pth
    parser.add_argument('--FEAT_precomputed', type=bool, default=True)
    parser.add_argument('--sem_map_module', type=bool, default=True)
    parser.add_argument('--triple', type=bool, default=False)


    parser.add_argument('--device', default="cuda:0" if torch.cuda.is_available() else "cpu")


    arg = parser.parse_args()

    if arg.triple:
        root_dir = "/home/nfs/inf6/data/datasets/alfred/generated_2.1.0"

        train_dataset = Alfred(arg, mode="train", root_dir=root_dir)
        train_loader = DataLoader(train_dataset, batch_size=arg.BSZ, shuffle=True,
                                  collate_fn=train_dataset.my_collate_fn)
        valid_dataset = Alfred(arg, mode="valid_seen", root_dir=root_dir)
        valid_loader = DataLoader(valid_dataset, batch_size=arg.BSZ, shuffle=True,
                                  collate_fn=valid_dataset.my_collate_fn)
        test_dataset = Alfred(arg, mode="valid_unseen", root_dir=root_dir)
        test_loader = DataLoader(test_dataset, batch_size=arg.BSZ, shuffle=False, collate_fn=test_dataset.my_collate_fn)
        model = SuperContrastiveModel().to(arg.device)

    else:
    # Dataloader for double contrastive learning
        train_dataset = ActionMappingRaw("train", arg)#ActionMapping("/home/local/ET/ActionMapping", "train")
        valid_dataset = ActionMappingRaw("valid_seen", arg)#ActionMapping("/home/local/ET/ActionMapping", "valid_seen")
        test_dataset = ActionMappingRaw("valid_unseen", arg)#ActionMapping("/home/local/ET/ActionMapping", "valid_unseen")
        #train_dataset = ActionMapping("/home/local/ET/ActionMapping", "train")
        train_loader = DataLoader(train_dataset, batch_size=arg.BSZ, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=arg.BSZ, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=arg.BSZ, shuffle=False)
        model = ContrastiveModel().to(arg.device)



    print("Training started, this model has {} parameters.".format(count_parameters(model)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, eps=1e-8, weight_decay=0.2)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=1, factor=0.8)
    total_steps = int(arg.epochs * (len(train_loader) // arg.BSZ))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=13)


    #train_model(model=model, optimizer=optimizer, scheduler=scheduler, train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
    #            criterion=criterion, num_epochs=21, device=device)

    #train_model_contrast(model, optimizer, scheduler, train_loader, valid_loader, test_loader, arg.epochs, arg.device, "rn50 triple", triple=arg.triple)
    # !TODO don't forget to change the batch size when evaluating the pre-trained models.
    #model, optimizer, _ = load_model(model, optimizer, "/home/local/saved_models_ma_wang/pre_rn50_triple_10.pth")
    # Double: pre_vitb32_35_2con.pth
    # Triple: pre_vitb32_triple_10.pth / pre_mobileclip_triple_8.pth
    model.eval()
    print("Load successful")


    #txt_acc, act_acc, loss = eval_model_triple(model, test_loader, arg.device)
    act_acc, loss = eval_model(model, test_loader, arg.device)

    #print(txt_acc)
    #print(f"    Test Accuracy (img-txt): {txt_acc}%")
    print(f"    Test Accuracy (img-act): {act_acc}%")
    print(f"    Test loss: {round(loss, 5)}")
    print("===" * 40)

    #val_acc, val_act_acc, loss = eval_model_triple(model, valid_loader, arg.device)
    val_act_acc, loss = eval_model(model, valid_loader, arg.device)


    #print(txt_acc)
    #print(f"    Valid Accuracy (img-txt): {val_acc}%")
    print(f"    Valid Accuracy (img-act): {val_act_acc}%")
    print(f"    Valid loss: {round(loss, 5)}")




