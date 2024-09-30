import numpy as np
import torch.nn as nn
import torch
import sys
import torch.nn.functional as F
import open_clip
import mobileclip

LOW_ACTION_IDX = ["LookDown_15", "LookUp_15", "RotateLeft_90", "RotateRight_90", "MoveAhead_25","PickupObject", "PutObject",  "SliceObject", "OpenObject",
                   "CloseObject", "ToggleObjectOn", "ToggleObjectOff", "<<stop>>", "<<pad>>"]

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)

    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class ContrastiveModel(nn.Module):
    def __init__(self):
        super(ContrastiveModel, self).__init__()
        #self.clip_model, _, _ = open_clip.create_model_and_transforms('RN50', pretrained='cc12m') #'ViT-B-32', pretrained='laion2b_s34b_b79k'
        #self.clip_model.transformer = None
        #self.input_dim = 1024
        #self.clip_model, _, _ = mobileclip.create_model_and_transforms('mobileclip_s0',pretrained='/home/nfs/inf6/data/datasets/alfred/mobile_clip_checkpoints/mobileclip_s0.pt')
        #self.clip_model.text_encoder = None   # remove text encoder of mobile CLIP
        #self.input_dim = 512
        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32',pretrained='laion2b_s34b_b79k')  # 'ViT-B-32', pretrained='laion2b_s34b_b79k'
        self.clip_model.transformer = None
        self.input_dim = 512
        self.vis_emb = Enc_vis(enc_transformers=self.clip_model, input_dim=self.input_dim)
        #self.vis_emb.load_state_dict(torch.load(pre_trained_dir)['model_state_dict'])
        #self.vis_emb.classify = nn.Identity()
        self.seq_fus = Vision_adapter()

        self.action_emb = nn.Embedding(len(LOW_ACTION_IDX), 768)   # 12 actions, 2 tokens (pad,stop)

        initial_temperature = 0.07  # Follow CLIP
        self.temperature = torch.nn.Parameter(torch.tensor(initial_temperature))   # Make the temperatue trainable

    def forward(self, vis, label):

        label = label.view(-1)
        unique_elements = torch.unique(label)
        unique_elements_li = unique_elements.tolist()

        # Ground Truth old, redundant action label!
        # targets = torch.zeros((vis.shape[0], vis.shape[0])).to("cuda:0")
        # for i, label_idx in enumerate(label):
        #     targets[i, label_idx] = 1

        # Ground Truth, no redundant actions
        #targets = torch.zeros((vis.shape[0] * vis.shape[1], len(unique_elements_li))).to("cuda:0")
        targets = torch.zeros((vis.shape[0], len(unique_elements_li))).to("cuda:0")
        for i, label_idx in enumerate(label):
            targets[i, unique_elements_li.index(label_idx)] = 1

        #for bsz in range(vis.shape[0]):
        #    cur_vis_emb = self.clip_model.encode_image(vis[bsz])
        #    output.append(cur_vis_emb)
        #vis_emb_in = torch.stack(output, dim=0)

        vis_emb = self.vis_emb(vis)                            # [BSZ, 2, embedding size]
        vis_emb = self.seq_fus(vis_emb)                        # [BSZ, embedding size]


        # act_emb = self.action_emb(label)                     # [BSZ, embedding size]
        act_emb = self.action_emb(unique_elements)             # [BSZ, num_act_in_batch]

        # Optional: Normalize the embedding
        vis_emb = vis_emb / vis_emb.norm(dim=-1, keepdim=True)
        act_emb = act_emb / act_emb.norm(dim=-1, keepdim=True)

        # Test for if there is any nan value exist
        #nan_mask = torch.isnan(vis_emb)
        #nan_indices = torch.nonzero(nan_mask)
        #nan_mask = torch.isnan(act_emb)
        #nan_indices = torch.nonzero(nan_mask)
        #print(vis_emb.shape, act_emb.shape)

        #self.temperature = torch.clamp(self.temperature, 0.01, 100)
        logits = (vis_emb @ act_emb.T) / self.temperature      #  [BSZ, num_act]
        logits = logits.log_softmax(dim=-1)  # follow mobile clip / CLIP, remove 100
        # ! if use cross entropy loss, then the softmax is not necessary. Pytorch already integrated in the loss function

        #nan_mask = torch.isnan(logits)
        #nan_indices = torch.nonzero(nan_mask)
        #print(nan_indices)
        #print("---"*40)

        # Compute the ground truth instead of using 1
        #images_similarity = vis_emb @ vis_emb.T    #[BSZ, BSZ]
        #action_similarity = act_emb @ act_emb.T    #[NUM_ACTIONS, NUM_ACTION]
        #targets = F.softmax((images_similarity + action_similarity) / 2 * self.temperature, dim=-1)


        ##### Cross entropy loss
        #texts_loss = cross_entropy(logits, targets, reduction='mean')
        #images_loss = cross_entropy(logits.T, targets.T, reduction='mean')
        #loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)  use the mean loss, other wise both loss should have same shape
        #####

        ##### KL divergence loss
        targets = targets.view(logits.shape)
        texts_loss = KLDivergence(logits, targets)
        images_loss = KLDivergence(logits.T, targets.T)
        loss = (images_loss + texts_loss) / 2.0


        #values, indices = torch.topk(logits.squeeze(0), n * 5)
        return loss, logits, targets

        '''
        texts_loss = loss_func(logits, targets)  # self.cross_entropy(logits, targets, reduction='none')
        images_loss = loss_func(logits.T, targets.T)  # self.cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        # print("texts loss: {}, images loss: {}".format(texts_loss, images_loss))
        # print("loss: {} and loss mean: {}".format(loss, loss.mean()))

        return loss.mean(), logits, vis_feat, txt_feat  # loss
        '''

class Vision_adapter(nn.Module):
    def __init__(self):
        super(Vision_adapter, self).__init__()
        #self.model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.conv = nn.Conv1d(in_channels=2, out_channels=768, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.residual_layer = nn.Conv1d(in_channels=2, out_channels=768, kernel_size=1, stride=1)

        #self.adapt_layer_vis = nn.Sequential(nn.Linear(768, 384),
        #                                     nn.LeakyReLU(),
        #                                     nn.Linear(384, 768),
        #                                     nn.LeakyReLU())
        #self.classify = nn.Linear(768, 12)

    def forward(self, x):
        #image_features = model.encode_image(x)
        bsz = x.shape[0]
        residual_connnection = self.residual_layer(x)

        x = self.conv(x)  # Apply 1D convolution
        x = self.relu(x)  # Apply ReLU activation

        x = x + residual_connnection
        x = self.global_pooling(x)  # Global mean pooling
        x = x.view(bsz, -1)  # Flatten to [BSZ, output_dim]
        return x



# KL divergence loss for contrastive pre-training, self-implementation
'''
def KLDivergence(input_prob, target_prob):
    each_class = target_prob.exp() * (target_prob - input_prob) #shape is BSZ x 64
    #each_class = target_prob * (torch.log(target_prob) - input_prob)  # 0 entry in target_prob, apply log leads to value -inf
    #each_class = each_class[torch.where(target_prob!=0)]
    return torch.mean(each_class)
'''

# Pytorch implementation
def KLDivergence(input_log_prob, target_prob):
    # Compute the KL divergence
    kl_div_loss = F.kl_div(input_log_prob, target_prob, reduction='batchmean')
    return kl_div_loss

### Pretraining for both modalities
class SuperContrastiveModel(nn.Module):
    def __init__(self):
        super(SuperContrastiveModel, self).__init__()
        #self.clip_model, _, _ = mobileclip.create_model_and_transforms('mobileclip_s0',
        #                                                               pretrained='/home/nfs/inf6/data/datasets/alfred/mobile_clip_checkpoints/mobileclip_s0.pt')
        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32',pretrained='laion2b_s34b_b79k')  # 'ViT-B-32', pretrained='laion2b_s34b_b79k'
        #self.clip_model, _, _ = open_clip.create_model_and_transforms('RN50', pretrained='cc12m')
        self.input_dim = 512 # 1024 if rn50 else 512

        self.vis_emb = Enc_vis(enc_transformers=self.clip_model, input_dim=self.input_dim)
        self.lang_emb = Enc_lang(enc_transformers=self.clip_model, input_dim=self.input_dim)
        self.act_emb = nn.Embedding(len(LOW_ACTION_IDX), 768)  # 12 actions, 2 tokens (pad,stop)

        initial_temperature = 0.07  # Follow CLIP
        self.temperature1 = torch.nn.Parameter(torch.tensor(initial_temperature))  # Make the temperatue trainable
        self.temperature2 = torch.nn.Parameter(torch.tensor(initial_temperature))  # Make the temperatue trainable
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.alpha = 0.8
        self.crossent_txtvis = True
        self.seq_fus = Vision_adapter()


    def forward(self, alfred_input):
        '''
        :param alfred_input:
        lang [BSZ, X, 77] -> txt_emb [BSZ, 768]
        vis [BSZ, Y, 3, 224, 224] -> vis_emb [BSZ, Y, 768] (txt-vis)
                                     vis_emb [BSZ * Y, 768] (vis-act)
        act [BSZ, Y] -> act_emb [n_act, 768]

        GT(txt-vis):
        '''
        #TODO based on the padding, remove the padding
        lang, vis, act, _, len_frames, *_ = alfred_input
        lang, vis, act = lang.to(self.device), vis.to(self.device), act.to(self.device)
        #print(lang.shape, vis.shape, act.shape)
        #print(act)
        #print(len_frames)

        # For each sample in the batch, the length of current action label should be len[frames] - 1
        act_l = []
        len_data = 20
        for idx, i in enumerate(act):
            i = i[i != 13]
            i = i[i != 12]
            if len_frames[idx] == len_data and i.shape != torch.Size([len_data - 1]):
                i = i[:-1]
            #print(i.shape)
            #print()
            act_l.append(i)
        act_label = torch.cat(act_l, dim=0)
        #print(f'{act_label.shape=}')


        #act_label = act_label[act_label != 13]
        #act_label = act_label[act_label != 12]


        unique_elements = torch.unique(act_label)
        unique_elements_li = unique_elements.tolist()
        #targets_vis_act = torch.zeros((vis.shape[0] * vis.shape[1], len(unique_elements_li))).to(self.device)
        targets_vis_act = torch.zeros((torch.sum(len_frames) - vis.shape[0], len(unique_elements_li))).to(self.device)
        #print(f'{targets_vis_act.shape=}')

        #print(targets_vis_act.shape)
        for i, label_idx in enumerate(act_label):
            try:
                targets_vis_act[i, unique_elements_li.index(label_idx)] = 1
            except:
                print("Loop error")
                target_len = torch.sum(len_frames) - lang.shape[0]
                current_len = act_label.shape
                print(len_frames)
                act_len = [a.shape for a in act_l]
                print(act_len)
                print("==="*40)
                print(target_len, current_len)

        txt_emb = self.lang_emb(lang)
        vis_emb = self.vis_emb(vis)

        vis_emb_seq = vis_emb / vis_emb.norm(dim=-1, keepdim=True)
        vis_emb = torch.mean(vis_emb_seq, dim=1)
        txt_emb = torch.mean(txt_emb / txt_emb.norm(dim=-1, keepdim=True), dim=1)
        # Now vis_emb and txt_emb should both have shape [BSZ, 768]


        logits_txt_vis = (vis_emb @ txt_emb.T) / self.temperature1
        targets_txt_vis = torch.eye(lang.shape[0]).to(self.device)

        if self.crossent_txtvis:
            texts_loss = cross_entropy(logits_txt_vis, targets_txt_vis, reduction="mean")
            images_loss = cross_entropy(logits_txt_vis.T, targets_txt_vis.T, reduction="mean")
            loss1 = (images_loss + texts_loss) / 2.0
        else:
            logits_txt_vis = logits_txt_vis.log_softmax(dim=-1)  # .softmax(dim=-1)  IF KL divergence, we need softmax
            texts_loss = KLDivergence(logits_txt_vis, targets_txt_vis)
            images_loss = KLDivergence(logits_txt_vis.T, targets_txt_vis.T)
            loss1 = (images_loss + texts_loss) / 2.0


        # vis_emb_samplewise = self.vis_emb_per_sample(vis)
        #vis_emb_seq = vis_emb_seq.view(-1, 768)
        vis_emb_seq_f = self.process_vis_emb_seq(vis_emb_seq, len_frames)

        #vis_emb_seq_f = vis_emb_seq_f / vis_emb_seq_f.norm(dim=-1, keepdim=True)
        #print(f'{vis_emb_seq_f.shape=}')


        act_emb = self.act_emb(unique_elements)
        act_emb = act_emb / act_emb.norm(dim=-1, keepdim=True)

        logits_vis_act = (vis_emb_seq_f @ act_emb.T) / self.temperature2

        logits_vis_act = logits_vis_act.log_softmax(dim=-1)

        images_loss2 = KLDivergence(logits_vis_act, targets_vis_act)
        act_loss = KLDivergence(logits_vis_act.T, targets_vis_act.T)
        loss2 = (images_loss2 + act_loss) / 2.0



        sum_loss = (1 - self.alpha) * loss1 + self.alpha * loss2


        return sum_loss, logits_txt_vis, logits_vis_act, targets_txt_vis, targets_vis_act

    def process_vis_emb_seq(self, vis_emb_seq, len_frames):
        non_padded_seq = []
        bsz = vis_emb_seq.shape[0]
        for i in range(bsz):
            current_seq_len = len_frames[i]
            current_seq = vis_emb_seq[i, :current_seq_len]    # remove padding

            tmp_mean = torch.zeros(current_seq.shape[0] - 1, 768).to(self.device)
            for j in range(current_seq.shape[0] - 1):
                current_action = torch.stack([current_seq[j], current_seq[j + 1]], dim=0)
                current_action = current_action.unsqueeze(0)
                output = self.seq_fus(current_action).squeeze()  # 1, 768

                tmp_mean[j] = output
            non_padded_seq.append(tmp_mean)

        flattened_vis = torch.cat(non_padded_seq, dim=0)
        #print(f'{flattened_vis.shape=}')
        return flattened_vis



class Enc_lang(nn.Module):
    def __init__(self, enc_transformers, input_dim=512):
        super(Enc_lang, self).__init__()

        self.enc_transformers = enc_transformers
        self.shortcut = nn.Sequential(nn.Linear(input_dim, 768),
                                      nn.GELU())
        self.adapt_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                         nn.GELU(),
                                         nn.Linear(256, 768),
                                         nn.Dropout(0.5),
                                         nn.LayerNorm(768))
    def forward(self, x):
        '''
        :param x: [BSZ, X, 77]
        :return: [BSZ, 768]
        '''
        # Input [BSZ, 768], TODO if the subgoal is also considered, should be [BSZ, len, 768]
        bsz = x.shape[0]
        output_li = []
        for idx in range(bsz):
            x_input = x[idx]
            text_features = self.enc_transformers.encode_text(x_input)
            output_li.append(text_features)

        emb_lang = torch.stack(output_li, dim=0)  # [4, len_lang, 512]
        x_org = self.shortcut(emb_lang)
        output = self.adapt_layer(emb_lang)


        output = x_org + output
        return output  #  [4, len_lang, 768]



class Enc_vis(nn.Module):
    def __init__(self, enc_transformers, input_dim=512):
        super(Enc_vis, self).__init__()
        self.enc_transformers = enc_transformers
        self.shortcut = nn.Sequential(nn.Linear(input_dim, 768),
                                      nn.GELU())
        self.adapt_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                         nn.GELU(),
                                         nn.Linear(256, 768),
                                         nn.Dropout(0.5),
                                         nn.LayerNorm(768))

    def forward(self, x):
        '''
        :param x: [BSZ, img_len(padded after 3rd max length), 3, 224, 224]
        :return: [BSZ, XXX, 768], [BSZ, 768](Representation of Sequence)
        For training, we load pre-computed features as a tensor
        For validation, interacting with the simulator, we have one image at a time
        '''
        emb_li = []
        for idx in range(x.shape[0]):
            embed = self.enc_transformers.encode_image(x[idx])       #[len_Frame, 512]
            emb_li.append(embed)
        emb_li = torch.stack(emb_li, dim=0)

        x_org = self.shortcut(emb_li)
        output = self.adapt_layer(emb_li)
        output = x_org + output


        return output # [BSZ, len_frame, 768]



if __name__ == '__main__':
    from dataset.Alfred import Alfred
    from config.config import CFG
    from torch.utils.data import Dataset, DataLoader
    arg = CFG()
    data = Alfred(arg, mode="valid_seen",
                  root_dir="/home/nfs/inf6/data/datasets/alfred/generated_2.1.0")  # /home/nfs/inf6/data/datasets/alfred/full_2.1.0/z/
    dataloader = DataLoader(data, batch_size=2, shuffle=False, collate_fn=data.my_collate_fn)
    lang_li, img_li, low_action_li, lang_len, lengths_frames, max_img, traj_data_li, subgoal = next(iter(dataloader))
    print(lang_li.shape)
    print(img_li.shape)
    print(low_action_li.shape)
    print("---"*40)
    model = ContrastiveModel().to(arg.device)
    #model = SuperContrastiveModel().to(arg.device)
    lang_li, img_li, low_action_li, lang_len, lengths_frames, max_img, traj_data_li, subgoal = lang_li.to(arg.device), img_li.to(arg.device), low_action_li.to(arg.device), lang_len.to(arg.device), lengths_frames.to(arg.device), max_img.to(arg.device), traj_data_li, subgoal.to(arg.device)
    out, *_ = model((lang_li, img_li, low_action_li, lang_len, lengths_frames, max_img, traj_data_li, subgoal))

    #pre_trained_dir = "saved_models/contrast_checkpoint_epoch_30.pth"
    #model.loiad_state_dict(torch.load(pre_trained_dir)['model_state_dict'])
    #print(model.state_dict().keys())

    #import torch
    #from PIL import Image

    #model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

    #image = preprocess(Image.open("000000000.png")).unsqueeze(0)
    #print(type(image), image.shape)



#### SUper contrastive loss forward backup code
"""
    def forward(self, alfred_input):
        '''
        :param alfred_input:
        lang [BSZ, X, 77] -> txt_emb [BSZ, 768]
        vis [BSZ, Y, 3, 224, 224] -> vis_emb [BSZ, Y, 768] (txt-vis)
                                     vis_emb [BSZ * Y, 768] (vis-act)
        act [BSZ, Y] -> act_emb [n_act, 768]

        GT(txt-vis):
        '''
        #TODO based on the padding, remove the padding
        lang, vis, act, _, len_frames, *_ = alfred_input
        lang, vis, act = lang.to(self.device), vis.to(self.device), act.to(self.device)
        print(lang.shape, vis.shape, act.shape)
        print(len_frames)

        act_label = act.view(-1)
        print(act_label)
        unique_elements = torch.unique(act_label)
        unique_elements_li = unique_elements.tolist()
        targets_vis_act = torch.zeros((vis.shape[0] * vis.shape[1], len(unique_elements_li))).to(self.device)
        for i, label_idx in enumerate(act_label):
            targets_vis_act[i, unique_elements_li.index(label_idx)] = 1
        print(targets_vis_act)

        txt_emb = self.lang_emb(lang)
        vis_emb = self.vis_emb(vis)

        vis_emb_seq = vis_emb / vis_emb.norm(dim=-1, keepdim=True)
        vis_emb = torch.mean(vis_emb_seq, dim=1)
        txt_emb = torch.mean(txt_emb / txt_emb.norm(dim=-1, keepdim=True), dim=1)

        logits_txt_vis = (vis_emb @ txt_emb.T) / self.temperature
        logits_txt_vis = logits_txt_vis.log_softmax(dim=-1)    # .softmax(dim=-1)  IF KL divergence, we need softmax

        targets_txt_vis = torch.eye(lang.shape[0]).to(self.device)

        texts_loss = KLDivergence(logits_txt_vis, targets_txt_vis)
        images_loss = KLDivergence(logits_txt_vis.T, targets_txt_vis.T)
        loss1 = (images_loss + texts_loss) / 2.0

        # vis_emb_samplewise = self.vis_emb_per_sample(vis)
        vis_emb_seq = vis_emb_seq.view(-1, 768)
        act_emb = self.act_emb(unique_elements)

        act_emb = act_emb / act_emb.norm(dim=-1, keepdim=True)


        logits_vis_act = (vis_emb_seq @ act_emb.T) / self.temperature
        logits_vis_act = logits_vis_act.log_softmax(dim=-1)

        images_loss2 = KLDivergence(logits_vis_act, targets_vis_act)
        act_loss = KLDivergence(logits_vis_act.T, targets_vis_act.T)
        loss2 = (images_loss2 + act_loss) / 2.0


        sum_loss = (1 - self.alpha) * loss1 + self.alpha * loss2


        return sum_loss, logits_txt_vis, logits_vis_act, targets_txt_vis, targets_vis_act"""