import sys

sys.path.append("/home/user/wang01/ma_wang")
import json
import torch
from torch import nn
from model.encoding import PosEncoding_ET, generate_attention_mask, generate_attention_mask_withmap
#from dataset.MyAlfredE2E import Alfred
from dataset.MyAlfred import Alfred
from config.config import CFG
from tqdm import tqdm
import open_clip
#import mobileclip
#from model.ET_openCLIP_loadPre import LoadPreTrainedModel
from utils.constants import LOW_ACTION_IDX


##########
# This is a cleaner version of the ET model with CLIP as the backbone
# Input is loaded in the Dataset/Dataloader to save computation cost during training
# Author: Yihao Wang
##########

class ET_openclip(nn.Module):
    def __init__(self, arg, num_modals, aux):
        super(ET_openclip, self).__init__()
        self.arg = arg
        self.enc_action = nn.Embedding(len(LOW_ACTION_IDX), 768)  # 12 actions, 2 tokens (pad, stop)
        self.dropout_act = nn.Dropout2d(0.1)
        # TODO load the acition embedding layer from pre-trained model or train it from scratch
        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32',pretrained='laion2b_s34b_b79k')  # 'ViT-B-32', pretrained='laion2b_s34b_b79k'

        #self.clip_model, _, _ = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='/home/nfs/inf6/data/datasets/alfred/mobile_clip_checkpoints/mobileclip_s0.pt')
        if self.arg.self_pretrain:  # Pre-trained clip, load the action embedding
            print("Load Pre-aligned Backbone...................................")
            self.pre_trained_model = torch.load(self.arg.pre_backbone_dir)
            self.enc_action.weight = nn.Parameter(
                self.pre_trained_model["model_state_dict"]["action_emb.weight"])  # action_emb.weight/act_emb.weight
        else:
            self.enc_action.weight.data.uniform_(-0.1, 0.1)

        # assert torch.allclose(self.enc_action.weight, self.pre_trained_model["model_state_dict"]["action_emb.weight"])

        self.enc_lang = Adapter()
        self.enc_vis = Adapter()

        self.multi_enc = Enc_Multimodal(arg, num_modals=num_modals)
        self.dec_action = nn.Linear(768, 768)
        # Initialize the decoder action
        self.dec_action.bias.data.zero_()
        self.dec_action.weight.data.uniform_(-0.1, 0.1)

        self.aux = aux
        if self.aux:
            self.object_feat = nn.Linear(512, 768)  # A linear layer that used as skip connection. equivalent to FeatureFlat Class from ET
            self.dec_object = nn.Linear(768, 82)

            self.dec_subgoal = nn.Linear(768, 1)
            self.dec_progress = nn.Linear(768, 1)


        if self.arg.sem_map_module:
            self.map_encoder = Map_Encoder()

    def forward(self, alfred_input):
        '''
        :param alfred_input: tuple.
        - lang [BSZ, 77](org. [BSZ, X, 512]) -> txt_emb [BSZ, 768]
        - vis [BSZ, Y, 768](org. [BSZ, Y, 3, 224, 224]) -> vis_emb [BSZ, Y, 768]
        - act [BSZ, Y] -> act_emb [BSZ, Y, 768]
        :return:
        '''
        output = {}
        # Loaded the batch and copy everything on GPU
        text_emb, vis, act, len_lang, len_frames, len_actions, *_, sem_map = alfred_input
        #print("Input following:")
        #print(text_emb.shape, vis.shape, act.shape)

        #lang = torch.tensor(lang).to(torch.int64)

        # TODO Zero shot + double contrastive
        '''
        text_li = []
        for l in text_emb:
            text_feat = self.clip_model.encode_text(l)
            text_features = text_feat / text_feat.norm(dim=-1, keepdim=True)   # [max_len, 512]
            text_li.append(text_features)
        text_emb = torch.stack(text_li, dim=0)
        '''


        emb_lang = self.enc_lang(text_emb)           # [32, 768]
        emb_vis = self.enc_vis(vis)

        emb_act = self.enc_action(act)
        emb_act = self.dropout_act(emb_act)

        if self.arg.sem_map_module:
            emb_map = self.map_encoder(sem_map)  # [BSZ, X, 768]
        else:
            emb_map = None

        # encoder_outputs, _ = self.multi_enc(emb_lang, emb_vis, emb_act, len_lang, len_frames, len_actions,
        #                                    int(max(len_frames)))  # [BSZ, XXX, 768]
        encoder_outputs, _ = self.multi_enc(emb_lang, emb_vis, emb_act, len_lang, len_frames, len_actions,
                                            int(max(len_frames)), True, emb_map)  # [BSZ, XXX, 768]

        encoder_out_visual = encoder_outputs[:, len_lang.max().item():len_lang.max().item() + len_frames.max().item()]

        # encoder_out_visual = encoder_outputs[:,emb_lang.shape[1]: emb_lang.shape[1] + emb_vis.shape[1]]  # remove cls token
        # print(f'{encoder_out_visual.shape=}')

        decoder_input = encoder_out_visual.reshape(-1, 768)
        # print(f'{decoder_input.shape=}')
        action_emb_flat = self.dec_action(decoder_input)
        # print(f'{action_emb_flat.shape=}')
        action_flat = action_emb_flat.mm(self.enc_action.weight.t())
        # print(f'{action_flat.shape=}')
        action = action_flat.view(*encoder_out_visual.shape[:2],
                                  *action_flat.shape[1:])
        # print(f'{action.shape=}')

        if self.aux:
            emb_object = self.object_feat(vis)
            # predict the object
            emb_object_flat = emb_object.view(-1, 768)
            decoder_input = decoder_input + emb_object_flat
            object_flat = self.dec_object(decoder_input)
            objects = object_flat.view(*encoder_out_visual.shape[:2], *object_flat.shape[1:])

            subgoal = torch.sigmoid(self.dec_subgoal(encoder_out_visual))
            progress = torch.sigmoid(self.dec_progress(encoder_out_visual))

            output.update({'action': action, 'object': objects, 'subgoal': subgoal, 'progress': progress})
        else:
            output.update({'action': action})

        return output


class Adapter(nn.Module):
    def __init__(self):
        super(Adapter, self).__init__()

        self.shortcut = nn.Sequential(nn.Linear(512, 768),
                                      nn.GELU())
        self.adapt_layer = nn.Sequential(nn.Linear(512, 256),
                                         nn.GELU(),
                                         nn.Linear(256, 768),
                                         nn.Dropout(0.5),
                                         nn.LayerNorm(768))

    def forward(self, x):
        '''
        :param x: a list of strings, length of the list is the batch size [[X_1, 768],[X_2, 768]...,[X_BSZ, 768]]
        :return: [BSZ, 1, 768]
        '''
        # Input [BSZ, 768], TODO if the subgoal is also considered, should be [BSZ, len, 768]
        # print(f'{x.shape=}')
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x_org = self.shortcut(x)
        output = self.adapt_layer(x)

        output = x_org + output
        return output


class Map_Encoder(nn.Module):
    def __init__(self):
        super(Map_Encoder, self).__init__()

        # Define the CNN to process each semantic map
        # Global Average Pooling
        # self.global_avg_pool = nn.AdaptiveAvgPool2d((20, 20))

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            # nn.Conv2d(in_channels=64, out_channels=768, kernel_size=3, stride=1, padding=1),
            # nn.GELU()
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layer to obtain the final embedding
        # self.fc = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=768, kernel_size=1600),   #replace fc layer
        #                        nn.GELU())
        self.feature_dim = 32 * 30 * 30
        self.fc = nn.Sequential(nn.Linear(self.feature_dim, 768),
                                nn.GELU())

    def forward(self, x):
        batch_size, seq_len, _, _, _ = x.shape  # BSZ, seqlen, 4, 60, 60

        # map_input = x.reshape(batch_size, -1)
        # print(map_input.shape)
        output = self.pool(self.cnn(x.view(batch_size * seq_len, 4, 60, 60)))
        output = output.view(batch_size * seq_len, -1)
        output = self.fc(output)
        # Process each map in the sequence through the CNN and Global Average Pooling
        features = output.view(batch_size, seq_len, -1)
        '''
        for t in range(seq_len):
            map_t = x[:, t, :, :, :]  # Get the t-th map in the sequence

            map_t = map_t.permute(0, 3, 1, 2)
            #gap_out = self.global_avg_pool(map_t) # .view(batch_size, -1)  # Shape: [batch_size, 768]
            #output = gap_out.reshape(batch_size, -1)
            output = self.fc(output)

            #cnn_out = self.cnn(map_t)  # Shape: [batch_size, 768, H, W]
            #print(cnn_out.shape)
            #gap_out = self.global_avg_pool(cnn_out).view(batch_size, -1)  # Shape: [batch_size, 768]
            features.append(output)

        features = torch.stack(features, dim=1)  # Shape: [batch_size, seq_len, cnn_output_dim]
        '''

        # Pass through fully connected layer to get final embedding
        # embedding = self.fc(features)  # Shape: [batch_size, seq_len, cnn_output_dim]

        return features


class Enc_Multimodal(nn.Module):
    def __init__(self, arg, num_modals):
        super(Enc_Multimodal, self).__init__()
        self.arg = arg
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=768, dropout=0.1)
        self.enc_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.enc_pos = PosEncoding_ET(768)
        self.enc_layernorm = nn.LayerNorm(768)
        self.attn_masks = True
        self.num_modals = num_modals
        self.modal_type_embeddings = nn.Parameter(torch.randn(self.num_modals, 768) * 0.1)
        self.enc_pos_learn = None
        self.enc_token = None
        self.enc_dropout = nn.Dropout(0.0, inplace=True)
        self.num_input_actions = 1  # how many last actions to attend to

        # self.cls_token = cls_token

    def forward(self, emb_lang, emb_frames, emb_actions, lengths_lang, lengths_frames, lengths_actions,
                length_frames_max, attn_masks=True, emb_map=None):
        '''

        :param lang_li: [[X_1, 768], [X_2, 768], ..., [X_BSZ, 768]]
        :param vis_li: same as above
        :param action_li:  initial LookDown_15, TODO convert the corresponding input here
        For each element in the list, vis and action should have corresponging same size
        :return:
        '''
        # print("Multi modal model investigation")
        # print("===" * 40)

        # print(emb_lang.shape, emb_frames.shape, emb_actions.shape)
        # print(emb_map.shape)
        # If threshold based cutting exist in the dataset
        # if length_frames_max > 65:
        #    length_frames_max = 65
        # emb_lang is processed on each GPU separately so they size can vary
        length_lang_max = lengths_lang.max().item()
        emb_lang = emb_lang[:, :length_lang_max]

        # create a mask for padded elements
        # ORG length_mask_pad = length_lang_max + length_frames_max * (2 if lengths_actions.max() > 0 else 1)  # lengths_actions.max()
        length_mask_pad = length_lang_max + length_frames_max * (
            (self.num_modals - 1) if lengths_actions.max() > 0 else 1)  # lengths_actions.max()

        mask_pad = torch.zeros((len(emb_lang), length_mask_pad), device=emb_lang.device).bool()

        for i, (len_l, len_f, len_a) in enumerate(zip(lengths_lang, lengths_frames, lengths_actions)):
            # mask padded words
            mask_pad[i, len_l: length_lang_max] = True
            # mask padded frames
            mask_pad[i, length_lang_max + len_f:length_lang_max + length_frames_max] = True
            # mask padded actions
            # ORG mask_pad[i, length_lang_max + length_frames_max + len_a:] = True
            mask_pad[i,
            length_lang_max + length_frames_max + len_a:length_lang_max + length_frames_max + length_frames_max] = True

            if emb_map is not None:
                # mask padded maps
                mask_pad[i, length_lang_max + length_frames_max + length_frames_max + len_a:] = True

        ## Add modal-type embedding
        emb_lang = emb_lang + self.modal_type_embeddings[0]
        emb_frames = emb_frames + self.modal_type_embeddings[1]
        emb_actions = emb_actions + self.modal_type_embeddings[2]
        if self.num_modals == 4:
            emb_map = emb_map + self.modal_type_embeddings[3]

        # encode the inputs
        emb_all = self.encode_inputs(emb_lang, emb_frames, emb_actions, emb_map, lengths_lang, lengths_frames, mask_pad)
        # ORG emb_all = self.encode_inputs(emb_lang, emb_frames, emb_actions, lengths_lang, lengths_frames, mask_pad)
        # create a mask for attention (prediction at t should not see frames at >= t+1)
        if attn_masks:
            # assert length_frames_max == max(lengths_actions)
            # print(f'{emb_lang.shape= }', f'{emb_frames.shape= }')
            # print("length_lang_max: ", length_lang_max)
            # print("length_frame_max: ", length_frames_max)
            if emb_map is not None:
                mask_attn = generate_attention_mask_withmap(length_lang_max, length_frames_max, emb_all.device,
                                                            self.num_input_actions)
            else:
                mask_attn = generate_attention_mask(length_lang_max, length_frames_max, emb_all.device,
                                                    self.num_input_actions)
        else:
            # allow every token to attend to all others
            mask_attn = torch.zeros(
                (mask_pad.shape[1], mask_pad.shape[1]),
                device=mask_pad.device).float()

        # print(f'{emb_all.shape=}')
        # encode the inputs
        # print("Encoded: ")
        # print(emb_lang.shape, emb_frames.shape, emb_actions.shape, emb_map.shape)
        # print(f'{emb_all.shape=}')
        #print(lengths_lang)
        #print(lengths_frames)
        mask_attn = mask_attn.to(torch.bool)
        output = self.enc_transformer(emb_all.transpose(0, 1), mask=mask_attn, src_key_padding_mask=mask_pad).transpose(0, 1)


        return output, mask_pad

    def encode_inputs(self, emb_lang, emb_frames, emb_actions, emb_maps,
                      lengths_lang, lengths_frames, mask_pad):
        '''
        add encodings (positional, token and so on)
        '''
        if self.enc_pos is not None:
            emb_lang, emb_frames, emb_actions, emb_maps = self.enc_pos(
                emb_lang, emb_frames, emb_actions, emb_maps, lengths_lang, lengths_frames)
        if self.enc_pos_learn is not None:
            emb_lang, emb_frames, emb_actions = self.enc_pos_learn(
                emb_lang, emb_frames, emb_actions, lengths_lang, lengths_frames)
        if self.enc_token is not None:
            emb_lang, emb_frames, emb_actions = self.enc_token(
                emb_lang, emb_frames, emb_actions)

        if emb_maps is not None:
            emb_cat = torch.cat((emb_lang, emb_frames, emb_actions, emb_maps), dim=1)
        else:
            emb_cat = torch.cat((emb_lang, emb_frames, emb_actions), dim=1)
        emb_cat = self.enc_layernorm(emb_cat)
        emb_cat = self.enc_dropout(emb_cat)
        return emb_cat


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--root_dir', type=str, default="/home/nfs/inf6/data/datasets/alfred/generated_2.1.0")
    # "/home/local/ET/data/generated_2.1.0/"
    # /home/nfs/inf6/data/datasets/alfred/full_2.1.0/z/
    # /home/user/wang01/alfred/data/full_2.1.0/z/

    # often adjustable argument
    parser.add_argument('--BSZ', default=32, type=int)
    parser.add_argument('-e', '--epochs', default=41)
    parser.add_argument('--nth_max_frame', default=-1)
    parser.add_argument('--acc_step', default=2)

    parser.add_argument('--use_et_generated', default=True)
    parser.add_argument('--clip_root',
                        default='/home/nfs/inf6/data/datasets/alfred/mobile_clip_checkpoints/mobileclip_s0.pt')
    # /home/user/wang01/ml-mobileclip-main/checkpoints/mobileclip_s0.pt'
    parser.add_argument('--self_pretrain', type=bool,
                        default=True)  # Use pre-aligned clip model, if False load normal embedding layer
    parser.add_argument('--pre_backbone_dir', type=str,
                        default="pre_vitb32_triple_10.pth")  # MA/3mobile_contrast_checkpoint_epoch_9.pth
    parser.add_argument('--FEAT_precomputed', type=bool, default=True)
    parser.add_argument('--sem_map_module', type=bool, default=False)

    parser.add_argument('--device', default="cuda:0" if torch.cuda.is_available() else "cpu")

    arg = parser.parse_args()
    data = Alfred(arg, mode="valid_seen",
                  root_dir="/home/nfs/inf6/data/datasets/alfred/generated_2.1.0")  # /home/nfs/inf6/data/datasets/alfred/full_2.1.0/z/
    dataloader = DataLoader(data, batch_size=4, shuffle=False, collate_fn=data.my_collate_fn)
    current_sample = next(iter(dataloader))
    model = ET_openclip(arg, 3, aux=False).to(arg.device)
    a = model(current_sample)
