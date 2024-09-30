import sys
import time

sys.path.append("/home/user/wang01/ma_wang")
import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config.config import CFG
import json
import random
import open_clip
from PIL import Image
import numpy as np
import mobileclip
from utils.constants import LOW_ACTION_DICT, ET_OBJ2IDX, TASK_TYPE_DICT
from transformers import BertTokenizer as Tokenizer
from transformers import BertForSequenceClassification
from utils.inference_util import delete_extra_key, get_target_objs, get_highlevel_transcripts


##########
# This is a cleaner version of the Alfred datasetloading
# This is aim for end 2 end training, which only load the pre-trained CLIP feature of image and clip tokenized text

# Author: Yihao Wang
##########


class Alfred(Dataset):
    def __init__(self, arg, mode, root_dir, LP_module=False, maps=False):
        super(Alfred, self).__init__()
        assert mode in ["train", "valid_seen", "valid_unseen", "test_seen", "test_unseen"]
        self.mode = mode
        self.arg = arg
        self.root_dir = root_dir
        self.root_v = os.path.join(root_dir, mode)
        self.samples = os.listdir(self.root_v)
        self.trials = [os.listdir(os.path.join(self.root_v, f)) for f in self.samples]


        samples_trials = [os.path.join(self.samples[idx], self.trials[idx][sidx]) for idx in
                          range(len(self.samples)) for sidx in range(len(self.trials[idx]))]
        self.maps = maps

        # self.samples_trials[idx]: pick_and_place_simple-Candle-None-Toilet-407/trial_T20190909_055248_059513
        # => TODO change it to "pick_and_place_simple-Candle-None-Toilet-407/trial_T20190909_055248_059513"/pp/ann_x.json
        self.samples_trials_all = []
        #self.samples_trials_all.append(
        #    "/home/nfs/inf6/data/datasets/alfred/full_2.1.0/pick_and_place_with_movable_recep-Pen-Bowl-Dresser-327/trial_T20190909_102323_673895/pp/ann_2.json")
        root_v = self.root_dir.replace("generated_2.1.0", "full_2.1.0/z/")
        dir_tmp = root_v.split("/z/")
        for s in samples_trials:
            tmp_ann_root = os.path.join(dir_tmp[0], s, "pp")
            anns = os.listdir(tmp_ann_root)
            self.samples_trials_all.extend([os.path.join(tmp_ann_root, a) for a in anns])
        #print(self.samples_trials_all[0])

        # Use the ET's numeralization as the tokenizier
        #self.vocab = torch.load("/home/user/wang01/ma_wang/dataset/human.vocab")

        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        #self.tokenizer = mobileclip.get_tokenizer('mobileclip_s0')
        self.custom_transform = transforms.Compose([transforms.Resize((224, 224)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.49, 0.42, 0.34], [0.18, 0.18, 0.17])])
        self.backbone, _, _ = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained=self.arg.clip_root)
        self.backbone.to(self.arg.device)

        # for name, param in self.backbone.named_parameters():
        #    print(f"Parameter: {name}, dtype: {param.dtype}")

        self.pad_txt = self.tokenizer(["<<pad>>"]) # Think of padding, maybe with zero
        pad_img = self.custom_transform(Image.open("/home/nfs/inf6/data/datasets/alfred/pad.jpg"))

        with torch.no_grad():
            #self.pad = self.backbone.encode_text(pad_txt.to(self.arg.device))
            self.pad_img = self.backbone.encode_image(pad_img.unsqueeze(dim=0).to(self.arg.device))
            # print(self.pad.norm(p=2, dim=-1), self.pad_img.norm(p=2, dim=-1))
            #self.pad /= self.pad.norm(dim=-1, keepdim=True)  # self.pad.norm(dim=-1) the euclidean norm of the each row is 1
            self.pad_img /= self.pad_img.norm(dim=-1, keepdim=True)
            # print(self.pad.norm(p=2, dim=-1), self.pad_img.norm(p=2, dim=-1))


    def __getitem__(self, idx):
        # Loading json file
        '''
        root_v = self.root_dir.replace("generated_2.1.0", "full_2.1.0/z/")
        dir_tmp = root_v.split("/z/")
        rndNum = random.randint(0, 2)
        trial_root = os.path.join(dir_tmp[0], self.samples_trials[
            idx])  # full_2.1.0/task_name(look_at_obj...)/trial_name(trial_T20...)/
        json_dir = os.path.join(trial_root, "pp/ann_{}.json".format(rndNum))
        if not os.path.exists(json_dir):
            json_dir = os.path.join(trial_root, "pp/ann_0.json")
        '''
        # print(self.samples_trials_all[idx])
        with open(self.samples_trials_all[idx]) as f:
            traj = json.load(f)
        f.close()

        trial = self.samples_trials_all[idx].split("/pp/")
        trial_root = trial[0].replace("full_2.1.0", "generated_2.1.0/{}".format(self.mode))
        # trial_root = os.path.join(self.root_dir, self.mode, self.samples_trials[idx])
        image_features = torch.load(os.path.join(trial_root, "CLIP_DB_VIS.pt"))  # CLIP_VIS_Pre
        #text_features = torch.load(os.path.join(trial_root, "CLIP_TRI_TXT.pt"))  # CLIP_VIS_Pre


        #img_feat_root = os.path.join(self.samples_trials_all[idx], "raw_images")
        #first_half = img_feat_root.split("/pp/")
        #second_half = img_feat_root.split(".json/")
        #img_feat_root = os.path.join(first_half[0], second_half[-1])
        # TODO add /z/mode in between

        #raw_img_tensor = torch.load(os.path.join(img_feat_root, "raw_as_tensor.pt"))  # [len_current_sample_img, 3, 224, 224]
        #with torch.no_grad():
        #    image_features = self.backbone.encode_image(
        #        raw_img_tensor.to(self.arg.device))  # [len_current_sample_img, 512]
        #    image_features /= image_features.norm(dim=-1, keepdim=True)

        # Use CLIP
        # Loading language
        goal_instr = traj['turk_annotations']['anns'][0]['task_desc'] + " <<goal>>"
        subgoal_instr = traj['turk_annotations']['anns'][0]['high_descs']

        instruction_li = [goal_instr] + subgoal_instr

        #instruction_li = [goal_instr]

        tokenized_goal_str = self.tokenizer(instruction_li)  # [len_current_sample_txt, 77]
        #non_zero_count = torch.count_nonzero(tokenized_goal_str).item()
        #print(tokenized_goal_str)
        #exit(non_zero_count)


        #with torch.no_grad():
        #    text_features = self.backbone.encode_text(
        #        tokenized_goal_str.to(self.arg.device))  # [len_current_sample_txt, 512]
        #    text_features /= text_features.norm(dim=-1, keepdim=True)

        # Loading action list
        action_low_txt = [d["api_action"]["action"] for d in traj["plan"][
            "low_actions"]]  # If need to change LookDown to LookDown_15, change "api_action" to "discrete_action"
        action_low_txt.append("<<stop>>")
        action_low_txt_2_idx = [LOW_ACTION_DICT[a] for a in action_low_txt]

        subgoal_completed = np.array(traj['num']['low_to_high_idx']) / 25  # self.max_subgoals = 25
        num_actions = len(traj['num']['low_to_high_idx'])  # equal to len(action_low_txt_2_idx)
        goal_progress = [(i + 1) / float(num_actions) for i in range(num_actions)]

        # Boolean mask, indicate each if current low action interact with objects or not
        # assert 1. same length as low action 2.the sum of the list should be equal to the length of objects below
        action_valid_interact = [a['valid_interact'] for a in sum(traj['num']['action_low'], [])]

        # load Object
        object_classes_li = []
        for action in traj['plan']['low_actions']:

            if self.has_interaction(action['api_action']['action']):
                obj_key = ('receptacleObjectId'
                           if 'receptacleObjectId' in action['api_action']
                           else 'objectId')
                object_class = action['api_action'][obj_key].split('|')[0]
                object_classes_li.append(ET_OBJ2IDX[object_class])

        if self.maps:
            map_dir = os.path.join(trial_root, "raw_images")
            '''
            # Load separate map files, stack all and save the.pt feature
            pt_files = sorted([f for f in os.listdir(map_dir) if f.endswith('.pt')])
            loaded_files = [torch.load(os.path.join(map_dir, pt_file)) for pt_file in pt_files]
            for l in loaded_files[::-1]:
                if l.shape != torch.Size([240, 240, 4]):
                    loaded_files.pop()
                else:
                    break
            loaded_files = torch.stack(loaded_files, dim=0).float()
            loaded_files = loaded_files.permute(0,3,1,2)
            down_sampled_map = F.adaptive_avg_pool2d(loaded_files, (60, 60))
            torch.save(down_sampled_map, os.path.join(map_dir, 'map_t.pt'))
            '''
            # Load saved map tensor
            loaded_files = torch.load(os.path.join(map_dir, 'map_t.pt'))
        else:
            loaded_files = 0

        alfred_data = dict(txt_feat=tokenized_goal_str, vis_feat=image_features, low_action_li=action_low_txt_2_idx,
                           traj_data=traj, subgoal_completed=subgoal_completed, goal_progress=goal_progress,
                           object_classes_li=object_classes_li, action_valid_interact=action_valid_interact,
                           sem_map=loaded_files)

        return alfred_data

    def __len__(self):
        return len(self.samples_trials_all)

    def has_interaction(self, action):
        non_interact_actions = ['MoveAhead', 'Rotate', 'Look', '<<stop>>', '<<pad>>', '<<seg>>']
        if any(a in action for a in non_interact_actions):
            return False
        else:
            return True

    @staticmethod
    def numericalize(vocab, words, train=True):
        '''
        converts words to unique integers
        '''
        if not train:
            new_words = set(words) - set(vocab.counts.keys())
            if new_words:
                # replace unknown words with <<pad>>
                words = [w if w not in new_words else '<<pad>>' for w in words]
        return vocab.word2index(words, train=train)

    ##################################################################################################################################
    # A customed collate_fn to deal with data with different length
    def my_collate_fn(self, batch):
        bsz = len(batch)
        length_lang = [batch[idx]['txt_feat'].shape[0] for idx in range(bsz)]
        max_lang = max(length_lang)

        lang_li = [self.pad_and_truncate_lang(batch[idx]['txt_feat'], max_lang).to(self.arg.device) for idx in range(bsz)]
        lang_tensor = torch.stack(lang_li, dim=0)   # [BSZ, max_lang, 77]


        length_frames = [batch[idx]["vis_feat"].shape[0] for idx in range(bsz)]

        #print(length_frames)
        try:
            max_img = sorted(length_frames)[-1]
            if max_img > 20:
                max_img = 20
        except:  # During inference
            max_img = length_frames[0]

        length_frames = [l if l < max_img else max_img for l in length_frames]

        img_li = [self.pad_and_truncate_vis(batch[idx]["vis_feat"].to(self.arg.device), max_img) for idx in range(bsz)]
        img_li = torch.stack(img_li, dim=0)

        length_lang = torch.as_tensor(length_lang).to(self.arg.device)

        length_frames = torch.as_tensor(length_frames).to(self.arg.device)
        length_action = length_frames

        act_li = [self.pad_and_truncate_act_li(batch[idx]["low_action_li"], max_img, pad_value=13) for idx in
                  range(bsz)]
        low_action_li = torch.stack(act_li, dim=0).to(self.arg.device)

        traj_data_li = [batch[idx]["traj_data"] for idx in range(bsz)]

        # subgoal and goalprogress are computed based on the action list, so the max length among mini-batch is equal to max_img
        sub_goal_completion = [self.pad_and_truncate_subgoal(torch.from_numpy(batch[idx]['subgoal_completed']), max_img)
                               for idx in range(bsz)]
        sub_goal_completion_tensor = torch.stack(sub_goal_completion, dim=0).to(self.arg.device)

        goal_progress_gt = [torch.as_tensor(self.pad_and_truncate_goal_progress(batch[idx]['goal_progress'], max_img))
                            for idx in range(bsz)]
        goal_progress_tensor = torch.stack(goal_progress_gt, dim=0).to(self.arg.device)

        obj_classes_list = [batch[idx]['object_classes_li'] for idx in range(bsz)]
        action_valid_interact_li = [self.pad_and_truncate_act_li(batch[idx]['action_valid_interact'], max_img, 0) for
                                    idx in range(bsz)]
        action_valid_interact_tensor = torch.stack(action_valid_interact_li, dim=0)

        if torch.is_tensor(batch[0]["sem_map"]):
            # Sem map has the shape [Length, 4, 60, 60]
            sem_map_li = [batch[idx]["sem_map"] for idx in range(bsz)]

            # sem_map_li is a list of tensor
            sem_map_li_t = [self.pad_and_truncate_sem_map(s, max_img) for s in sem_map_li]
            sem_map_li_t = torch.stack(sem_map_li_t, dim=0).to(self.arg.device)
        else:
            sem_map_li_t = torch.empty(1)

        return lang_tensor, img_li, low_action_li,length_lang, length_frames, length_action, traj_data_li, sub_goal_completion_tensor, \
            goal_progress_tensor, obj_classes_list, action_valid_interact_tensor, sem_map_li_t

    def pad_and_truncate_lang(self, input_li, max_len):
        # Input_li [XXX, 77]
        current_len = input_li.shape[0]
        if current_len >= max_len:
            return input_li[:max_len, :]  # .to(self.arg.device)
        else:
            #padding = torch.zeros((max_len - current_len), 77)
            padding = [self.pad_txt] * (max_len - current_len)
            padding_tensor = torch.cat(padding, dim=0)

            return torch.cat((input_li, padding_tensor), dim=0)



    def pad_and_truncate_vis(self, input_li, max_len):
        # Visual input [X, 512]
        current_len = input_li.shape[0]
        if current_len > max_len:
            return input_li[:max_len, :]  # .to(self.arg.device)
        else:
            padding = self.pad_img.repeat((max_len - current_len), 1)
            return torch.cat((input_li, padding), dim=0)

    def pad_and_truncate_act_li(self, act_li, max_len, pad_value):
        # for single 1 dimension low action list
        current_len = len(act_li)
        if current_len > max_len:
            return torch.tensor(act_li[:max_len])  # .to(self.arg.device)
        else:
            act_li.extend([pad_value] * (max_len - current_len))
            return torch.tensor(act_li)  # .to(self.arg.device)

    def pad_and_truncate_subgoal(self, lang_t, max_len):
        current_len = lang_t.shape[0]
        if current_len > max_len:
            return lang_t[:max_len]  # .to(self.arg.device)
        else:
            padding = torch.zeros((max_len - current_len), dtype=lang_t.dtype)
            return torch.cat((lang_t, padding), dim=0)  # .to(self.arg.device)

    def pad_and_truncate_goal_progress(self, goal_prg, max_len):
        current_len = len(goal_prg)
        if current_len > max_len:
            return goal_prg[:max_len]  # .to(self.arg.device)
        else:
            goal_prg.extend([1.0] * (max_len - current_len))
            return goal_prg

    def pad_and_truncate_sem_map(self, sem_map_li, max_len):
        # sem_map_li has shape [BSZ, X, 4, 60, 60]
        # The length of semantic map list is two element less than vision and action, since the semantic map was not saved for the first two steps
        last_element = sem_map_li[-1].unsqueeze(0)
        repeated_elements = last_element.repeat(2, 1, 1, 1)  # Shape [2, 240, 240, 4]      [2, 4, 60, 60]
        # Concatenate the repeated elements with the original tensor
        new_tensor = torch.cat([last_element, repeated_elements], dim=0)  # This should be as long as the action
        current_len = new_tensor.shape[0]
        if current_len > max_len:
            return new_tensor[:max_len, :]
        else:
            padding_value = torch.zeros((max_len - current_len), 4, 60, 60)
            padded = torch.cat([new_tensor, padding_value], dim=0)
            return padded

    ##################################################################################################################################

    def LP_processing(self, goal_instruction):
        encoding = self.BERT_tokenizer(goal_instruction, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to("cuda:0")
        attention_mask = encoding['attention_mask'].to("cuda:0")
        outputs = self.model_task_type(input_ids, attention_mask=attention_mask)
        maxed = torch.max(outputs.logits, 1)
        y_hat = maxed.indices.item()
        task_type = TASK_TYPE_DICT[y_hat]
        mrecep_target = get_target_objs(goal_instruction, self.model_mrecep_type, self.BERT_tokenizer, 1)
        obj_target = get_target_objs(goal_instruction, self.model_obj_type, self.BERT_tokenizer, 0)
        parent_target = get_target_objs(goal_instruction, self.model_parent_type, self.BERT_tokenizer, 2)
        sliced = get_target_objs(goal_instruction, self.model_sliced_type, self.BERT_tokenizer, 4)
        obj_li = [mrecep_target, obj_target, parent_target]
        high_level_instructions = get_highlevel_transcripts(task_type=task_type, mrecep_target=mrecep_target,
                                                            obj_target=obj_target,
                                                            parent_target=parent_target, sliced=sliced)
        return high_level_instructions

    def remove_spaces_and_lower(self, s):
        cs = ' '.join(s.split())
        cs = cs.lower()
        return cs

    @staticmethod
    def numericalize(vocab, words, train=True):
        '''
        converts words to unique integers
        '''
        if not train:
            new_words = set(words) - set(vocab.counts.keys())
            if new_words:
                # replace unknown words with <<pad>>
                words = [w if w not in new_words else '<<pad>>' for w in words]
        return vocab.word2index(words, train=train)
    ###########################



###########################################################################################################################

if __name__ == '__main__':
    '''
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    #image = preprocess(Image.open("docs/CLIP.png")).unsqueeze(0)
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    text = tokenizer(["a diagram", "a dog", "a cat"])
    print(text.shape)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text)
        print(text_features.shape)
    '''




    # are you familiar with CLIP from openAI(connecting text and images)? If i have a relative long
    from tqdm import tqdm
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
                        default=False)  # Use pre-aligned clip model, if False load normal embedding layer
    parser.add_argument('--pre_backbone_dir', type=str,
                        default="MA/triple_contrast_checkpoint_epoch_18.pth")  # MA/3mobile_contrast_checkpoint_epoch_9.pth
    parser.add_argument('--FEAT_precomputed', type=bool, default=True)
    parser.add_argument('--sem_map_module', type=bool, default=False)

    parser.add_argument('--device', default="cuda:0" if torch.cuda.is_available() else "cpu")

    arg = parser.parse_args()

    # arg = CFG()
    # data = Alfred(arg, mode="valid_unseen", root_dir=arg.root_dir, maps=True)
    # Drop last since cls token is initialized with the batch size
    # trainloader = DataLoader(data, batch_size=arg.BSZ, shuffle=False, collate_fn=data.my_collate_fn, drop_last=False)

    # data = AlfredPreFeat(arg, mode="train", root_dir=arg.root_dir)
    data = Alfred(arg, mode="valid_seen", root_dir=arg.root_dir)

    trainloader = DataLoader(data, batch_size=arg.BSZ, shuffle=False, collate_fn=data.my_collate_fn, drop_last=False)

    print("Train data loaded")
    lang_li, img_li, low_action_li, length_txt, length_frames, length_action, traj_data_li, sub_goal_completion_tensor, \
        goal_progress_tensor, obj_classes_list, action_valid_interact_tensor, sem_map_li_t = next(iter(trainloader))
    print(lang_li.shape)
    print(img_li.shape)
    print()
    # length_action = length_frames maybe not needed both
    # for idx, batch in enumerate(tqdm(trainloader)):
    #    i = 0

    # valid_data = Alfred(arg, mode="valid_seen", root_dir=arg.root_dir)
    # validloader = DataLoader(valid_data, batch_size=arg.BSZ, shuffle=False, collate_fn=valid_data.my_collate_fn,
    #                         drop_last=False)
    # print("Valid data loaded")
    # test_data = Alfred(arg, mode="valid_unseen", root_dir=arg.root_dir)
    # testloader = DataLoader(test_data, batch_size=arg.BSZ, shuffle=False, collate_fn=test_data.my_collate_fn,
    #                        drop_last=False)
    # print("Test data loaded")
    # print(len(dataloader))
    # for idx, batch in enumerate(tqdm(dataloader)):
    #    i = 0

    ## Code for checking output in collate_fn
    # print(f'{lang_li.shape=}')
    # print(f'{img_li.shape=}')
    # print(f'{low_action_li.shape=}')
    # print(f'{sub_goal_completion_tensor.shape=}')
    # print(length_txt)
    # print(length_frames)
    # print(length_action)
    # Device checking
    # print(lang_li.get_device(), img_li.get_device(), low_action_li.get_device(), length_txt.get_device(),
    #      length_frames.get_device(), length_action.get_device(), sub_goal_completion_tensor.get_device())
