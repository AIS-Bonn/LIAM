import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import json
from PIL import Image
#from tqdm import tqdm
import open_clip
import mobileclip



actiontxt2label = dict(LookDown=0, LookUp=1, RotateLeft=2, RotateRight=3, MoveAhead=4, PickupObject=5, PutObject=6, SliceObject=7,
                  OpenObject=8, CloseObject=9, ToggleObjectOn=10, ToggleObjectOff=11)   #"<<stop>>", "<<pad>>"

class ActionMapping(Dataset):
    def __init__(self, root_dir, mode):
        self.root_dir = os.path.join(root_dir, mode)
        self.action_dict = dict(LookDown=[], LookUp=[], MoveAhead=[], RotateRight=[], RotateLeft=[],
                                SliceObject=[], PutObject=[], OpenObject=[], CloseObject=[], PickupObject=[],
                                ToggleObjectOn=[], ToggleObjectOff=[])
        self.img_li = []
        self.label_li = []
        for keys in actiontxt2label:
            current_dir = os.path.join(self.root_dir, keys)
            self.action_dict[keys] = len(os.listdir(current_dir))
            cur_img_li, cur_label_li = self.store_list(keys)
            self.img_li.extend(cur_img_li)
            self.label_li.extend(cur_label_li)

    def store_list(self, action):
        img_li = [os.path.join(self.root_dir, action, "{:06d}.pt".format(idx)) for idx in range(self.action_dict[action])]
        label_li = [action] * self.action_dict[action]
        return img_li, label_li

    def __len__(self):
       return len(self.label_li)

    def __getitem__(self, idx):
        feat = torch.load(self.img_li[idx])
        # TODO process input
        feat = torch.mean(feat, dim=0)
        #l = label.index(self.label_li[idx])
        return feat


# Action Mapping raw image
class ActionMappingProcess(Dataset):
    def __init__(self, arg, mode):
        super(ActionMappingProcess, self).__init__()
        assert mode in ["train", "valid_seen", "valid_unseen", "test_seen", "test_unseen"]
        self.arg = arg
        #self.root_v = os.path.join("/home/local/ET/data/generated_2.1.0", mode)
        self.root_v = os.path.join("/home/nfs/inf6/data/datasets/alfred/generated_2.1.0", mode)
        self.samples = os.listdir(self.root_v)
        self.trials = [os.listdir(os.path.join(self.root_v, f)) for f in self.samples]
        self.samples_trials = [os.path.join(self.samples[idx], self.trials[idx][sidx]) for idx in range(len(self.samples)) for sidx in range(len(self.trials[idx]))]
        self.custom_transform = transforms.Compose([transforms.Resize((224, 224)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.49, 0.42, 0.34], [0.18, 0.18, 0.17])])

    def __getitem__(self, idx):
        # TODO change the root_dir to ET main directory
        json_dir = os.path.join(self.root_v, self.samples_trials[idx], "traj_data.json")
        '''
        dir_tmp = self.root_v.split("/z/")
        rndNum = random.randint(0, 2)
        trial_root = os.path.join(dir_tmp[0], self.samples_trials[
            idx])  # full_2.1.0/task_name(look_at_obj...)/trial_name(trial_T20...)/
        json_dir = os.path.join(trial_root, "pp/ann_{}.json".format(rndNum))
        if not os.path.exists(json_dir):
            json_dir = os.path.join(trial_root, "pp/ann_0.json")
        '''
        with open(json_dir) as f:
            traj = json.load(f)
        f.close()

        #save_root = "/home/local/ET/ActionMapping/valid_unseen"
        img_root = os.path.join(self.root_v, self.samples_trials[idx], "raw_images")
        img_name_li = sorted(os.listdir(img_root))
        img_names = [os.path.join(self.samples_trials[idx], "raw_images", i) for i in img_name_li]
        #img_li = [self.custom_transform(Image.open(os.path.join(img_root, i))) for i in img_name_li]

        #img_li_stack = torch.stack(img_li, dim=0)
        action_low_txt = [d["api_action"]["action"] for d in traj["plan"]["low_actions"]]   # If need to change LookDown to LookDown_15, change "api_action" to "discrete_action"
        #action_low_label = [actiontxt2label[a] for a in action_low_txt]
        #print(img_li_stack.shape)
        #print(action_low_label)
        #data_tuple = ["{} {} {}".format(img_names[idx], img_names[idx + 1], action_low_label[idx]) for idx in range(len(action_low_label))]
        data_tuple = []
        for idx in range(len(action_low_txt)):
            try:
                d = "{} {} {}".format(img_names[idx], img_names[idx + 1], action_low_txt[idx])
                data_tuple.append(d)
            except IndexError:
                print(idx, len(action_low_txt), len(img_names))
                print(img_names)
                continue
        #if len(img_li) - len(action_low_txt) == 1:     ## The length of low_action should be one less than img_name_li
        #    for i in range(1, len(img_li)):
        #        action_root_cur = os.path.join(save_root,action_low_txt[i - 1])
        #        if (not os.path.exists(action_root_cur)):
        #            os.makedirs(action_root_cur)

        #        action_map = ([self.custom_transform(img_li[i - 1]), self.custom_transform(img_li[i])], action_low_txt[i - 1])
        #        #action_map_trial.append(action_map)

                #save_feat(img_li[i - 1], img_li[i], action_low_txt[i - 1], len(os.listdir(action_root_cur)), self.model, self.preprocess)

        
        '''
        # Save combined image features
        for idx, img in enumerate(img_li):

            img_processed = self.preprocess(Image.open(img)).unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = self.model.encode_image(img_processed)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                img_tensor.append(image_features)

        img_tensor_combined = torch.cat(img_tensor)


        pre_saved_file = os.path.join(save_root, "vis_seq_feat.pt")
        if os.path.isfile(pre_saved_file):
            os.remove(pre_saved_file)
        pre_saved_file = os.path.join(save_root, "VIS_feat.pt")
        torch.save(img_tensor_combined, pre_saved_file)
        '''
        return data_tuple

    def __len__(self):
        return len(self.samples_trials)


# Action Mapping raw image
class ActionMappingRaw(Dataset):
    def __init__(self, mode, arg):
        super(ActionMappingRaw, self).__init__()
        assert mode in ["train", "valid_seen", "valid_unseen", "test_seen", "test_unseen"]
        #self.root_v = os.path.join("/home/local/ET/data/generated_2.1.0", mode)

        self.root_v = os.path.join("/home/nfs/inf6/data/datasets/alfred/generated_2.1.0", mode)
        self.arg = arg

        #self.samples = os.listdir(self.root_v)
        #self.trials = [os.listdir(os.path.join(self.root_v, f)) for f in self.samples]
        #self.samples_trials = [os.path.join(self.samples[idx], self.trials[idx][sidx]) for idx in
        #                       range(len(self.samples)) for sidx in range(len(self.trials[idx]))]
        with open("{}_num.txt".format(mode), "r") as f:
            self.samples = f.readlines()
        f.close()
        self.custom_transform = transforms.Compose([transforms.Resize((224, 224)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.49, 0.42, 0.34], [0.18, 0.18, 0.17])])

        #self.tokenizer = open_clip.get_tokenizer('ViT-B-32')  #RN50
        self.tokenizer = open_clip.get_tokenizer('RN50')  #RN50

        #self.tokenizer = mobileclip.get_tokenizer('mobileclip_s0')

        _, _, self.preprocess = open_clip.create_model_and_transforms('RN50', pretrained='cc12m')
        #_, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        #_, _, self.preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained=self.arg.clip_root)



    def __getitem__(self, idx):
        sample = self.samples[idx].split()
        #img1 = self.custom_transform(Image.open(os.path.join(self.root_v, sample[0])))
        #img2 = self.custom_transform(Image.open(os.path.join(self.root_v, sample[1])))

        img1 = self.preprocess(Image.open(os.path.join(self.root_v, sample[0])))
        img2 = self.preprocess(Image.open(os.path.join(self.root_v, sample[1])))
        img = torch.stack([img1, img2])
        label = sample[2]
        #label = self.tokenizer(label)
        return img, torch.tensor(int(label))

    def __len__(self):
        return len(self.samples)


# Currently only use data from training
def save_feat(img1, img2, action, idx, model, preprocess):
    '''

    :param img1: String: directory
    :param img2: String: directory
    :param action: The action class
    :param idx: save index
    :return:
    '''
    root_dir = "/home/local/ET/ActionMapping/valid_unseen"
    img1_pre = preprocess(Image.open(img1)).unsqueeze(0)
    img2_pre = preprocess(Image.open(img2)).unsqueeze(0)
    with torch.no_grad(), torch.cuda.amp.autocast():
        img_feat1 = model.encode_image(img1_pre)
        img_feat2 = model.encode_image(img2_pre)  # Check if dim = [1,512]
        img_feat = torch.cat([img_feat1, img_feat2])
    save_dir = os.path.join(root_dir, action, "{:06d}.pt".format(idx))
    torch.save(img_feat, save_dir)




if __name__ == '__main__':
    train_dataset = ActionMappingRaw("train")  # Is this traj based?
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    a = next(iter(train_loader))
    '''
    with open("test.txt", "w") as f:
        for idx, d in enumerate(tqdm(train_loader)):
            for sample in d:
                f.write(sample[0] + "\n")
    f.close()
    '''



    #data_tuple = next(iter(train_loader))
    #print(data_tuple)
    #print(feat.shape, l.shape)
