import os
import torch.utils.data as data
import numpy as np
import random
import json
from tokenizer import *
import torch


def build_causal_attention_mask(context_length, top_left_square = None):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.ones(context_length, context_length, requires_grad=False)
    #mask.fill_(float("-inf"))
    #mask.triu_(1)  # zero out the lower diagonal

    # unmask the lower triangle with 1
    mask.triu_(0)
    if top_left_square:
        mask[:top_left_square, :top_left_square] = 1

    # mask the text tokens, make a deal ????????
    # mask[top_left_square:, :] = 0

    # don't mask text with video and audio
    # if top_left_square:
    #     mask[:top_left_square, :] = 1

#     return mask

# def build_causal_attention_mask(context_length, top_left_square = None):
#     mask = torch.ones(context_length, context_length, requires_grad=False)
#     mask.triu_(0)
#     if top_left_square:
#         mask[:top_left_square, :top_left_square] = 1

#     return mask

# -inf mask
def build_causal_attention_mask(context_length, top_left_square = None):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.zeros(context_length, context_length, requires_grad=False)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal

    # unmask the lower triangle with 0
    #mask.triu_(0)
    if top_left_square:
        mask[:top_left_square, :top_left_square] = 0

    return mask


class MMDataset(data.Dataset):
    def __init__(self, root_dir='/data1/wangqiurui/code/datasets/fs/finefs/', 
                split='train',  context_length=100,
                vocab_file="/data1/wangqiurui/code/helping/ImageBind/imagebind/bpe/bpe_simple_vocab_16e6.txt.gz", 
                num_tokens_per_video: int = 2,
                prompt: str = "How did the athlete perform in the show?"
            ):

        self.st = SimpleTokenizer(vocab_file, context_length)
        self.prompt = prompt

        self.root = root_dir
        self.split = split
        if not os.path.exists(root_dir):
            raise Exception('No such directory: {}'.format(root_dir))

        self.video_path = os.path.join(self.root, "i3d")
        self.audio_path = os.path.join(self.root, "vggish")
        #self.video_list = sorted(os.listdir(self.video_path),key= lambda x:int(x))
        #self.audio_list = sorted(os.listdir(self.audio_path),key= lambda x:int(x))

        # mismatch
        if self.split == 'train':
            self.label_path = os.path.join(self.root, "annotation")
        elif self.split == 'val':
            self.label_path = os.path.join(self.root, "val_annotation")

        # need get video list from annotation file
        self.label_list = os.listdir(self.label_path)
        # remove the .json suffix from each label file name
        self.video_list = [label[:-5] for label in self.label_list]
        
        self.context_length = context_length

        self.num_tokens_per_video = num_tokens_per_video


    def __getitem__(self, index):
        
        video_dir = os.path.join(self.video_path, self.video_list[index] + '_rgb.npy')
        video_feature = torch.tensor(np.load(video_dir))

        audio_dir = os.path.join(self.audio_path, self.video_list[index] + '_vggish.npy')
        audio_feature = torch.tensor(np.load(audio_dir))

        prompt_ids = torch.tensor(self.st.encode(self.prompt)).squeeze(0)
        pad_ids = torch.tensor([self.st.pad_token])

        try:
            label_path = os.path.join(self.label_path, f'{self.video_list[index]}.json')
            with open(label_path, 'r') as f:
                js = json.load(f)
                text_caption = js['simple_caption']
                tes = js['total_element_score']
                pcs = js['total_program_component_score(factored)']
                input_ids = torch.tensor(self.st.encode(text_caption)).squeeze(0)
                if "finefs" in self.root:
                    locate_label = js['locate_labels']
                else:
                    locate_label = [1]
    
        except Exception as e:
            print(f"Error loading label for video {self.video_list[index]}: {e}")
        
        label = {'input_ids': input_ids, 'prompt_ids' : prompt_ids, 'tes': tes, 'pcs': pcs, 'locate_label': locate_label, 'sample_name': self.video_list[index], 'pad_ids': pad_ids}

        # no inputs, mask
        return label, video_feature, audio_feature


    def __len__(self):
        return len(self.video_list)

if __name__ == '__main__':
    dataset = MMDataset()
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    for i, (label, video_feature, audio_feature) in enumerate(dataloader):
        print(label, video_feature.shape, audio_feature.shape)
        break