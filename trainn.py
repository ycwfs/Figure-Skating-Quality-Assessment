from torch.nn import CrossEntropyLoss, MSELoss
from torch import nn, optim, tensor
import hydra
import torch
import numpy as np
import torch.nn.functional as F
from data import *
from mllm import MMModel
from tokenizer import *
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from scipy.stats import spearmanr
from loguru import logger
import sys


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def val(val_loader, mmm, score_cri, locate_cri, epoch, root_dir):
    mmm.eval()
    losses = []
    tes_losses = []; pcs_losses = []; locate_losses = []
    tes_true = []; pcs_true = []
    tes_pred = []; pcs_pred = []

    print(f"------------------------------epoch {epoch+1} valding------------------------------")
    for j, data in enumerate(val_loader):
        labels, video_feature, audio_feature = data
        video_feature = video_feature.to(device,dtype = torch.float32); audio_feature = audio_feature.to(device,dtype = torch.float32)
        
        # float, float, input_ids(batch_size, max_seq_length)
        tes = labels['tes']; pcs = labels['pcs']; caption = labels['input_ids']; prompt_ids = labels['prompt_ids']; locate_label = torch.tensor(labels['locate_label']); sample_name = labels['sample_name']
        pad_ids = labels['pad_ids'].to(device)
        
        tes_true.append(tes.numpy()); pcs_true.append(pcs.numpy())
        tes = tes.to(device, dtype = torch.float32); pcs = pcs.to(device, dtype = torch.float32); caption = caption.to(device); prompt_ids = prompt_ids.to(device); locate_label = locate_label.to(device, dtype = torch.float32)

        mmm.eval()
        p_tes, p_pcs, hidden_state, located, shift_language = mmm(video_feature, audio_feature, caption, prompt_ids, pad_ids)
        tes_pred.append(p_tes.detach().cpu().numpy()); pcs_pred.append(p_pcs.detach().cpu().numpy())

        tes_loss = score_cri(p_tes, tes)
        pcs_loss = score_cri(p_pcs, pcs)
        tes_losses.append(tes_loss.item()); pcs_losses.append(pcs_loss.item())
        if locate_label.shape[0] > located.shape[0]:
            locate_label = locate_label[:located.shape[0]]
        elif locate_label.shape[0] < located.shape[0]:
            located = located[:locate_label.shape[0]]

        if "finefs" in root_dir:
            locate_loss = locate_cri(located, locate_label)
            locate_losses.append(locate_loss.item())

        if "finefs" in root_dir:
            loss = tes_loss + pcs_loss + locate_loss
        else:
            loss = tes_loss + pcs_loss
        losses.append(loss.item())

    
    tes_spear = spearmanr(tes_pred, tes_true).correlation
    pcs_spear = spearmanr(pcs_pred, pcs_true).correlation
    avg_tes_losses = sum(tes_losses) / len(tes_losses)
    avg_pcs_losses = sum(pcs_losses) / len(pcs_losses)
    if "finefs" in root_dir:
        avg_locate_losses = sum(locate_losses) / len(locate_losses)
    else:
        avg_locate_losses = 0

    print(f"val res: tes_spear: {tes_spear}, pcs_spear: {pcs_spear}, avg_tes_losses: {avg_tes_losses}, avg_pcs_losses: {avg_pcs_losses}, avg_locate_losses: {avg_locate_losses},all_loss: {sum(losses)/len(losses)} \n")

    return tes_spear, pcs_spear, avg_tes_losses, avg_pcs_losses


@hydra.main(config_path="config", config_name="config",version_base=None)
def train(cfg):
    if cfg.concat_strategy not in ['insert', 'embed_hack']:
        raise ValueError(f"Invalid concat strategy: {cfg.concat_strategy}, must be one of 'insert' or 'embed_hack'.")

    if cfg.concat_strategy == 'insert':
        num_tokens_per_video = 4
    else:
        num_tokens_per_video = 2

    mmm = MMModel(embed_dim = cfg.embed_dim, context_length = cfg.content_length, vocab_size = cfg.vocab_size, vaf_num_blocks = cfg.num_blocks, vaf_num_heads = cfg.num_heads, num_tokens_per_video = num_tokens_per_video,vocab_file = cfg.vocab_file).to(device)
    if cfg.checkpoint_path != None:
        logger.info(f"loading checkpoint from {cfg.checkpoint_path}")
        mmm.load_state_dict(torch.load(cfg.checkpoint_path))

    score_cri = MSELoss()
    lan_cri = CrossEntropyLoss() 
    locate_cri = BCELoss()

    save_path = './result_thre/' + cfg.root_dir.split('/')[-2]
    os.makedirs(save_path, exist_ok=True)

    print(f"training dataset: {save_path}")

    logger.add(sink=f"{save_path}.log", level="INFO", format="{time:HH:mm:ss}  | {message}| {level}")

    logger.info(f"training dataset: {save_path}")

    dataset = MMDataset(root_dir = cfg.root_dir, split = 'train', context_length = cfg.content_length, vocab_file = cfg.vocab_file, num_tokens_per_video = num_tokens_per_video, prompt = cfg.prompt)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataset = MMDataset(root_dir = cfg.root_dir, split = 'val', context_length = cfg.content_length, vocab_file = cfg.vocab_file, num_tokens_per_video = num_tokens_per_video, prompt = cfg.prompt)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)


    opt = torch.optim.Adam(mmm.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

    best_tes_loss = float('inf')
    best_pcs_loss = float('inf')
    best_tes_spear = float('-inf')
    best_pcs_spear = float('-inf')

    for i in range(cfg.epoch):
        losses = []
        tes_losses = []; pcs_losses = []; locate_losses = []
        tes_true = []; pcs_true = []
        tes_pred = []; pcs_pred = []

        print(f"-----------------------------------------epoch {i+1} training-----------------------------------------")
        for j, data in enumerate(dataloader):
            # attention mask related to the head_of_attention
            labels, video_feature, audio_feature = data
            video_feature = video_feature.to(device,dtype = torch.float32); audio_feature = audio_feature.to(device,dtype = torch.float32)
            
            # float, float, input_ids(batch_size, max_seq_length)
            tes = labels['tes']; pcs = labels['pcs']; caption = labels['input_ids']; prompt_ids = labels['prompt_ids']; locate_label = torch.tensor(labels['locate_label']); sample_name = labels['sample_name']
            pad_ids = labels['pad_ids'].to(device)
            tes_true.append(tes.numpy()); pcs_true.append(pcs.numpy())
            tes = tes.to(device, dtype = torch.float32); pcs = pcs.to(device, dtype = torch.float32); caption = caption.to(device); prompt_ids = prompt_ids.to(device);locate_label = locate_label.to(device, dtype = torch.float32)

            mmm.train()
            if "finefs" not in cfg.root_dir:
                mmm.agl.requires_grad = False
            p_tes, p_pcs, hidden_state, located, shifted_language = mmm(video_feature, audio_feature, caption, prompt_ids, pad_ids)
            tes_pred.append(p_tes.detach().cpu().numpy()); pcs_pred.append(p_pcs.detach().cpu().numpy())

            shifted_language = shifted_language.reshape(-1, cfg.vocab_size) # [len,embed_dim]
            shift_length = shifted_language.shape[0]
            shifted_caption = torch.cat([prompt_ids, caption], dim=1)
            # pad to shift_length
            if shift_length > shifted_caption.shape[1]:
                shifted_caption = torch.cat([shifted_caption, pad_ids.repeat(1, shift_length - shifted_caption.shape[1])], dim=1)
            shifted_caption = shifted_caption.reshape(-1)

            tes_loss = score_cri(p_tes, tes)
            pcs_loss = score_cri(p_pcs, pcs)
            tes_losses.append(tes_loss.item()); pcs_losses.append(pcs_loss.item())

            if locate_label.shape[0] > located.shape[0]:
                locate_label = locate_label[:located.shape[0]]
            elif locate_label.shape[0] < located.shape[0]:
                located = located[:locate_label.shape[0]]

            locate_loss = locate_cri(located, locate_label)
            locate_losses.append(locate_loss.item())
            lan_loss = lan_cri(shifted_language, shifted_caption)

            if "finefs" in cfg.root_dir:
                loss = tes_loss + pcs_loss + locate_loss + lan_loss
            else:
                loss = tes_loss + pcs_loss + lan_loss

            loss.backward()

            opt.step()
            opt.zero_grad()

            scheduler.step()
            losses.append(loss.item())

        # log
        tes_spear = spearmanr(tes_pred, tes_true).correlation
        pcs_spear = spearmanr(pcs_pred, pcs_true).correlation
        avg_tes_losses = sum(tes_losses) / len(tes_losses)
        avg_pcs_losses = sum(pcs_losses) / len(pcs_losses)
        if "finefs" in cfg.root_dir:
            avg_locate_losses = sum(locate_losses) / len(locate_losses)
        else:
            avg_locate_losses = 0
        
        print(f"epoch {i + 1}: ALL loss:{sum(losses)/len(losses)}, TES loss:{avg_tes_losses}, PCS loss:{avg_pcs_losses}, locate loss:{avg_locate_losses}, lan loss:{lan_loss}")
        print(f"epoch {i + 1}: TES spearman:{tes_spear}, PCS spearman:{pcs_spear}")


        # val
        val_tes_spear, val_pcs_spear, val_avg_tes_losses, val_avg_pcs_losses = val(val_loader, mmm, score_cri, locate_cri, i, cfg.root_dir)

        if val_avg_tes_losses < best_tes_loss:
            best_tes_loss = val_avg_tes_losses
        if val_avg_pcs_losses < best_pcs_loss:
            best_pcs_loss = val_avg_pcs_losses

        if val_tes_spear > best_tes_spear:
            best_tes_spear = val_tes_spear
        if val_pcs_spear > best_pcs_spear:
            best_pcs_spear = val_pcs_spear

        logger.info(f"epoch {i + 1}: best tes loss: {best_tes_loss}, best pcs loss: {best_pcs_loss}")
        logger.info(f"epoch {i + 1}: best tes spearman: {best_tes_spear}, best pcs spearman: {best_pcs_spear}")

        #save checkpoint
        torch.save(mmm.state_dict(), f"{save_path}/tes_spear_{val_tes_spear:.4f}_pcs_spear_{val_pcs_spear:.4f}_tes_loss_{val_avg_tes_losses:.4f}_pcs_loss_{val_avg_pcs_losses:.4f}.pth")
        
    logger.info(f"best tes loss: {best_tes_loss}, best pcs loss: {best_pcs_loss}")
    logger.info(f"best tes spearman: {best_tes_spear}, best pcs spearman: {best_pcs_spear}")

if __name__ == '__main__':
    train()