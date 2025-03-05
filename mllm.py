from functools import partial
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
# random delete branch(droppath)
from timm.models.layers import DropPath, trunc_normal_
import einops
from decoder import *
from agl import AGL
from tokenizer import SimpleTokenizer

def build_causal_attention_mask(context_length, top_left_square = None):
    mask = torch.ones(context_length, context_length, requires_grad=False)
    # mask.triu_(0)
    if top_left_square:
        mask[:top_left_square, :top_left_square] = 1

    return mask


class MMModel(nn.Module):
    def __init__(self, embed_dim: int = 1024, context_length: int = 100, vocab_size: int = 49413, num_blocks: int = 8, num_heads: int = 8, vaf_num_blocks: int = 8, vaf_num_heads: int = 8, num_tokens_per_video: int = 2, vocab_file: str = "/data1/1/code/helping/ImageBind/imagebind/bpe/bpe_simple_vocab_16e6.txt.gz"):
        super().__init__()

        self.st = SimpleTokenizer(vocab_file, context_length)
        self.tes = self.st.tes_token
        self.pcs = self.st.pcs_token

        # cross attention segmentation
        self.agl = AGL(embed_dim=embed_dim, num_blocks=16, num_heads=8, max_temporal_length=265)

        # vocab_size is 49413 (0 - 49412), special padding_idx????? don't influence the embedding, why not update????
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.token_embedding.weight.requires_grad = True
        self.pos_embed = nn.Parameter(
            torch.empty(1, context_length, embed_dim)
        )
        self.context_length = context_length

        self.vaf = SimpleTransformer(
            embed_dim=embed_dim,
            num_blocks=vaf_num_blocks,
            ffn_dropout_rate=0.0,
            drop_path_rate=0.1,
            attn_target=partial(
                Cross_MultiheadAttention,
                batch_first=True,
                embed_dim=embed_dim,
                num_heads=vaf_num_heads,
                bias=True,
                add_bias_kv=True,
            ),
            pre_transformer_layer=nn.Sequential(
                nn.LayerNorm(embed_dim, eps=1e-6),
                nn.Identity()
            ),
        )

        self.mllm = SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=0.1,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=True,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6),
                    nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b d l"),
            )


        self.tes_score_head = nn.Sequential(
            nn.Linear(embed_dim ,int(embed_dim/2), bias=True),
            nn.ReLU(),
            nn.Linear(int(embed_dim/2), 1, bias=True),
        )

        self.pcs_score_head = nn.Sequential(
            nn.Linear(embed_dim ,int(embed_dim/2), bias=True),
            nn.ReLU(),
            nn.Linear(int(embed_dim/2), 1, bias=True),
        )

        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.init_parameters()
        self.num_tokens_per_video = num_tokens_per_video


    @torch.no_grad()
    def init_parameters(self, init_param_style="openclip"):
        # OpenCLIP style initialization
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.01)

    def _merge_audio_and_visual_features(self, input_embed, visual_features, audio_features):
        # [bs, seq_len, embed_dim], [bs, 7 or 12, embed_dim], [bs, 7 or 12, embed_dim]
        for i in range(visual_features.shape[1]):
            input_embed[:, 2 * i, :] = visual_features[:, i, :]
            input_embed[:, 2 * i + 1, :] = audio_features[:, i, :]

        return input_embed

    def forward(self, video_featuers: torch.Tensor = None, audio_features: torch.Tensor = None, input_ids: torch.Tensor = None, prompt_ids: torch.Tensor = None, pad_ids: torch.Tensor = None):
        # [bs, nos, embed_dim], [bs, nos, embed_dim], [bs, seq_len], [bs, nos], [bs, nos]
        stacked_video_segments, stacked_audio_segments, located, segments, segment_types = self.agl(video_featuers, audio_features)
        bs = stacked_video_segments.shape[0]; nos = stacked_video_segments.shape[1]; embed_dim = stacked_video_segments.shape[2]

        # ff [bs, nos+2, embed_dim]
        fused_features = torch.zeros([bs, self.context_length, embed_dim])
        for i in range(nos):
            fused_features[:, i, :] = self.vaf([stacked_video_segments[:, i, :].clone(), stacked_audio_segments[:, i, :].clone()])

        # add tes,pcs token
        tes = self.token_embedding(torch.tensor(self.tes,dtype=torch.int32,device=self.token_embedding.weight.device))
        pcs = self.token_embedding(torch.tensor(self.pcs,dtype=torch.int32,device=self.token_embedding.weight.device))
        text = self.token_embedding(torch.cat([prompt_ids, input_ids], dim=1)).squeeze(0)
        pad = self.token_embedding(pad_ids.clone().detach().to(dtype=torch.int32,device=self.token_embedding.weight.device))

        # multi-model feature series construction
        t_len = text.shape[0]
        fused_features[:, nos, :] = tes
        fused_features[:, nos+1, :] = pcs
        fused_features[:, nos+2:nos+2+t_len, :] = text
        fused_features[:, nos+2+t_len:, :] = pad
        fused_features = fused_features.to(self.token_embedding.weight.device)

        fused_features = fused_features + self.pos_embed
        # trunk [bs, embed_dim, seq_len]
        attn_mask = build_causal_attention_mask(self.context_length).to(fused_features.device)
        # maksed transformer decoder
        hidden_state = self.mllm(fused_features, attn_mask)

        # use tes,pcs token to regression
        tes_state = hidden_state[:, :, nos]
        pcs_state = hidden_state[:, :, nos+1]
        text_state = hidden_state[:, :, nos+2:]
        tes = self.tes_score_head(tes_state).reshape(-1)
        pcs = self.pcs_score_head(pcs_state).reshape(-1)
        shift_logits = self.lm_head(text_state.transpose(1,2))[:,:-1,:].contiguous()

        return tes, pcs, hidden_state, located.reshape(-1), shift_logits


if __name__ == "__main__":
    pass

    

