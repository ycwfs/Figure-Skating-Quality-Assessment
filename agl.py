from decoder import MultiheadAttention, SimpleTransformer, Cross_MultiheadAttention
import torch
import torch.nn as nn
from functools import partial
import numpy as np

class AGL(nn.Module):
    def __init__(self, embed_dim: int = 1024, num_blocks: int = 8, num_heads: int = 8, max_temporal_length: int = 265,  threshold: int = 0.6):
        super().__init__()

        self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        # [b,l,128] -> [b,l,1024]
        self.audio_projector = nn.Sequential(
            nn.Linear(128, int(embed_dim/2)),
            nn.ReLU(),
            nn.Linear(int(embed_dim/2), embed_dim)
        )

        self.video_projector = nn.Sequential(
            nn.Linear(1024, int(embed_dim/2)),
            nn.ReLU(),
            nn.Linear(int(embed_dim/2), embed_dim)
        )

        self.ca = SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=0.1,
                attn_target=partial(
                    Cross_MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=True,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6),
                    nn.Identity(),
                    #EinOpsRearrange("b l d -> l b d"),
                ),
                #post_transformer_layer=EinOpsRearrange("l b d -> b d l"),
            )

        # 
        self.locate_head = nn.Sequential(
            nn.Linear(embed_dim,int(embed_dim/2)),
            nn.ReLU(),
            nn.Linear(int(embed_dim/2), 1),
            nn.Sigmoid()
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)




    def forward(self, video_features: torch.Tensor = None, audio_features: torch.Tensor = None):
        #assert video_features.shape[1] == audio_features.shape[1]
        #cut longer features to the same length
        gap = video_features.shape[1] - audio_features.shape[1]
        if np.abs(gap) > 10:
            pass
            #print(f"warning: VA features has {gap} frame gaps")
        if video_features.shape[1] > audio_features.shape[1]:
            video_features = video_features[:, :audio_features.shape[1], :]
        elif video_features.shape[1] < audio_features.shape[1]:
            audio_features = audio_features[:, :video_features.shape[1], :]

        audio_features = self.audio_projector(audio_features)
        video_features = self.video_projector(video_features)

        hidden_states = self.ca([audio_features,video_features])
        located = self.locate_head(hidden_states).squeeze(-1)
        #segments = self.get_segments(located.reshape(-1).detach().cpu().numpy())
        clips = (located > self.threshold).type(torch.int)
        # from the [0,0,0,1,1,1,0,0,0] to get three segments token [3,1024] after adaptive_pooling
        segments = []
        current_segment = []
        current_type = clips[0, 0].item()

        for i in range(clips.shape[1]):
            if clips[0, i] == current_type:
                current_segment.append(i)
            else:
                if current_segment:
                    segments.append((current_type, current_segment))
                current_segment = [i]
                current_type = clips[0, i].item()

        if current_segment:
            segments.append((current_type, current_segment))

        pooled_segments = []
        segment_types = []

        for segment_type, segment in segments:
            video_segment_features = video_features[:, segment, :]
            audio_segment_features = audio_features[:, segment, :]
            video_pooled_segment = self.adaptive_pool(video_segment_features.permute(0, 2, 1)).reshape(-1)
            audio_pooled_segment = self.adaptive_pool(audio_segment_features.permute(0, 2, 1)).reshape(-1)
            pooled_segments.append((video_pooled_segment, audio_pooled_segment))
            segment_types.append(segment_type)

        stacked_video_segments = torch.stack([seg[0] for seg in pooled_segments]).unsqueeze(0)
        stacked_audio_segments = torch.stack([seg[1] for seg in pooled_segments]).unsqueeze(0)

        return stacked_video_segments, stacked_audio_segments, located, segments, segment_types


if __name__ == "__main__":
    import torch.optim as optim
        # Initialize the model
    model = AGL(embed_dim=1024, num_blocks=8, num_heads=8, max_temporal_length=265)

    # Create dummy input data
    batch_size = 1
    seq_length = 16
    video_dim = 1024
    audio_dim = 128

    video_features = torch.randn(batch_size, seq_length, video_dim)
    audio_features = torch.randn(batch_size, seq_length, audio_dim)

    # Define the target label (binary for BCELoss)
    target = torch.ones(batch_size, seq_length)  # Example target, use appropriate labels

    # Define loss criterion
    criterion = nn.BCELoss()

    # Forward pass
    vs,aps,l,s,st = model(video_featuers=video_features, audio_features=audio_features)

    # threshold the output to get binary predictions
    res = (l > 0.5).float()

    # Compute loss
    loss = criterion(l, target)

    # Print the loss for verification
    print(f"Loss before backward: {loss.item()}")

    # Backward pass
    loss.backward()

    # Check gradients (optionally you can also perform an optimizer step to check updates)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()

    # Print some gradients or updated parameters for verification
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Grad for {name}: {param.grad.norm()}")  # Just an example to show gradient
