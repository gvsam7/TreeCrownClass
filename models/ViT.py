"""
Author: Georgios Voulgaris
Date: 15/01/2025
Description: Implementation of ViT transformer for image classification.
"""

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4):
        super(TransformerBlock, self).__init__()

        # ensure attention heads divide evenly
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.norm1 = nn.LayerNorm(hidden_size)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        x = self.multihead_attn(norm_x, norm_x, norm_x)[0] + x
        norm_x = self.norm2(x)
        x = self.mlp(norm_x) + x
        return x


class ViT(nn.Module):
    def __init__(self, img_size, in_channels, patch_size, hidden_size, num_layers, num_heads=8, num_classes=2):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        self.fc_in = nn.Linear(in_channels * patch_size * patch_size, hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_size, num_classes)
        self.out_vec = nn.Parameter(torch.zeros(1, 1, hidden_size))

        seq_length = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length + 1, hidden_size))
        # initialise embeddings and class token
        nn.init.normal_(self.pos_embedding, std=0.01)
        nn.init.zeros_(self.out_vec)

    def forward(self, image):
        bs = image.shape[0]
        patch_seq = self.extract_patches(image)  # (bs, seq, patch_dim)
        patch_emb = self.fc_in(patch_seq)  # (bs, seq, hidden)
        # add pos embeddings (include class token slot)
        pos = self.pos_embedding.expand(bs, -1, -1).to(patch_emb.device)
        embs = torch.cat((self.out_vec.expand(bs, 1, -1).to(patch_emb.device), patch_emb + pos[:, 1:, :]), dim=1)

        for block in self.blocks:
            embs = block(embs)

        return self.fc_out(embs[:, 0])

    def extract_patches(self, img):
        bs, c, h, w = img.size()
        assert h % self.patch_size == 0 and w % self.patch_size == 0, \
        "Image dimensions must be divisable by the patch size"

        patches = img.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(bs, -1, c * self.patch_size * self.patch_size)
        return patches


"""
# Test Validation
dummy_image = torch.randn(4, 3, 128, 128)  # Batch size=4, 3-channels, 128x128 image
vit = ViT(img_size=128, in_channels=3, patch_size=16, hidden_size=128, num_layers=4, num_heads=4, num_classes=16)
patches = vit.extract_patches(dummy_image)
print(patches.shape)  # Expected: (4, (128//16)**2, 3*16*16) = (4, 64, 768)

# full forward pass
with torch.no_grad():
    out = vit(dummy_image)
print(out.shape)  # expect: torch.Size([4, 16])"""

