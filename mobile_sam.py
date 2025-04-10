import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Dropout
import math
from typing import Optional, Tuple, Type

# MobileSAM Tiny ViT Encoder
class TinyViTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, bias=qkv_bias, batch_first=True
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        x = x + self.drop_path(self.attn(x_norm, x_norm, x_norm)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TinyViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dims: int = 768,
        num_heads: int = 12,
        mlp_ratios: float = 4.0,
        depths: int = 12,
        drop_path_rate: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.patch_size = patch_size
        
        # Split image into patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dims,
        )
        
        # Add position embedding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))
        
        # Drop path rate increases with depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        
        # Build transformer blocks
        self.blocks = nn.ModuleList(
            [
                TinyViTBlock(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depths)
            ]
        )
        
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dims,
                256,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.norm = norm_layer(embed_dims)
        
        # Initialize position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # Convert to patches and add position embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        # Reshape to spatial features
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Apply neck
        x = self.neck(x)
        
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


# MobileSAM Mask Decoder
class MaskDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_multimask_outputs: int = 3,
        transformer_dim: int = 256,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(
            num_multimask_outputs, transformer_dim
        )

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            nn.LayerNorm([transformer_dim // 4, 2 * 64, 2 * 64]),
            nn.GELU(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            nn.LayerNorm([transformer_dim // 8, 4 * 64, 4 * 64]),
            nn.GELU(),
            nn.ConvTranspose2d(
                transformer_dim // 8, transformer_dim // 16, kernel_size=2, stride=2
            ),
            nn.LayerNorm([transformer_dim // 16, 8 * 64, 8 * 64]),
            nn.GELU(),
            nn.ConvTranspose2d(
                transformer_dim // 16, 1, kernel_size=1, stride=1
            ),
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, 1, iou_head_depth
        )

        self.transformer = nn.Transformer(
            d_model=transformer_dim,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=2048,
            dropout=0.0,
            batch_first=True,
        )

    def forward(self, image_embeddings):
        batch_size = image_embeddings.shape[0]
        
        # Prepare tokens
        src = image_embeddings
        iou_token = self.iou_token.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        mask_tokens = self.mask_tokens.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Prepare for transformer
        tgt = torch.cat([iou_token, mask_tokens], dim=1)
        
        # Run transformer
        hs = self.transformer(src, tgt)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:, :]
        
        # Upscale tokens to masks
        masks = []
        for i in range(self.num_multimask_outputs):
            mask_token = mask_tokens_out[:, i, :].reshape(batch_size, self.transformer_dim, 1, 1)
            mask = self.output_upscaling(mask_token)
            masks.append(mask)
        
        # Stack masks and predict IoU
        masks = torch.cat(masks, dim=1)
        iou_pred = self.iou_prediction_head(iou_token_out)
        
        return masks, iou_pred


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


# MobileSAM Model
class MobileSAM(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        decoder_num_masks: int = 3,
    ):
        super().__init__()
        
        # Image encoder
        self.image_encoder = TinyViT(
            img_size=img_size,
            embed_dims=embed_dim,
            depths=encoder_depth,
            num_heads=encoder_num_heads,
        )
        
        # Mask decoder
        self.mask_decoder = MaskDecoder(
            transformer_dim=256,  # Adjusted to match neck output
            num_multimask_outputs=decoder_num_masks,
        )
        
    def forward(self, image):
        image_embeddings = self.image_encoder(image)
        masks, iou_pred = self.mask_decoder(image_embeddings)
        
        return masks, iou_pred
    
    def get_image_embeddings(self, image):
        return self.image_encoder(image)

# Function to load MobileSAM pre-trained weights
def setup_model(checkpoint_path=None):
    model = MobileSAM()
    
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
    
    return model 