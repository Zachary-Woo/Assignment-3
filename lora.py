import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

# Import math for initializing weights
import math

class LoRALinear(nn.Module):
    """
    Linear layer with LoRA (Low-Rank Adaptation)
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 4, 
        lora_alpha: int = 4, 
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__()
        
        # Original linear layer
        self.linear = nn.Linear(in_features, out_features, bias=True)
        
        # LoRA parameters
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        
        # LoRA dropout
        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
            
        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Weight initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.merged = False
        self.merge_weights = merge_weights
        
        if merge_weights:
            self.merge()
    
    def merge(self):
        """
        Merge LoRA weights with the original linear weights
        """
        if self.merged:
            return
            
        with torch.no_grad():
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            
        self.merged = True
    
    def unmerge(self):
        """
        Unmerge LoRA weights from the original linear weights
        """
        if not self.merged:
            return
            
        with torch.no_grad():
            self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            
        self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.linear(x)
        else:
            # Apply original linear layer
            result = self.linear(x)
            
            # Apply LoRA path
            lora_x = self.lora_dropout(x)
            lora_result = (lora_x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
            
            return result + lora_result


class LoRAMultiheadAttention(nn.Module):
    """
    Apply LoRA to MultiheadAttention module's query and value projections
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        r: int = 4,
        lora_alpha: int = 4,
        lora_dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        
        # Store embed_dim for use in forward method
        self.embed_dim = embed_dim
        
        # Original MultiheadAttention with unchanged parameters
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            batch_first=batch_first,
        )
        
        # Create LoRA Q and V projections
        # We apply LoRA to the query and value projections following the recommendations
        # in the original LoRA paper (https://arxiv.org/pdf/2106.09685.pdf)
        head_dim = embed_dim // num_heads
        
        # LoRA for query projection
        self.q_proj_lora = LoRALinear(
            in_features=embed_dim,
            out_features=embed_dim,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        
        # LoRA for value projection
        self.v_proj_lora = LoRALinear(
            in_features=embed_dim,
            out_features=embed_dim,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        
        # Store original projections
        self.original_q_proj = self.mha.in_proj_weight[:embed_dim]
        self.original_v_proj = self.mha.in_proj_weight[embed_dim*2:]
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with LoRA applied to query and value projections
        """
        # Since nn.MultiheadAttention doesn't expose individual Q, K, V projections,
        # we need to compute them separately and then modify the internal state
        
        # Apply LoRA to query projection
        q = self.q_proj_lora(query)
        
        # Key is unchanged - use the built-in projection
        k = F.linear(key, self.mha.in_proj_weight[self.embed_dim:self.embed_dim*2], 
                    self.mha.in_proj_bias[self.embed_dim:self.embed_dim*2] if self.mha.in_proj_bias is not None else None)
        
        # Apply LoRA to value projection
        v = self.v_proj_lora(value)
        
        # Now we can use the multi-head attention with our pre-projected q, k, v
        return self.mha(q, k, v, key_padding_mask, need_weights, attn_mask)


class LoRATransformer(nn.Module):
    """
    Transformer with LoRA adapters for attention blocks.
    """
    def __init__(
        self,
        model: nn.Transformer,
        r: int = 4,
        lora_alpha: int = 4,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        
        self.model = model
        
        # Apply LoRA to each attention layer in encoder
        for i, layer in enumerate(model.encoder.layers):
            # Replace self-attention with LoRA version
            layer.self_attn = LoRAMultiheadAttention(
                embed_dim=model.d_model,
                num_heads=model.nhead,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=True,
                batch_first=True,
            )
        
        # Apply LoRA to each attention layer in decoder
        for i, layer in enumerate(model.decoder.layers):
            # Replace self-attention with LoRA version
            layer.self_attn = LoRAMultiheadAttention(
                embed_dim=model.d_model,
                num_heads=model.nhead,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=True,
                batch_first=True,
            )
            
            # Replace cross-attention with LoRA version
            layer.multihead_attn = LoRAMultiheadAttention(
                embed_dim=model.d_model,
                num_heads=model.nhead,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=True,
                batch_first=True,
            )
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        return self.model(src, tgt, src_mask, tgt_mask, 
                           memory_mask, src_key_padding_mask,
                           tgt_key_padding_mask, memory_key_padding_mask)


class MobileSAM_LoRA(nn.Module):
    """
    MobileSAM with LoRA adapters applied to both image encoder and mask decoder.
    """
    def __init__(
        self,
        model,
        r: int = 4,
        lora_alpha: int = 4,
        lora_dropout: float = 0.0,
        train_encoder: bool = True,
        train_decoder: bool = True,
    ):
        super().__init__()
        
        self.model = model
        self.train_encoder = train_encoder
        self.train_decoder = train_decoder
        
        # Apply LoRA to image encoder if needed
        if train_encoder:
            for i, block in enumerate(model.image_encoder.blocks):
                # Replace attention in each transformer block with LoRA attention
                block.attn = LoRAMultiheadAttention(
                    embed_dim=block.norm1.normalized_shape[0],
                    num_heads=block.attn.num_heads,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=True,
                    batch_first=True,
                )
        
        # Apply LoRA to mask decoder if needed
        if train_decoder:
            # Replace transformer in mask decoder with LoRA transformer
            model.mask_decoder.transformer = LoRATransformer(
                model=model.mask_decoder.transformer,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
    
    def forward(self, image):
        # Freeze encoder if not training
        if not self.train_encoder:
            with torch.no_grad():
                image_embeddings = self.model.image_encoder(image)
        else:
            image_embeddings = self.model.image_encoder(image)
        
        # Freeze decoder if not training
        if not self.train_decoder:
            with torch.no_grad():
                masks, iou_pred = self.model.mask_decoder(image_embeddings)
        else:
            masks, iou_pred = self.model.mask_decoder(image_embeddings)
        
        return masks, iou_pred
    
    def get_trainable_parameters(self):
        """
        Get all LoRA trainable parameters
        """
        params = []
        
        for name, param in self.named_parameters():
            if 'lora_' in name or 'bias' in name:
                params.append(param)
        
        return params 