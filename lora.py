import torch
import torch.nn as nn
import torch.nn.functional as F

# Import math for initializing weights
import math

# Assume official mobile_sam package is installed
from mobile_sam.modeling.sam import Sam # Import base Sam model

class LoRALinear(nn.Module):
    """
    Linear layer with LoRA (Low-Rank Adaptation)
    """
    def __init__(
        self, 
        original_layer: nn.Linear, 
        r: int = 4, 
        lora_alpha: int = 4, 
        lora_dropout: float = 0.0,
        merge_weights: bool = False, # Merging not fully supported here easily
    ):
        super().__init__()
        
        # Original linear layer (keep frozen)
        self.linear = original_layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
             self.linear.bias.requires_grad = False # Freeze bias too

        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # LoRA parameters
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r if r > 0 else 0.0
        
        # LoRA dropout
        if lora_dropout > 0.0 and r > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
            
        # LoRA low-rank matrices (only if r > 0)
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            # Weight initialization
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        
        # self.merged = False # Merging logic removed for simplicity
        # self.merge_weights = merge_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply original linear layer (frozen)
        result = self.linear(x)
        
        # Apply LoRA path if r > 0
        if self.r > 0:
            lora_x = self.lora_dropout(x)
            # Use F.linear for matrix multiplication with parameters
            lora_update = F.linear(F.linear(lora_x, self.lora_A), self.lora_B) * self.scaling
            result = result + lora_update
            
        return result


def apply_lora_to_attn(module: nn.Module, r: int, alpha: int, dropout: float):
    """
    Recursively applies LoRA to QKV projections in attention modules.
    Looks for nn.Linear layers typically used for Q, K, V projections.
    NOTE: This is heuristic and might need adjustment based on exact layer names 
          in the specific attention implementation used by official MobileSAM.
    """
    for name, child_module in module.named_children():
        if isinstance(child_module, nn.Linear):
            # Heuristic: Replace Linear layers likely involved in QKV projections
            # This might target more layers than just QKV, depending on the structure.
            # A more robust approach would identify specific QKV layers by name pattern if possible.
            # print(f"Applying LoRA to: {name} in {module.__class__.__name__}")
            setattr(module, name, LoRALinear(child_module, r, alpha, dropout))
        elif isinstance(child_module, nn.MultiheadAttention):
             # If we find MHA, try to replace its internal linear projections 
             # (Requires knowledge of MHA's internal structure: in_proj_weight)
             # This part is complex and might break easily. We'll stick to replacing 
             # nn.Linear layers found within the attention block for now.
             # print(f"Found MHA: {name}, attempting recursive application...")
             apply_lora_to_attn(child_module, r, alpha, dropout) # Recurse
        else:
            # Recursively apply to other submodules
            apply_lora_to_attn(child_module, r, alpha, dropout)

class MobileSAM_LoRA_Adapted(nn.Module):
    """
    MobileSAM wrapper that applies LoRA to the official model structure 
    and adds a simple segmentation head.
    """
    def __init__(
        self,
        model: Sam, # Takes the official SAM model instance
        r: int = 4,
        lora_alpha: int = 4,
        lora_dropout: float = 0.0,
        train_encoder: bool = True,
        train_decoder: bool = False, # Defaulting to not training the original decoder
        use_temp_head: bool = True, # Flag to use the added simple head
    ):
        super().__init__()
        
        self.model = model
        self.use_temp_head = use_temp_head

        # Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Apply LoRA to image encoder if requested
        if train_encoder:
            print("Applying LoRA to Image Encoder...")
            apply_lora_to_attn(self.model.image_encoder, r, lora_alpha, lora_dropout)
        
        # Apply LoRA to mask decoder if requested (more complex)
        if train_decoder:
            print("Applying LoRA to Mask Decoder...")
            # This requires careful targeting of attention layers within the decoder
            apply_lora_to_attn(self.model.mask_decoder, r, lora_alpha, lora_dropout)
        
        # Add a temporary segmentation head if flag is set
        self.temp_decoder_head = None
        if use_temp_head:
             print("Adding temporary decoder head...")
             # The input channel size (256) depends on the output of the image encoder's neck
             self.temp_decoder_head = nn.Conv2d(256, 1, kernel_size=1)
             # Ensure this head's parameters are trainable
             for param in self.temp_decoder_head.parameters():
                  param.requires_grad = True

    def forward(self, image):
        # Preprocess using the model's method
        input_images_preprocessed = self.model.preprocess(image)
        
        # Pass through image encoder
        # No need for torch.no_grad context here, grad is controlled by requires_grad
        image_embeddings = self.model.image_encoder(input_images_preprocessed)
        
        # Use the temporary head for prediction if enabled
        if self.use_temp_head and self.temp_decoder_head is not None:
            outputs = self.temp_decoder_head(image_embeddings)
            # Upsample to match input image size (before preprocessing)
            # Note: SAM's preprocess might pad, check target size
            outputs = F.interpolate(outputs, size=image.shape[-2:], mode='bilinear', align_corners=False)
            iou_pred = None # No IoU prediction from temp head
        else:
            # If not using temp head, we'd need to use the original decoder
            # This requires handling prompts, which we are avoiding for automated seg.
            # Returning zeros as a placeholder if temp head isn't used.
            print("Warning: LoRA model configured without temp_decoder_head. Returning zeros.")
            outputs = torch.zeros((image.shape[0], 1, image.shape[2], image.shape[3]), device=image.device)
            iou_pred = None
            
            # --- Placeholder for using original decoder (requires prompts) ---
            # sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            #     points=None, # Provide dummy prompts here
            #     boxes=None,  # e.g., whole image box
            #     masks=None,
            # )
            # outputs, iou_pred = self.model.mask_decoder(
            #     image_embeddings=image_embeddings,
            #     image_pe=self.model.prompt_encoder.get_dense_pe(),
            #     sparse_prompt_embeddings=sparse_embeddings,
            #     dense_prompt_embeddings=dense_embeddings,
            #     multimask_output=False, # Usually False for single object segmentation
            # )
            # outputs = F.interpolate(outputs, size=image.shape[-2:], mode='bilinear', align_corners=False)
            # --------------------------------------------------------------
        
        return outputs, iou_pred # Return None for iou_pred if using temp head
    
    def get_trainable_parameters(self):
        """
        Get all parameters that require gradients (LoRA params + temp head).
        """
        params = []
        total_params = 0
        trainable_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                params.append(param)
                trainable_params += param.numel()
                # print(f"  Trainable: {name} ({param.numel()})")
        
        print(f"Total model parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params} ({100 * trainable_params / total_params:.4f}%)")
        
        return params 