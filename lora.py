import torch
import torch.nn as nn
import torch.nn.functional as F

# Import math for initializing weights
import math

# Assume official mobile_sam package is installed
from mobile_sam.modeling.sam import Sam # Import base Sam model

class LoRALinear(nn.Module):
    """
    Linear layer with Low-Rank Adaptation (LoRA).
    
    LoRA is a parameter-efficient fine-tuning method that injects trainable low-rank
    matrices into frozen pre-trained layers. Instead of fine-tuning all parameters,
    LoRA freezes the pre-trained model weights and introduces pairs of rank-decomposition
    matrices for specific layers, significantly reducing the number of trainable parameters.
    
    This implementation applies LoRA to a linear layer, adding the low-rank update:
    h = W0x + ∆Wx = W0x + BAx, where:
    - W0 is the frozen pre-trained weights
    - B ∈ ℝ^(d_out×r) and A ∈ ℝ^(r×d_in) are the trainable low-rank matrices
    - r << min(d_in, d_out) is the rank parameter controlling parameter efficiency
    """
    def __init__(
        self, 
        original_layer: nn.Linear, 
        r: int = 4, 
        lora_alpha: int = 4, 
        lora_dropout: float = 0.0
    ):
        """
        Initialize LoRA-adapted linear layer.
        
        Args:
            original_layer: Pre-trained linear layer to adapt
            r: Rank of the low-rank adaptation matrices
            lora_alpha: Scaling factor for the LoRA contribution
            lora_dropout: Dropout probability applied before LoRA adaptation
        """
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
        
        # LoRA dropout regularization
        if lora_dropout > 0.0 and r > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
            
        # Initialize LoRA low-rank matrices (only if r > 0)
        if r > 0:
            # Matrix A projects from input dimension to rank r
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            # Matrix B projects from rank r to output dimension
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            
            # Weight initialization - important for good optimization
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B) # Zero-initialize B for stability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining original linear transformation with LoRA adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with LoRA adaptation applied
        """
        # Apply original linear transformation (frozen weights)
        result = self.linear(x)
        
        # Apply LoRA adaptation path if rank > 0
        if self.r > 0:
            # Apply dropout to input for LoRA path
            lora_x = self.lora_dropout(x)
            
            # Compute low-rank adaptation: BA·x
            # First project to lower dimension with A, then to output dimension with B
            lora_update = F.linear(F.linear(lora_x, self.lora_A), self.lora_B) * self.scaling
            
            # Add LoRA adaptation to original output
            result = result + lora_update
            
        return result


def apply_lora_to_attn(module: nn.Module, r: int, alpha: int, dropout: float):
    """
    Recursively applies LoRA adaptation to attention mechanisms in a neural network.
    
    This function traverses the module hierarchy, identifying linear layers within
    attention blocks and replacing them with LoRA-adapted equivalents.
    
    Args:
        module: PyTorch module to adapt
        r: LoRA rank parameter
        alpha: LoRA scaling factor
        dropout: Dropout probability for LoRA layers
    """
    for name, child_module in module.named_children():
        if isinstance(child_module, nn.Linear):
            # Apply LoRA to linear layers in attention mechanisms
            # This heuristic approach targets all linear layers that might be part of
            # query, key, value projections in attention blocks
            setattr(module, name, LoRALinear(child_module, r, alpha, dropout))
        elif isinstance(child_module, nn.MultiheadAttention):
             # Handle MultiheadAttention modules
             # These have internal linear projections that need special handling
             apply_lora_to_attn(child_module, r, alpha, dropout) # Recurse into module
        else:
            # Recursively process other nested modules
            apply_lora_to_attn(child_module, r, alpha, dropout)

class MobileSAM_LoRA_Adapted(nn.Module):
    """
    Parameter-efficient adaptation of MobileSAM using LoRA.
    
    This wrapper applies LoRA to attention layers within the official MobileSAM model
    and optionally adds a lightweight segmentation head for direct mask prediction.
    The approach preserves most of the pre-trained model's parameters while enabling
    fine-tuning with significantly reduced parameter count and memory requirements.
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
        """
        Initialize the LoRA-adapted MobileSAM model.
        
        Args:
            model: Base MobileSAM model instance
            r: LoRA rank parameter controlling adaptation capacity
            lora_alpha: Scaling factor for LoRA contributions
            lora_dropout: Dropout probability in LoRA layers
            train_encoder: Whether to apply LoRA to image encoder
            train_decoder: Whether to apply LoRA to mask decoder
            use_temp_head: Whether to add a temporary segmentation head
        """
        super().__init__()
        
        self.model = model
        self.use_temp_head = use_temp_head

        # Freeze all parameters of the original model
        for param in self.model.parameters():
            param.requires_grad = False

        # Apply LoRA to image encoder (transformer-based)
        if train_encoder:
            print("Applying LoRA to Image Encoder...")
            apply_lora_to_attn(self.model.image_encoder, r, lora_alpha, lora_dropout)
        
        # Apply LoRA to mask decoder if requested (more complex)
        if train_decoder:
            print("Applying LoRA to Mask Decoder...")
            apply_lora_to_attn(self.model.mask_decoder, r, lora_alpha, lora_dropout)
        
        # Add a task-specific segmentation head
        self.temp_decoder_head = None
        if use_temp_head:
             print("Adding temporary decoder head...")
             # The input channels (256) must match the output of the image encoder's feature dimensions
             self.temp_decoder_head = nn.Conv2d(256, 1, kernel_size=1)
             # Make the segmentation head trainable
             for param in self.temp_decoder_head.parameters():
                  param.requires_grad = True

    def forward(self, image):
        """
        Forward pass through the LoRA-adapted model.
        
        Args:
            image: Input image tensor (B, C, H, W)
            
        Returns:
            Tuple of (outputs, iou_pred) where:
            - outputs: Predicted segmentation mask
            - iou_pred: IoU prediction (None if using temp head)
        """
        # Preprocess using SAM's standardized method
        input_images_preprocessed = self.model.preprocess(image)
        
        # Forward pass through the image encoder (with LoRA if applied)
        image_embeddings = self.model.image_encoder(input_images_preprocessed)
        
        # Use the task-specific segmentation head if enabled
        if self.use_temp_head and self.temp_decoder_head is not None:
            # Generate mask prediction directly from embeddings
            outputs = self.temp_decoder_head(image_embeddings)
            
            # Upsample to match original image dimensions
            outputs = F.interpolate(outputs, size=image.shape[-2:], mode='bilinear', align_corners=False)
            iou_pred = None # No IoU prediction with temp head
        else:
            # Fallback for case when not using the temp head
            # This would typically require prompts for the original decoder
            print("Warning: LoRA model configured without temp_decoder_head. Returning zeros.")
            outputs = torch.zeros((image.shape[0], 1, image.shape[2], image.shape[3]), device=image.device)
            iou_pred = None
            
            # Example of how original SAM decoder would be used (commented out):
            # sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            #     points=None,  # Would need prompt points
            #     boxes=None,   # Would need bounding boxes
            #     masks=None,
            # )
            # outputs, iou_pred = self.model.mask_decoder(
            #     image_embeddings=image_embeddings,
            #     image_pe=self.model.prompt_encoder.get_dense_pe(),
            #     sparse_prompt_embeddings=sparse_embeddings,
            #     dense_prompt_embeddings=dense_embeddings,
            #     multimask_output=False,
            # )
            # outputs = F.interpolate(outputs, size=image.shape[-2:], mode='bilinear', align_corners=False)
        
        return outputs, iou_pred
    
    def get_trainable_parameters(self):
        """
        Identify and return all trainable parameters in the model.
        
        Returns:
            List of trainable parameters (LoRA matrices + segmentation head)
        """
        params = []
        total_params = 0
        trainable_params = 0
        
        # Count parameters and identify trainable ones
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                params.append(param)
                trainable_params += param.numel()
        
        # Report parameter efficiency stats
        print(f"Total model parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params} ({100 * trainable_params / total_params:.4f}%)")
        
        return params 