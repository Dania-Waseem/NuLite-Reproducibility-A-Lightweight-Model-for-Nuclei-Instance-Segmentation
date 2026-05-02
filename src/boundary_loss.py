import torch
import torch.nn.functional as F

def boundary_loss(pred, target):
    """
    Boundary-aware loss for nucleus segmentation.
    Encourages correct separation of touching nuclei.
    """

    # Make sure target is float
    target = target.float()

    # 3x3 kernel for morphological operations
    kernel = torch.ones(1, 1, 3, 3, device=pred.device)

    # Dilation (expands nuclei)
    dilated = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)

    # Erosion approximation (shrinks nuclei)
    eroded = -F.max_pool2d(-target, kernel_size=3, stride=1, padding=1)

    # Boundary = difference between outer and inner region
    boundary = dilated - eroded

    # Loss: compare predicted boundary vs real boundary
    loss = F.binary_cross_entropy_with_logits(pred, boundary)

    return loss