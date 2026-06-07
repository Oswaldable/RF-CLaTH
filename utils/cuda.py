import torch


def configure_cuda_attention(device: torch.device, logger=None):
    """Disable flash SDP on pre-Ampere GPUs.

    Some PyTorch builds may select FlashAttention for Transformer layers even
    on V100-class GPUs, where flash SDP is unsupported. Keeping math and
    memory-efficient SDP enabled preserves correctness on those devices.
    """

    if device.type != "cuda" or not torch.cuda.is_available():
        return
    props = torch.cuda.get_device_properties(device)
    if props.major >= 8:
        return
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is None:
        return
    if hasattr(cuda_backend, "enable_flash_sdp"):
        cuda_backend.enable_flash_sdp(False)
    if hasattr(cuda_backend, "enable_mem_efficient_sdp"):
        cuda_backend.enable_mem_efficient_sdp(True)
    if hasattr(cuda_backend, "enable_math_sdp"):
        cuda_backend.enable_math_sdp(True)
    if logger is not None:
        logger.info(
            "disabled_flash_sdp_for_gpu name=%s capability=%d.%d",
            props.name,
            props.major,
            props.minor,
        )
