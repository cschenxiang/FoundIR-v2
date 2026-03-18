import contextlib
import copy
import gc
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from foundirv2.models.unet_2d_condition_vae_extension import UNet2DConditionModel
from diffusers.schedulers import SchedulerMixin
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_state_dict_to_peft,
    deprecate,
    is_peft_available,
    is_torch_npu_available,
    is_torchvision_available,
    is_transformers_available,
)


if is_transformers_available():
    import transformers

    if transformers.integrations.deepspeed.is_deepspeed_zero3_enabled():
        import deepspeed

if is_peft_available():
    from peft import set_peft_model_state_dict

if is_torchvision_available():
    from torchvision import transforms

if is_torch_npu_available():
    import torch_npu  # noqa: F401


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`): The seed to set.

    Returns:
        `None`
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_torch_npu_available():
        torch.npu.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    for the given timesteps using the provided noise scheduler.

    Args:
        noise_scheduler (`NoiseScheduler`):
            An object containing the noise schedule parameters, specifically `alphas_cumprod`, which is used to compute
            the SNR values.
        timesteps (`torch.Tensor`):
            A tensor of timesteps for which the SNR is computed.

    Returns:
        `torch.Tensor`: A tensor containing the computed SNR values for each timestep.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def resolve_interpolation_mode(interpolation_type: str):
    """
    Maps a string describing an interpolation function to the corresponding torchvision `InterpolationMode` enum. The
    full list of supported enums is documented at
    https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.functional.InterpolationMode.

    Args:
        interpolation_type (`str`):
            A string describing an interpolation method. Currently, `bilinear`, `bicubic`, `box`, `nearest`,
            `nearest_exact`, `hamming`, and `lanczos` are supported, corresponding to the supported interpolation modes
            in torchvision.

    Returns:
        `torchvision.transforms.InterpolationMode`: an `InterpolationMode` enum used by torchvision's `resize`
        transform.
    """
    if not is_torchvision_available():
        raise ImportError(
            "Please make sure to install `torchvision` to be able to use the `resolve_interpolation_mode()` function."
        )

    if interpolation_type == "bilinear":
        interpolation_mode = transforms.InterpolationMode.BILINEAR
    elif interpolation_type == "bicubic":
        interpolation_mode = transforms.InterpolationMode.BICUBIC
    elif interpolation_type == "box":
        interpolation_mode = transforms.InterpolationMode.BOX
    elif interpolation_type == "nearest":
        interpolation_mode = transforms.InterpolationMode.NEAREST
    elif interpolation_type == "nearest_exact":
        interpolation_mode = transforms.InterpolationMode.NEAREST_EXACT
    elif interpolation_type == "hamming":
        interpolation_mode = transforms.InterpolationMode.HAMMING
    elif interpolation_type == "lanczos":
        interpolation_mode = transforms.InterpolationMode.LANCZOS
    else:
        raise ValueError(
            f"The given interpolation mode {interpolation_type} is not supported. Currently supported interpolation"
            f" modes are `bilinear`, `bicubic`, `box`, `nearest`, `nearest_exact`, `hamming`, and `lanczos`."
        )

    return interpolation_mode


def compute_dream_and_update_latents(
    unet: UNet2DConditionModel,
    noise_scheduler: SchedulerMixin,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    noisy_latents: torch.Tensor,
    target: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    dream_detail_preservation: float = 1.0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Implements "DREAM (Diffusion Rectification and Estimation-Adaptive Models)" from http://arxiv.org/abs/2312.00210.
    DREAM helps align training with sampling to help training be more efficient and accurate at the cost of an extra
    forward step without gradients.

    Args:
        `unet`: The state unet to use to make a prediction.
        `noise_scheduler`: The noise scheduler used to add noise for the given timestep.
        `timesteps`: The timesteps for the noise_scheduler to user.
        `noise`: A tensor of noise in the shape of noisy_latents.
        `noisy_latents`: Previously noise latents from the training loop.
        `target`: The ground-truth tensor to predict after eps is removed.
        `encoder_hidden_states`: Text embeddings from the text model.
        `dream_detail_preservation`: A float value that indicates detail preservation level.
          See reference.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: Adjusted noisy_latents and target.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)[timesteps, None, None, None]
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # The paper uses lambda = sqrt(1 - alpha) ** p, with p = 1 in their experiments.
    dream_lambda = sqrt_one_minus_alphas_cumprod**dream_detail_preservation

    pred = None
    with torch.no_grad():
        pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    _noisy_latents, _target = (None, None)
    if noise_scheduler.config.prediction_type == "epsilon":
        predicted_noise = pred
        delta_noise = (noise - predicted_noise).detach()
        delta_noise.mul_(dream_lambda)
        _noisy_latents = noisy_latents.add(sqrt_one_minus_alphas_cumprod * delta_noise)
        _target = target.add(delta_noise)
    elif noise_scheduler.config.prediction_type == "v_prediction":
        raise NotImplementedError("DREAM has not been implemented for v-prediction")
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    return _noisy_latents, _target


def unet_lora_state_dict(unet: UNet2DConditionModel) -> Dict[str, torch.Tensor]:
    r"""
    Returns:
        A state dict containing just the LoRA parameters.
    """
    lora_state_dict = {}

    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                current_lora_layer_sd = lora_layer.state_dict()
                for lora_layer_matrix_name, lora_param in current_lora_layer_sd.items():
                    # The matrix name can either be "down" or "up".
                    lora_state_dict[f"{name}.lora.{lora_layer_matrix_name}"] = lora_param

    return lora_state_dict


def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32):
    """
    Casts the training parameters of the model to the specified data type.

    Args:
        model: The PyTorch model whose parameters will be cast.
        dtype: The data type to which the model parameters will be cast.
    """
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if param.requires_grad:
                param.data = param.to(dtype)


def _set_state_dict_into_text_encoder(
    lora_state_dict: Dict[str, torch.Tensor], prefix: str, text_encoder: torch.nn.Module
):
    """
    Sets the `lora_state_dict` into `text_encoder` coming from `transformers`.

    Args:
        lora_state_dict: The state dictionary to be set.
        prefix: String identifier to retrieve the portion of the state dict that belongs to `text_encoder`.
        text_encoder: Where the `lora_state_dict` is to be set.
    """

    text_encoder_state_dict = {
        f'{k.replace(prefix, "")}': v for k, v in lora_state_dict.items() if k.startswith(prefix)
    }
    text_encoder_state_dict = convert_state_dict_to_peft(convert_state_dict_to_diffusers(text_encoder_state_dict))
    set_peft_model_state_dict(text_encoder, text_encoder_state_dict, adapter_name="default")


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """
    Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


def free_memory():
    """
    Runs garbage collection. Then clears the cache of the available accelerator.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif is_torch_npu_available():
        torch_npu.npu.empty_cache()


# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
def should_update_ema(args, step):
    if args.ema_update_interval is None:
        # If the EMA update interval is not set, always update the EMA.
        return True
    else:
        should_update = step % args.ema_update_interval == 0
        # if should_update:
        #     logger.info("Updating EMA weights...")
        return should_update


class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        args,
        accelerator,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
        foreach: bool = True,
        model_cls: Optional[Any] = None,
        model_config: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Args:
            parameters (Iterable[torch.nn.Parameter]): The parameters to track.
            decay (float): The decay factor for the exponential moving average.
            min_decay (float): The minimum decay factor for the exponential moving average.
            update_after_step (int): The number of steps to wait before starting to update the EMA weights.
            use_ema_warmup (bool): Whether to use EMA warmup.
            inv_gamma (float):
                Inverse multiplicative factor of EMA warmup. Default: 1. Only used if `use_ema_warmup` is True.
            power (float): Exponential factor of EMA warmup. Default: 2/3. Only used if `use_ema_warmup` is True.
            foreach (bool): Use torch._foreach functions for updating shadow parameters. Should be faster.
            device (Optional[Union[str, torch.device]]): The device to store the EMA weights on. If None, the EMA
                        weights will be stored on CPU.

        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        """

        if isinstance(parameters, torch.nn.Module):
            deprecation_message = (
                "Passing a `torch.nn.Module` to `ExponentialMovingAverage` is deprecated. "
                "Please pass the parameters of the module instead."
            )
            deprecate(
                "passing a `torch.nn.Module` to `ExponentialMovingAverage`",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            parameters = parameters.parameters()

            # set use_ema_warmup to True if a torch.nn.Module is passed for backwards compatibility
            use_ema_warmup = True

        if kwargs.get("max_value", None) is not None:
            deprecation_message = (
                "The `max_value` argument is deprecated. Please use `decay` instead."
            )
            deprecate("max_value", "1.0.0", deprecation_message, standard_warn=False)
            decay = kwargs["max_value"]

        if kwargs.get("min_value", None) is not None:
            deprecation_message = "The `min_value` argument is deprecated. Please use `min_decay` instead."
            deprecate("min_value", "1.0.0", deprecation_message, standard_warn=False)
            min_decay = kwargs["min_value"]

        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        if kwargs.get("device", None) is not None:
            deprecation_message = (
                "The `device` argument is deprecated. Please use `to` instead."
            )
            deprecate("device", "1.0.0", deprecation_message, standard_warn=False)
            self.to(device=kwargs["device"])

        self.temp_stored_params = None

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value = None  # set in `step()`
        self.foreach = foreach

        self.model_cls = model_cls
        self.model_config = model_config
        self.args = args
        self.accelerator = accelerator
        self.training = True  # To emulate nn.Module's training mode

    def save_state_dict(self, path: str) -> None:
        """
        Save the EMA model's state directly to a file.

        Args:
            path (str): The file path where the EMA state will be saved.
        """
        # if the folder containing the path does not exist, create it
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # grab state dict
        state_dict = self.state_dict()
        # save it using torch.save
        torch.save(state_dict, path)
        logger.info(f"EMA model state saved to {path}")

    def load_state_dict(self, path: str) -> None:
        """
        Load the EMA model's state from a file and apply it to this instance.

        Args:
            path (str): The file path from where the EMA state will be loaded.
        """
        state_dict = torch.load(path, map_location="cpu", weights_only=True)

        # Load metadata
        self.decay = state_dict.get("decay", self.decay)
        self.min_decay = state_dict.get("min_decay", self.min_decay)
        self.optimization_step = state_dict.get(
            "optimization_step", self.optimization_step
        )
        self.update_after_step = state_dict.get(
            "update_after_step", self.update_after_step
        )
        self.use_ema_warmup = state_dict.get("use_ema_warmup", self.use_ema_warmup)
        self.inv_gamma = state_dict.get("inv_gamma", self.inv_gamma)
        self.power = state_dict.get("power", self.power)

        # Load shadow parameters
        shadow_params = []
        idx = 0
        while f"shadow_params.{idx}" in state_dict:
            shadow_params.append(state_dict[f"shadow_params.{idx}"])
            idx += 1

        if len(shadow_params) != len(self.shadow_params):
            raise ValueError(
                f"Mismatch in number of shadow parameters: expected {len(self.shadow_params)}, "
                f"but found {len(shadow_params)} in the state dict."
            )

        for current_param, loaded_param in zip(self.shadow_params, shadow_params):
            current_param.data.copy_(loaded_param.data)

        logger.info(f"EMA model state loaded from {path}")

    @classmethod
    def from_pretrained(cls, path, model_cls) -> "EMAModel":
        _, ema_kwargs = model_cls.load_config(path, return_unused_kwargs=True)
        model = model_cls.from_pretrained(path)

        ema_model = cls(
            model.parameters(), model_cls=model_cls, model_config=model.config
        )

        ema_model.load_state_dict(ema_kwargs)
        return ema_model

    def save_pretrained(self, path, max_shard_size: str = "10GB"):
        if self.model_cls is None:
            raise ValueError(
                "`save_pretrained` can only be used if `model_cls` was defined at __init__."
            )

        if self.model_config is None:
            raise ValueError(
                "`save_pretrained` can only be used if `model_config` was defined at __init__."
            )

        model = self.model_cls.from_config(self.model_config)
        state_dict = self.state_dict(exclude_params=True)
        state_dict.pop("shadow_params", None)

        model.register_to_config(**state_dict)
        self.copy_to(model.parameters())
        model.save_pretrained(path, max_shard_size=max_shard_size)

    def get_decay(self, optimization_step: int = None) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        if optimization_step is None:
            optimization_step = self.optimization_step

        step = max(0, optimization_step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)

        cur_decay_value = min(cur_decay_value, self.decay)
        # make sure decay is not smaller than min_decay
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter], global_step: int = None):
        if not should_update_ema(self.args, global_step):
            return
        # print(self.args.ema_device, not self.args.ema_cpu_only)
        if self.args.ema_device == "cpu" and not self.args.ema_cpu_only:
            # Move EMA to accelerator for faster update.
            self.to(device=self.accelerator.device, non_blocking=True)
        if isinstance(parameters, torch.nn.Module):
            deprecation_message = (
                "Passing a `torch.nn.Module` to `ExponentialMovingAverage.step` is deprecated. "
                "Please pass the parameters of the module instead."
            )
            deprecate(
                "passing a `torch.nn.Module` to `ExponentialMovingAverage.step`",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            parameters = parameters.parameters()

        parameters = list(parameters)

        if global_step is not None:
            # When we're updating the EMA periodically, we can't trust the counter.
            self.optimization_step = global_step
        else:
            self.optimization_step += 1

        # Compute the decay factor for the exponential moving average.
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        context_manager = contextlib.nullcontext
        if (
            is_transformers_available()
            and transformers.integrations.deepspeed.is_deepspeed_zero3_enabled()
        ):
            import deepspeed

        if self.foreach:
            if (
                is_transformers_available()
                and transformers.integrations.deepspeed.is_deepspeed_zero3_enabled()
            ):
                context_manager = deepspeed.zero.GatheredParameters(
                    parameters, modifier_rank=None
                )

            with context_manager():
                if self.args.ema_cpu_only:
                    params_grad = [param.cpu() for param in parameters if param.requires_grad]
                else:
                    params_grad = [param for param in parameters if param.requires_grad]
                s_params_grad = [
                    s_param
                    for s_param, param in zip(self.shadow_params, parameters)
                    if param.requires_grad
                ]

                if len(params_grad) < len(parameters):
                    if self.args.ema_cpu_only:
                        torch._foreach_copy_(
                            [
                                s_param
                                for s_param, param in zip(self.shadow_params, parameters)
                                if not param.requires_grad
                            ],
                            [param.cpu() for param in parameters if not param.requires_grad],
                            non_blocking=True,
                        )
                    else:
                        if self.args.ema_cpu_only:
                            torch._foreach_copy_(
                                [
                                    s_param
                                    for s_param, param in zip(self.shadow_params, parameters)
                                    if not param.requires_grad
                                ],
                                [param for param in parameters if not param.requires_grad],
                                non_blocking=True,
                            )      

                torch._foreach_sub_(
                    s_params_grad,
                    torch._foreach_sub(s_params_grad, params_grad),
                    alpha=one_minus_decay,
                )

        else:
            for s_param, param in zip(self.shadow_params, parameters):
                if (
                    is_transformers_available()
                    and transformers.integrations.deepspeed.is_deepspeed_zero3_enabled()
                ):
                    context_manager = deepspeed.zero.GatheredParameters(
                        param, modifier_rank=None
                    )

                with context_manager():
                    if param.requires_grad:
                        s_param.sub_(
                            one_minus_decay * (s_param - param.to(s_param.device))
                        )
                    else:
                        s_param.copy_(param)
        if self.args.ema_device == "cpu" and not self.args.ema_cpu_only:
            # Move back to CPU for safe-keeping.
            self.to(device=self.args.ema_device, non_blocking=True)

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        parameters = list(parameters)
        if self.foreach:
            torch._foreach_copy_(
                [param.data for param in parameters],
                [
                    s_param.to(param.device).data
                    for s_param, param in zip(self.shadow_params, parameters)
                ],
            )
        else:
            for s_param, param in zip(self.shadow_params, parameters):
                param.data.copy_(s_param.to(param.device).data)

    def pin_memory(self) -> None:
        r"""
        Move internal buffers of the ExponentialMovingAverage to pinned memory. Useful for non-blocking transfers for
        offloading EMA params to the host.
        """
        if torch.backends.mps.is_available():
            logger.warning("Apple silicon does not support pinned memory. Skipping.")
            return

        if self.args.ema_cpu_only:
            return

        # This probably won't work, but we'll do it anyway.
        self.shadow_params = [p.pin_memory() for p in self.shadow_params]

    def to(self, *args, **kwargs):
        for param in self.shadow_params:
            param.data = param.data.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        return self.to(device="cuda" if device is None else f"cuda:{device}")

    def cpu(self):
        return self.to(device="cpu")

    def state_dict(
        self, destination=None, prefix="", keep_vars=False, exclude_params: bool = False
    ):
        r"""
        Returns a dictionary containing a whole state of the EMA model.
        """
        state_dict = {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
        }
        if exclude_params:
            return state_dict
        for idx, param in enumerate(self.shadow_params):
            state_dict[f"{prefix}shadow_params.{idx}"] = (
                param if keep_vars else param.detach()
            )
        return state_dict

    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""
        Save the current parameters for restoring later.
        """
        self.temp_stored_params = [param.detach().cpu().clone() for param in parameters]

    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""
        Restore the parameters stored with the `store` method.
        """
        if self.temp_stored_params is None:
            raise RuntimeError(
                "This ExponentialMovingAverage has no `store()`ed weights "
                "to `restore()`"
            )
        if self.foreach:
            torch._foreach_copy_(
                [param.data for param in parameters],
                [c_param.data for c_param in self.temp_stored_params],
            )
        else:
            for c_param, param in zip(self.temp_stored_params, parameters):
                param.data.copy_(c_param.data)

        # Better memory-wise.
        self.temp_stored_params = None

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.shadow_params)

    # Implementing nn.Module methods to emulate its behavior

    def named_children(self):
        # No child modules
        return iter([])

    def children(self):
        return iter([])

    def modules(self):
        yield self

    def named_modules(self, memo=None, prefix=""):
        yield prefix, self

    def parameters(self, recurse=True):
        return iter(self.shadow_params)

    def named_parameters(self, prefix="", recurse=True):
        for i, param in enumerate(self.shadow_params):
            name = f"{prefix}shadow_params.{i}"
            yield name, param

    def buffers(self, recurse=True):
        return iter([])

    def named_buffers(self, prefix="", recurse=True):
        return iter([])

    def train(self, mode=True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        # No gradients to zero in EMA model
        pass