from diffusers import StableDiffusionPipeline
import argparse
from accelerate import Accelerator, PartialState
from diffusers import (
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    AutoencoderKL,)
from lpw_diffusion import StableDiffusionLongPromptWeightingPipeline
import os
import torch
import time
import gc
import toml, json
import re
# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"

def clean_memory_on_device(device: torch.device):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()

    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()

def get_my_scheduler(
    *,
    sample_sampler: str,
    v_parameterization: bool,
):
    sched_init_args = {}
    if sample_sampler == "ddim":
        scheduler_cls = DDIMScheduler
    elif sample_sampler == "ddpm":  # ddpmはおかしくなるのでoptionから外してある
        scheduler_cls = DDPMScheduler
    elif sample_sampler == "pndm":
        scheduler_cls = PNDMScheduler
    elif sample_sampler == "lms" or sample_sampler == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
    elif sample_sampler == "euler" or sample_sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
    elif sample_sampler == "euler_a" or sample_sampler == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif sample_sampler == "dpmsolver" or sample_sampler == "dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = sample_sampler
    elif sample_sampler == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
    elif sample_sampler == "heun":
        scheduler_cls = HeunDiscreteScheduler
    elif sample_sampler == "dpm_2" or sample_sampler == "k_dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
    elif sample_sampler == "dpm_2_a" or sample_sampler == "k_dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    else:
        scheduler_cls = DDIMScheduler

    if v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction"

    scheduler = scheduler_cls(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
        **sched_init_args,
    )

    # clip_sample=Trueにする
    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
        # logger.info("set clip_sample to True")
        scheduler.config.clip_sample = True

    return scheduler

def sample_image_inference(
    accelerator: Accelerator,
    args: argparse.Namespace,
    pipeline,
    save_dir,
    prompt_dict,
    epoch,
    steps,
    prompt_replacement,
    controlnet=None,
):
    assert isinstance(prompt_dict, dict)


    negative_prompt = prompt_dict.get("negative_prompt")
    sample_steps = prompt_dict.get("sample_steps", 30)
    width = prompt_dict.get("width", 512)
    height = prompt_dict.get("height", 512)
    scale = prompt_dict.get("scale", 7.5)
    seed = prompt_dict.get("seed")
    controlnet_image = prompt_dict.get("controlnet_image")
    prompt: str = prompt_dict.get("prompt", "")
    sampler_name: str = prompt_dict.get("sample_sampler", args.sample_sampler)

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        if negative_prompt is not None:
            negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    else:
        # True random sample image generation
        torch.seed()
        torch.cuda.seed()

    scheduler = get_my_scheduler(
        sample_sampler=sampler_name,
        v_parameterization=args.v_parameterization,
    )
    pipeline.scheduler = scheduler

    if controlnet_image is not None:
        controlnet_image = Image.open(controlnet_image).convert("RGB")
        controlnet_image = controlnet_image.resize((width, height), Image.LANCZOS)

    height = max(64, height - height % 8)  # round to divisible by 8
    width = max(64, width - width % 8)  # round to divisible by 8
    with accelerator.autocast():
        latents = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=sample_steps,
            guidance_scale=scale,
            negative_prompt=negative_prompt,
            controlnet=controlnet,
            controlnet_image=controlnet_image,
        )

    with torch.cuda.device(torch.cuda.current_device()):
        torch.cuda.empty_cache()
    image = pipeline.latents_to_image(latents)[0]

    # adding accelerator.wait_for_everyone() here should sync up and ensure that sample images are saved in the same order as the original prompt list
    # but adding 'enum' to the filename should be enough

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i: int = prompt_dict["enum"]
    img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
    image.save(os.path.join(save_dir, img_filename))

    # wandb有効時のみログを送信
    try:
        wandb_tracker = accelerator.get_tracker("wandb")
        try:
            import wandb
        except ImportError:  # 事前に一度確認するのでここはエラー出ないはず
            raise ImportError("No wandb / wandb がインストールされていないようです")

        wandb_tracker.log({f"sample_{i}": wandb.Image(image)})
    except:  # wandb 無効時
        pass

def sample_images(*args, **kwargs):
    return sample_images_common(StableDiffusionLongPromptWeightingPipeline, *args, **kwargs)

def line_to_prompt_dict(line: str) -> dict:
    # subset of gen_img_diffusers
    prompt_args = line.split(" --")
    prompt_dict = {}
    prompt_dict["prompt"] = prompt_args[0]

    for parg in prompt_args:
        try:
            m = re.match(r"w (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["width"] = int(m.group(1))
                continue

            m = re.match(r"h (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["height"] = int(m.group(1))
                continue

            m = re.match(r"d (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["seed"] = int(m.group(1))
                continue

            m = re.match(r"s (\d+)", parg, re.IGNORECASE)
            if m:  # steps
                prompt_dict["sample_steps"] = max(1, min(1000, int(m.group(1))))
                continue

            m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # scale
                prompt_dict["scale"] = float(m.group(1))
                continue

            m = re.match(r"n (.+)", parg, re.IGNORECASE)
            if m:  # negative prompt
                prompt_dict["negative_prompt"] = m.group(1)
                continue

            m = re.match(r"ss (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["sample_sampler"] = m.group(1)
                continue

            m = re.match(r"cn (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["controlnet_image"] = m.group(1)
                continue

        except ValueError as ex:
            print(f"Exception in parsing / 解析エラー: {parg}")

    return prompt_dict


def sample_images_common(
    pipe_class,
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch,
    steps,
    vae,
    tokenizer,
    text_encoder,
    unet,
    prompt_replacement=None,
    controlnet=None,
):
    """
    StableDiffusionLongPromptWeightingPipelineの改造版を使うようにしたので、clip skipおよびプロンプトの重みづけに対応した
    """

    if steps == 0:
        if not args.sample_at_first:
            return
    else:
        if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
            return
        if args.sample_every_n_epochs is not None:
            # sample_every_n_steps は無視する
            if epoch is None or epoch % args.sample_every_n_epochs != 0:
                return
        else:
            if steps % args.sample_every_n_steps != 0 or epoch is not None:  # steps is not divisible or end of epoch
                return

    distributed_state = PartialState()  # for multi gpu distributed inference. this is a singleton, so it's safe to use it here

    org_vae_device = vae.device  # CPUにいるはず
    vae.to(distributed_state.device)  # distributed_state.device is same as accelerator.device

    # unwrap unet and text_encoder(s)
    unet = accelerator.unwrap_model(unet)
    if isinstance(text_encoder, (list, tuple)):
        text_encoder = [accelerator.unwrap_model(te) for te in text_encoder]
    else:
        text_encoder = accelerator.unwrap_model(text_encoder)

    # read prompts
    if args.sample_prompts.endswith(".txt"):
        with open(args.sample_prompts, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    elif args.sample_prompts.endswith(".toml"):
        with open(args.sample_prompts, "r", encoding="utf-8") as f:
            data = toml.load(f)
        prompts = [dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]]
    elif args.sample_prompts.endswith(".json"):
        with open(args.sample_prompts, "r", encoding="utf-8") as f:
            prompts = json.load(f)

    # schedulers: dict = {}  cannot find where this is used
    default_scheduler = get_my_scheduler(sample_sampler=args.sample_sampler,
                                         v_parameterization=args.v_parameterization,)

    pipeline = pipe_class(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=default_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        clip_skip=args.clip_skip,
    )
    pipeline.to(distributed_state.device)
    save_dir = args.output_dir + "/sample"
    os.makedirs(save_dir, exist_ok=True)

    # preprocess prompts
    for i in range(len(prompts)):
        prompt_dict = prompts[i]
        if isinstance(prompt_dict, str):
            prompt_dict = line_to_prompt_dict(prompt_dict)
            prompts[i] = prompt_dict
        assert isinstance(prompt_dict, dict)

        # Adds an enumerator to the dict based on prompt position. Used later to name image files. Also cleanup of extra data in original prompt dict.
        prompt_dict["enum"] = i
        prompt_dict.pop("subset", None)

    # save random state to restore later
    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    if distributed_state.num_processes <= 1:
        # If only one device is available, just use the original prompt list. We don't need to care about the distribution of prompts.
        with torch.no_grad():
            for prompt_dict in prompts:
                sample_image_inference(
                    accelerator, args, pipeline, save_dir, prompt_dict, epoch, steps, prompt_replacement, controlnet=controlnet
                )
    else:
        # Creating list with N elements, where each element is a list of prompt_dicts, and N is the number of processes available (number of devices available)
        # prompt_dicts are assigned to lists based on order of processes, to attempt to time the image creation time to match enum order. Probably only works when steps and sampler are identical.
        per_process_prompts = []  # list of lists
        for i in range(distributed_state.num_processes):
            per_process_prompts.append(prompts[i :: distributed_state.num_processes])

        with torch.no_grad():
            with distributed_state.split_between_processes(per_process_prompts) as prompt_dict_lists:
                for prompt_dict in prompt_dict_lists[0]:
                    sample_image_inference(
                        accelerator, args, pipeline, save_dir, prompt_dict, epoch, steps, prompt_replacement, controlnet=controlnet
                    )

    # clear pipeline and cache to reduce vram usage
    del pipeline

    # I'm not sure which of these is the correct way to clear the memory, but accelerator's device is used in the pipeline, so I'm using it here.
    # with torch.cuda.device(torch.cuda.current_device()):
    #     torch.cuda.empty_cache()
    clean_memory_on_device(accelerator.device)

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    vae.to(org_vae_device)