from diffusers import (
    StableDiffusionPipeline,
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
    KDPM2AncestralDiscreteScheduler,)
import inspect
import torch

def get_scheduler(args) :

    sched_init_args = {}

    SCHEDULER_LINEAR_START = 0.00085
    SCHEDULER_LINEAR_END = 0.0120
    SCHEDULER_TIMESTEPS = 1000
    SCHEDLER_SCHEDULE = "scaled_linear"

    if args.sample_sampler == "ddim":
        scheduler_cls = DDIMScheduler
    elif args.sample_sampler == "ddpm":  # ddpmはおかしくなるのでoptionから外してある
        scheduler_cls = DDPMScheduler
    elif args.sample_sampler == "pndm":
        scheduler_cls = PNDMScheduler
    elif args.sample_sampler == "lms" or args.sample_sampler == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
    elif args.sample_sampler == "euler" or args.sample_sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
    elif args.sample_sampler == "euler_a" or args.sample_sampler == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif args.sample_sampler == "dpmsolver" or args.sample_sampler == "dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = args.sample_sampler
    elif args.sample_sampler == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
    elif args.sample_sampler == "heun":
        scheduler_cls = HeunDiscreteScheduler
    elif args.sample_sampler == "dpm_2" or args.sample_sampler == "k_dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
    elif args.sample_sampler == "dpm_2_a" or args.sample_sampler == "k_dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    else:
        scheduler_cls = DDIMScheduler
    if args.v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction"

    scheduler = scheduler_cls(num_train_timesteps=SCHEDULER_TIMESTEPS,
                              beta_start=SCHEDULER_LINEAR_START,
                              beta_end=SCHEDULER_LINEAR_END,
                              beta_schedule=SCHEDLER_SCHEDULE,
                              **sched_init_args,)
    return scheduler

def retrieve_timesteps(scheduler,
                       num_inference_steps,
                       device,
                       timesteps,
                       **kwargs,):
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    else:
        # scheduler.set timesteps
        scheduler.set_timesteps(num_inference_steps,
                                device=device,
                                **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps

def sample_images(dataloader,
                  condition_model,
                  weight_dtype,
                  simple_linear,
                  device,
                  timesteps,
                  num_inference_steps,
                  unet,
                  vae,
                  args):

    # [1] make stable diffusion pipeline
    # [1.1] scheduler
    scheduler = get_scheduler(args)

    for i, batch in enumerate(dataloader) :

        if i == 0 :
            # [2] image generating
            height, width = 64,64
            # [3] generate condition
            condition_pixel = batch['condition_image']['pixel_values'].to(dtype=weight_dtype, )
            batch['condition_image']['pixel_values'] = condition_pixel
            feat = condition_model(**batch['condition_image']).last_hidden_state  # processor output
            encoder_hidden_states = simple_linear(feat.contiguous())  # [batch=1, 197, 768]
            # [4] generating
            timesteps, num_inference_steps = retrieve_timesteps(scheduler,
                                                                num_inference_steps, # 30
                                                                device,
                                                                timesteps)

            # 5. Prepare latent variables
            latents = torch.randn(1, 4, height, width).to(device, dtype=weight_dtype)
            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = {}

            # 6.2 Optionally get Guidance Scale Embedding
            timestep_cond = None

            # 7. Denoising loop
            for i, t in enumerate(timesteps):

                # expand the latents if we are doing classifier free guidance
                latent_model_input = latents

                # predict the noise residual
                noise_pred = unet(latent_model_input,
                                  t,
                                  encoder_hidden_states=encoder_hidden_states,
                                  timestep_cond=timestep_cond,
                                  return_dict=False,)[0]
                latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # 8. final image
            image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]

    return image
