import argparse
import time
from accelerate import Accelerator
def prepare_accelerator(args: argparse.Namespace):
    print(f'args.logging_dir : {args.logging_dir}')
    if args.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if args.log_prefix is None else args.log_prefix
        logging_dir = args.logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

    if args.log_with is None:
        if logging_dir is not None:
            log_with = "tensorboard"
        else:
            log_with = None
    else:
        log_with = args.log_with
        if log_with in ["tensorboard", "all"]:
            if logging_dir is None:
                raise ValueError("logging_dir is required when log_with is tensorboard / Tensorboardを使う場合、logging_dirを指定してください")

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,
                              log_with=log_with,
                              project_dir=logging_dir,)
    return accelerator