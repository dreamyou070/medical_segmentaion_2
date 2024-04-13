# Copyright (c) MONA Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlparse

import torch

from monai.apps import download_and_extract
from monai.apps.utils import get_logger
from monai.auto3dseg.algo_gen import Algo, AlgoGen
from monai.auto3dseg.utils import (
    _prepare_cmd_bcprun,
    _prepare_cmd_default,
    _prepare_cmd_torchrun,
    _run_cmd_bcprun,
    _run_cmd_torchrun,
    algo_to_pickle,
)
from monai.bundle.config_parser import ConfigParser
from monai.config import PathLike
from monai.utils import ensure_tuple, look_up_option, run_cmd
from monai.utils.enums import AlgoKeys
from monai.utils.misc import MONAIEnvVars

logger = get_logger(module_name=__name__)
ALGO_HASH = MONAIEnvVars.algo_hash()

__all__ = ["BundleAlgo", "BundleGen"]


class BundleAlgo(Algo):
    """
    An algorithm represented by a set of bundle configurations and scripts.

    ``BundleAlgo.cfg`` is a ``monai.bundle.ConfigParser`` instance.

    .. code-block:: python

        from monai.apps.auto3dseg import BundleAlgo

        data_stats_yaml = "../datastats.yaml"
        algo = BundleAlgo(template_path="../algorithm_templates")
        algo.set_data_stats(data_stats_yaml)
        # algo.set_data_src("../data_src.json")
        algo.export_to_disk(".", algo_name="segresnet2d_1")

    This class creates MONA bundles from a directory of 'bundle template'. Different from the regular MONA bundle
    format, the bundle template may contain placeholders that must be filled using ``fill_template_config`` during
    ``export_to_disk``. Then created bundle keeps the same file structure as the template.

    """

    def __init__(self, template_path: PathLike):
        """
        Create an Algo instance based on the predefined Algo template.

        Args:
            template_path: path to a folder that contains the algorithm templates.
                Please check https://github.com/Project-MONAI/research-contributions/tree/main/auto3dseg/algorithm_templates

        """

        self.template_path = template_path
        self.data_stats_files = ""
        self.data_list_file = ""
        self.mlflow_tracking_uri: str | None = None
        self.mlflow_experiment_name: str | None = None
        self.output_path = ""
        self.name = ""
        self.best_metric = None
        # track records when filling template config: {"<config name>": {"<placeholder key>": value, ...}, ...}
        self.fill_records: dict = {}
        # device_setting set default value and sanity check, in case device_setting not from autorunner
        self.device_setting: dict[str, int | str] = {
            "CUDA_VISIBLE_DEVICES": ",".join([str(x) for x in range(torch.cuda.device_count())]),
            "n_devices": int(torch.cuda.device_count()),
            "NUM_NODES": int(os.environ.get("NUM_NODES", 1)),
            "MN_START_METHOD": os.environ.get("MN_START_METHOD", "bcprun"),
            "CMD_PREFIX": os.environ.get("CMD_PREFIX", ""),
        }

    def pre_check_skip_algo(self, skip_bundlegen: bool = False, skip_info: str = "") -> tuple[bool, str]:
        """
        Analyse the data analysis report and check if the algorithm needs to be skipped.
        This function is overriden within algo.
        Args:
            skip_bundlegen: skip generating bundles for this algo if true.
            skip_info: info to print when skipped.
        """
        return skip_bundlegen, skip_info

    def set_data_stats(self, data_stats_files: str) -> None:
        """
        Set the data analysis report (generated by DataAnalyzer).

        Args:
            data_stats_files: path to the datastats yaml file
        """
        self.data_stats_files = data_stats_files

    def set_data_source(self, data_src_cfg: str) -> None:
        """
        Set the data source configuration file

        Args:
            data_src_cfg: path to a configuration file (yaml) that contains datalist, dataroot, and other params.
                The config will be in a form of {"modality": "ct", "datalist": "path_to_json_datalist", "dataroot":
                "path_dir_data"}
        """
        self.data_list_file = data_src_cfg

    def set_mlflow_tracking_uri(self, mlflow_tracking_uri: str | None) -> None:
        """
        Set the tracking URI for MLflow server

        Args:
            mlflow_tracking_uri: a tracking URI for MLflow server which could be local directory or address of
                the remote tracking Server; MLflow runs will be recorded locally in algorithms' model folder if
                the value is None.
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri

    def set_mlflow_experiment_name(self, mlflow_experiment_name: str | None) -> None:
        """
        Set the experiment name for MLflow server

        Args:
            mlflow_experiment_name: a string to specify the experiment name for MLflow server.
        """
        self.mlflow_experiment_name = mlflow_experiment_name

    def fill_template_config(self, data_stats_filename: str, algo_path: str, **kwargs: Any) -> dict:
        """
        The configuration files defined when constructing this Algo instance might not have a complete training
        and validation pipelines. Some configuration components and hyperparameters of the pipelines depend on the
        training data and other factors. This API is provided to allow the creation of fully functioning config files.
        Return the records of filling template config: {"<config name>": {"<placeholder key>": value, ...}, ...}.

        Args:
            data_stats_filename: filename of the data stats report (generated by DataAnalyzer)

        Notes:
            Template filling is optional. The user can construct a set of pre-filled configs without replacing values
            by using the data analysis results. It is also intended to be re-implemented in subclasses of BundleAlgo
            if the user wants their own way of auto-configured template filling.
        """
        return {}

    def export_to_disk(self, output_path: str, algo_name: str, **kwargs: Any) -> None:
        """
        Fill the configuration templates, write the bundle (configs + scripts) to folder `output_path/algo_name`.

        Args:
            output_path: Path to export the 'scripts' and 'configs' directories.
            algo_name: the identifier of the algorithm (usually contains the name and extra info like fold ID).
            kwargs: other parameters, including: "copy_dirs=True/False" means whether to copy the template as output
                instead of inplace operation, "fill_template=True/False" means whether to fill the placeholders
                in the template. other parameters are for `fill_template_config` function.

        """
        if kwargs.pop("copy_dirs", True):
            self.output_path = os.path.join(output_path, algo_name)
            os.makedirs(self.output_path, exist_ok=True)
            if os.path.isdir(self.output_path):
                shutil.rmtree(self.output_path)
            # copy algorithm_templates/<Algo> to the working directory output_path
            shutil.copytree(os.path.join(str(self.template_path), self.name), self.output_path)
        else:
            self.output_path = str(self.template_path)
        if kwargs.pop("fill_template", True):
            self.fill_records = self.fill_template_config(self.data_stats_files, self.output_path, **kwargs)
        logger.info(f"Generated:{self.output_path}")

    def _create_cmd(self, train_params: None | dict = None) -> tuple[str, str]:
        """
        Create the command to execute training.

        """
        if train_params is None:
            train_params = {}
        params = deepcopy(train_params)

        train_py = os.path.join(self.output_path, "scripts", "train.py")
        config_dir = os.path.join(self.output_path, "configs")

        config_files = []
        if os.path.isdir(config_dir):
            for file in sorted(os.listdir(config_dir)):
                if file.endswith("yaml") or file.endswith("json"):
                    # Python Fire may be confused by single-quoted WindowsPath
                    config_files.append(Path(os.path.join(config_dir, file)).as_posix())

        if int(self.device_setting["NUM_NODES"]) > 1:
            # multi-node command
            # only bcprun is supported for now
            try:
                look_up_option(self.device_setting["MN_START_METHOD"], ["bcprun"])
            except ValueError as err:
                raise NotImplementedError(
                    f"{self.device_setting['MN_START_METHOD']} is not supported yet."
                    "Try modify BundleAlgo._create_cmd for your cluster."
                ) from err

            return (
                _prepare_cmd_bcprun(
                    f"{train_py} run",
                    cmd_prefix=f"{self.device_setting['CMD_PREFIX']}",
                    config_file=config_files,
                    **params,
                ),
                "",
            )
        elif int(self.device_setting["n_devices"]) > 1:
            return _prepare_cmd_torchrun(f"{train_py} run", config_file=config_files, **params), ""
        else:
            return (
                _prepare_cmd_default(
                    f"{train_py} run",
                    cmd_prefix=f"{self.device_setting['CMD_PREFIX']}",
                    config_file=config_files,
                    **params,
                ),
                "",
            )

    def _run_cmd(self, cmd: str, devices_info: str = "") -> subprocess.CompletedProcess:
        """
        Execute the training command with target devices information.

        """
        if devices_info:
            warnings.warn(f"input devices_info {devices_info} is deprecated and ignored.")

        ps_environ = os.environ.copy()
        ps_environ["CUDA_VISIBLE_DEVICES"] = str(self.device_setting["CUDA_VISIBLE_DEVICES"])

        # delete pattern "VAR=VALUE" at the beginning of the string, with optional leading/trailing whitespaces
        cmd = re.sub(r"^\s*\w+=.*?\s+", "", cmd)

        if int(self.device_setting["NUM_NODES"]) > 1:
            try:
                look_up_option(self.device_setting["MN_START_METHOD"], ["bcprun"])
            except ValueError as err:
                raise NotImplementedError(
                    f"{self.device_setting['MN_START_METHOD']} is not supported yet."
                    "Try modify BundleAlgo._run_cmd for your cluster."
                ) from err

            return _run_cmd_bcprun(cmd, n=self.device_setting["NUM_NODES"], p=self.device_setting["n_devices"])
        elif int(self.device_setting["n_devices"]) > 1:
            return _run_cmd_torchrun(
                cmd, nnodes=1, nproc_per_node=self.device_setting["n_devices"], env=ps_environ, check=True
            )
        else:
            return run_cmd(cmd.split(), run_cmd_verbose=True, env=ps_environ, check=True)

    def train(
        self, train_params: None | dict = None, device_setting: None | dict = None
    ) -> subprocess.CompletedProcess:
        """
        Load the run function in the training script of each model. Training parameter is predefined by the
        algo_config.yaml file, which is pre-filled by the fill_template_config function in the same instance.

        Args:
            train_params:  training parameters
            device_setting: device related settings, should follow the device_setting in auto_runner.set_device_info.
                'CUDA_VISIBLE_DEVICES' should be a string e.g. '0,1,2,3'
        """
        if device_setting is not None:
            self.device_setting.update(device_setting)
            self.device_setting["n_devices"] = len(str(self.device_setting["CUDA_VISIBLE_DEVICES"]).split(","))

        if train_params is not None and "CUDA_VISIBLE_DEVICES" in train_params:
            warnings.warn("CUDA_VISIBLE_DEVICES is deprecated from train_params!")
            train_params.pop("CUDA_VISIBLE_DEVICES")

        cmd, _unused_return = self._create_cmd(train_params)
        return self._run_cmd(cmd)

    def get_score(self, *args, **kwargs):
        """
        Returns validation scores of the model trained by the current Algo.
        """
        config_yaml = os.path.join(self.output_path, "configs", "hyper_parameters.yaml")
        parser = ConfigParser()
        parser.read_config(config_yaml)
        ckpt_path = parser.get_parsed_content("ckpt_path", default=self.output_path)

        dict_file = ConfigParser.load_config_file(os.path.join(ckpt_path, "progress.yaml"))
        # dict_file: a list of scores saved in the form of dict in progress.yaml
        return dict_file[-1]["best_avg_dice_score"]  # the last one is the best one

    def get_inferer(self, *args, **kwargs):
        """
        Load the InferClass from the infer.py. The InferClass should be defined in the template under the path of
        `"scripts/infer.py"`. It is required to define the "InferClass" (name is fixed) with two functions at least
        (``__init__`` and ``infer``). The init class has an override kwargs that can be used to override parameters in
        the run-time optionally.

        Examples:

        .. code-block:: python

            class InferClass
                def __init__(self, config_file: Optional[Union[str, Sequence[str]]] = None, **override):
                    # read configs from config_file (sequence)
                    # set up transforms
                    # set up model
                    # set up other hyper parameters
                    return

                @torch.no_grad()
                def infer(self, image_file):
                    # infer the model and save the results to output
                    return output

        """
        infer_py = os.path.join(self.output_path, "scripts", "infer.py")
        if not os.path.isfile(infer_py):
            raise ValueError(f"{infer_py} is not found, please check the path.")

        config_dir = os.path.join(self.output_path, "configs")
        configs_path = [os.path.join(config_dir, f) for f in os.listdir(config_dir)]

        spec = importlib.util.spec_from_file_location("InferClass", infer_py)
        infer_class = importlib.util.module_from_spec(spec)  # type: ignore
        sys.modules["InferClass"] = infer_class
        spec.loader.exec_module(infer_class)  # type: ignore
        return infer_class.InferClass(configs_path, *args, **kwargs)

    def predict(self, predict_files: list, predict_params: dict | None = None) -> list:
        """
        Use the trained model to predict the outputs with a given input image.

        Args:
            predict_files: a list of paths to files to run inference on ["path_to_image_1", "path_to_image_2"]
            predict_params: a dict to override the parameters in the bundle config (including the files to predict).

        """
        params = {} if predict_params is None else deepcopy(predict_params)
        inferer = self.get_inferer(**params)
        return [inferer.infer(f) for f in ensure_tuple(predict_files)]

    def get_output_path(self):
        """Returns the algo output paths to find the algo scripts and configs."""
        return self.output_path


# path to download the algo_templates
default_algo_zip = (
    f"https://github.com/Project-MONAI/research-contributions/releases/download/algo_templates/{ALGO_HASH}.tar.gz"
)

# default algorithms
default_algos = {
    "segresnet2d": dict(_target_="segresnet2d.scripts.algo.Segresnet2dAlgo"),
    "dints": dict(_target_="dints.scripts.algo.DintsAlgo"),
    "swinunetr": dict(_target_="swinunetr.scripts.algo.SwinunetrAlgo"),
    "segresnet": dict(_target_="segresnet.scripts.algo.SegresnetAlgo"),
}


def _download_algos_url(url: str, at_path: str) -> dict[str, dict[str, str]]:
    """
    Downloads the algorithm templates release archive, and extracts it into a parent directory of the at_path folder.
    Returns a dictionary of the algorithm templates.
    """
    at_path = os.path.abspath(at_path)
    zip_download_dir = TemporaryDirectory()
    algo_compressed_file = os.path.join(zip_download_dir.name, "algo_templates.tar.gz")

    download_attempts = 3
    for i in range(download_attempts):
        try:
            download_and_extract(url=url, filepath=algo_compressed_file, output_dir=os.path.dirname(at_path))
        except Exception as e:
            msg = f"Download and extract of {url} failed, attempt {i+1}/{download_attempts}."
            if i < download_attempts - 1:
                warnings.warn(msg)
                time.sleep(i)
            else:
                zip_download_dir.cleanup()
                raise ValueError(msg) from e
        else:
            break

    zip_download_dir.cleanup()

    algos_all = deepcopy(default_algos)
    for name in algos_all:
        algos_all[name]["template_path"] = at_path

    return algos_all


def _copy_algos_folder(folder, at_path):
    """
    Copies the algorithm templates folder to at_path.
    Returns a dictionary of algorithm templates.
    """
    folder = os.path.abspath(folder)
    at_path = os.path.abspath(at_path)

    if folder != at_path:
        if os.path.exists(at_path):
            shutil.rmtree(at_path)
        shutil.copytree(folder, at_path)

    algos_all = {}
    for name in os.listdir(at_path):
        if os.path.exists(os.path.join(folder, name, "scripts", "algo.py")):
            algos_all[name] = dict(_target_=f"{name}.scripts.algo.{name.capitalize()}Algo", template_path=at_path)
            logger.info(f"Copying template: {name} -- {algos_all[name]}")
    if not algos_all:
        raise ValueError(f"Unable to find any algos in {folder}")

    return algos_all


class BundleGen(AlgoGen):
    """
    This class generates a set of bundles according to the cross-validation folds, each of them can run independently.

    Args:
        algo_path: the directory path to save the algorithm templates. Default is the current working dir.
        algos: If dictionary, it outlines the algorithm to use. If a list or a string, defines a subset of names of
            the algorithms to use, e.g. ('segresnet', 'dints') out of the full set of algorithm templates provided
            by templates_path_or_url. Defaults to None - to use all available algorithms.
        templates_path_or_url: the folder with the algorithm templates or a url. If None provided, the default template
            zip url will be downloaded and extracted into the algo_path. The current default options are released at:
            https://github.com/Project-MONAI/research-contributions/tree/main/auto3dseg.
        data_stats_filename: the path to the data stats file (generated by DataAnalyzer).
        data_src_cfg_name: the path to the data source config YAML file. The config will be in a form of
                           {"modality": "ct", "datalist": "path_to_json_datalist", "dataroot": "path_dir_data"}.
        mlflow_tracking_uri: a tracking URI for MLflow server which could be local directory or address of
            the remote tracking Server; MLflow runs will be recorded locally in algorithms' model folder if
            the value is None.
        mlfow_experiment_name: a string to specify the experiment name for MLflow server.
    .. code-block:: bash

        python -m monai.apps.auto3dseg BundleGen generate --data_stats_filename="../algorithms/datastats.yaml"
    """

    def __init__(
        self,
        algo_path: str = ".",
        algos: dict | list | str | None = None,
        templates_path_or_url: str | None = None,
        data_stats_filename: str | None = None,
        data_src_cfg_name: str | None = None,
        mlflow_tracking_uri: str | None = None,
        mlflow_experiment_name: str | None = None,
    ):
        if algos is None or isinstance(algos, (list, tuple, str)):
            if templates_path_or_url is None:
                templates_path_or_url = default_algo_zip

            at_path = os.path.join(os.path.abspath(algo_path), "algorithm_templates")

            if os.path.isdir(templates_path_or_url):
                # if a local folder, copy if necessary
                logger.info(f"BundleGen from directory {templates_path_or_url}")
                algos_all = _copy_algos_folder(folder=templates_path_or_url, at_path=at_path)
            elif urlparse(templates_path_or_url).scheme in ("http", "https"):
                # if url, trigger the download and extract process
                logger.info(f"BundleGen from {templates_path_or_url}")
                algos_all = _download_algos_url(url=templates_path_or_url, at_path=at_path)
            else:
                raise ValueError(f"{self.__class__} received invalid templates_path_or_url: {templates_path_or_url}")

            if algos is not None:
                algos = {k: v for k, v in algos_all.items() if k in ensure_tuple(algos)}  # keep only provided
                if len(algos) == 0:
                    raise ValueError(f"Unable to find provided algos in {algos_all}")
            else:
                algos = algos_all

        self.algos: Any = []
        if isinstance(algos, dict):
            for algo_name, algo_params in sorted(algos.items()):
                template_path = algo_params.get("template_path", ".")
                if len(template_path) > 0 and template_path not in sys.path:
                    sys.path.append(template_path)

                try:
                    onealgo = ConfigParser(algo_params).get_parsed_content()
                    onealgo.name = algo_name
                    self.algos.append(onealgo)
                except RuntimeError as e:
                    msg = """Please make sure the folder structure of an Algo Template follows
                        [algo_name]
                        ├── configs
                        │   ├── hyper_parameters.yaml  # automatically generated yaml from a set of ``template_configs``
                        └── scripts
                            ├── test.py
                            ├── __init__.py
                            └── validate.py
                    """
                    raise RuntimeError(msg) from e
        else:
            raise ValueError("Unexpected error algos is not a dict")

        self.data_stats_filename = data_stats_filename
        self.data_src_cfg_name = data_src_cfg_name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self.history: list[dict] = []

    def set_data_stats(self, data_stats_filename: str) -> None:
        """
        Set the data stats filename

        Args:
            data_stats_filename: filename of datastats
        """
        self.data_stats_filename = data_stats_filename

    def get_data_stats(self):
        """Get the filename of the data stats"""
        return self.data_stats_filename

    def set_data_src(self, data_src_cfg_name):
        """
        Set the data source filename

        Args:
            data_src_cfg_name: filename of data_source file
        """
        self.data_src_cfg_name = data_src_cfg_name

    def get_data_src(self):
        """Get the data source filename"""
        return self.data_src_cfg_name

    def set_mlflow_tracking_uri(self, mlflow_tracking_uri):
        """
        Set the tracking URI for MLflow server

        Args:
            mlflow_tracking_uri: a tracking URI for MLflow server which could be local directory or address of
                the remote tracking Server; MLflow runs will be recorded locally in algorithms' model folder if
                the value is None.
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri

    def set_mlflow_experiment_name(self, mlflow_experiment_name):
        """
        Set the experiment name for MLflow server

        Args:
            mlflow_experiment_name: a string to specify the experiment name for MLflow server.
        """
        self.mlflow_experiment_name = mlflow_experiment_name

    def get_mlflow_tracking_uri(self):
        """Get the tracking URI for MLflow server"""
        return self.mlflow_tracking_uri

    def get_mlflow_experiment_name(self):
        """Get the experiment name for MLflow server"""
        return self.mlflow_experiment_name

    def get_history(self) -> list:
        """Get the history of the bundleAlgo object with their names/identifiers"""
        return self.history

    def generate(
        self,
        output_folder: str = ".",
        num_fold: int = 5,
        gpu_customization: bool = False,
        gpu_customization_specs: dict[str, Any] | None = None,
        allow_skip: bool = True,
    ) -> None:
        """
        Generate the bundle scripts/configs for each bundleAlgo

        Args:
            output_folder: the output folder to save each algorithm.
            num_fold: the number of cross validation fold.
            gpu_customization: the switch to determine automatically customize/optimize bundle script/config
                parameters for each bundleAlgo based on gpus. Custom parameters are obtained through dummy
                training to simulate the actual model training process and hyperparameter optimization (HPO)
                experiments.
            gpu_customization_specs: the dictionary to enable users overwrite the HPO settings. user can
                overwrite part of variables as follows or all of them. The structure is as follows.
            allow_skip: a switch to determine if some Algo in the default templates can be skipped based on the
                analysis on the dataset from Auto3DSeg DataAnalyzer.

                .. code-block:: python

                    gpu_customization_specs = {
                        'ALGO': {
                            'num_trials': 6,
                            'range_num_images_per_batch': [1, 20],
                            'range_num_sw_batch_size': [1, 20]
                        }
                    }

            ALGO: the name of algorithm. It could be one of algorithm names (e.g., 'dints') or 'universal' which
                would apply changes to all algorithms. Possible options are

                - {``"universal"``, ``"dints"``, ``"segresnet"``, ``"segresnet2d"``, ``"swinunetr"``}.

            num_trials: the number of HPO trials/experiments to run.
            range_num_images_per_batch: the range of number of images per mini-batch.
            range_num_sw_batch_size: the range of batch size in sliding-window inferer.
        """
        fold_idx = list(range(num_fold))
        for algo in self.algos:
            for f_id in ensure_tuple(fold_idx):
                data_stats = self.get_data_stats()
                data_src_cfg = self.get_data_src()
                mlflow_tracking_uri = self.get_mlflow_tracking_uri()
                mlflow_experiment_name = self.get_mlflow_experiment_name()
                gen_algo = deepcopy(algo)
                gen_algo.set_data_stats(data_stats)
                gen_algo.set_data_source(data_src_cfg)
                gen_algo.set_mlflow_tracking_uri(mlflow_tracking_uri)
                gen_algo.set_mlflow_experiment_name(mlflow_experiment_name)
                name = f"{gen_algo.name}_{f_id}"

                if allow_skip:
                    skip_bundlegen, skip_info = gen_algo.pre_check_skip_algo()
                    if skip_bundlegen:
                        logger.info(f"{name} is skipped! {skip_info}")
                        continue

                if gpu_customization:
                    gen_algo.export_to_disk(
                        output_folder,
                        name,
                        fold=f_id,
                        gpu_customization=True,
                        gpu_customization_specs=gpu_customization_specs,
                    )
                else:
                    gen_algo.export_to_disk(output_folder, name, fold=f_id)

                algo_to_pickle(gen_algo, template_path=algo.template_path)
                self.history.append(
                    {AlgoKeys.ID: name, AlgoKeys.ALGO: gen_algo}
                )  # track the previous, may create a persistent history
