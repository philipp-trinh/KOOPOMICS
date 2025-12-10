from __future__ import annotations
import os

from datetime import datetime
import logging
from typing import Dict, List, Union, Optional, Any, Tuple

#from ..model import build_model_from_config, KoopmanModel
from pathlib import Path
from koopomics.utils import torch, pd, np, wandb


# Configure logging
logger = logging.getLogger("koopomics")


# ================================================
# ============= INITIALIZATION MIXIN =============
# ================================================
class InitializationMixin:
    """
    Provides initialization and setup utilities for the Koopman model workflow.

    Handles:
      • Initialization from fresh configurations
      • Loading from saved run directories or ZIP bundles
      • Common component setup and reproducibility
    """

    # ------------------------------------------------------------------
    # 🎯 1. Initialization Entry Points
    # ------------------------------------------------------------------
    def _init_from_config(self, config: Union[Dict, "ConfigManager"]) -> None:
        """
        Initialize a new model instance from a given configuration.

        Parameters
        ----------
        config : dict | ConfigManager
            Configuration dictionary or pre-validated ConfigManager.
        """
        from ..config import ConfigManager

        logger.info("🧩 Initializing new Koopman model from configuration...")
        self.config = ConfigManager(config) if not isinstance(config, ConfigManager) else config


        self.config.save()

        self.build_model()
        logger.info("✅ Model architecture built successfully from configuration.")

    def _init_from_path(self, run_path: str) -> None:
        """
        Initialize model and configuration from a saved run.

        Parameters
        ----------
        run_path : str
            Either a directory or a ZIP bundle containing model artifacts.
        """
        if not os.path.exists(run_path):
            raise FileNotFoundError(f"❌ Provided run path not found: {run_path}")

        if os.path.isdir(run_path):
            logger.info(f"📂 Loading run from directory: {run_path}")
            self._init_from_directory(run_path)
        elif run_path.endswith(".zip"):
            logger.info(f"📦 Loading run from bundle: {run_path}")
            self._init_from_zip(run_path)
        else:
            raise ValueError(f"❌ Unsupported run path format: {run_path}")

    # ------------------------------------------------------------------
    # 🗂️ 2. Directory & ZIP Loading
    # ------------------------------------------------------------------
    def _init_from_directory(self, dir_path: str) -> None:
        """
        Load model configuration and weights directly from a directory.
        Expects at least `config.yaml` and `weights_best.pth` inside.
        """
        from ..config import ConfigManager

        cfg_path = os.path.join(dir_path, "config.yaml")
        weights_path = os.path.join(dir_path, "weights_best.pth")

        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"❌ Missing config.yaml in directory: {dir_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"❌ Missing weights_best.pth in directory: {dir_path}")

        # Load configuration & model weights
        self.config = ConfigManager(cfg_path)
        self.build_model()
        self.load_model(weights_path)

        # Cache resolved paths for potential re-saving or zipping
        self.run_dir = dir_path
        self.bundle_path = None

        logger.info(f"✅ Model and configuration restored successfully from directory: {dir_path}")

    def _init_from_zip(self, zip_path: str) -> None:
        """
        Load model configuration and weights from a ZIP bundle.
        The ZIP is extracted temporarily for loading.
        """
        from zipfile import ZipFile
        import tempfile
        from ..config import ConfigManager

        extract_dir = tempfile.mkdtemp(prefix="koop_run_")
        with ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        cfg_path = os.path.join(extract_dir, "config.yaml")
        weights_path = os.path.join(extract_dir, "weights_best.pth")

        if not os.path.exists(cfg_path):
            raise FileNotFoundError("❌ Missing config.yaml inside ZIP bundle.")
        if not os.path.exists(weights_path):
            raise FileNotFoundError("❌ Missing weights_best.pth inside ZIP bundle.")

        # Load configuration & weights
        self.config = ConfigManager(cfg_path)
        self.build_model()
        self.load_model(weights_path)

        # Track origin & temp extraction dir
        self.bundle_path = zip_path
        self.extracted_dir = extract_dir
        self.run_dir = None

        logger.info(f"✅ Model and configuration restored successfully from ZIP bundle: {zip_path}")

    # ------------------------------------------------------------------
    # ⚙️ 3. Core Component Setup
    # ------------------------------------------------------------------
    def _init_components(self) -> None:
        """
        Initialize core attributes used across all subsystems.
        """
        # Core model and training infrastructure
        self.model = None
        self.data_loader = None
        self.train_loader = None
        self.test_loader = None
        self.trainer = None

        # Evaluation and analysis placeholders
        self.feature_errors = None
        self.embedding_metrics = None
        self.last_evaluation_results = None

        logger.debug("🧩 Core components initialized to None.")

# ===========================================================
# ============= MODEL MANAGEMENT MIXIN ======================
# ===========================================================
class ModelManagementMixin:
    """
    Provides model construction, saving, and loading utilities.

    Responsibilities:
      • Build Koopman model from configuration  
      • Save model weights, config, and logs  
      • Load model from run directory or bundle (.zip)  
      • Bundle and version model checkpoints cleanly
    """

    # ------------------------------------------------------------------
    # 🧱 1. Model Construction
    # ------------------------------------------------------------------
    def build_model(self) -> "KoopmanModel":
        """
        Build a Koopman model instance from current configuration.
        """
        from koopomics.model import build_model_from_config

        if not hasattr(self, "config") or self.config is None:
            raise ValueError("❌ No configuration attached. Initialize ConfigManager first.")

        self.model = build_model_from_config(self.config)
        logger.info(f"✅ Model built successfully: {self.model.__class__.__name__}")
        return self.model

    # ------------------------------------------------------------------
    # 💾 2. Model Saving (with optional zipping & cleanup)
    # ------------------------------------------------------------------
    def save_model(
        self,
        include_config: bool = True,
        include_logs: bool = True,
        zip_bundle: bool = True,
        cleanup_after_bundle: bool = True,
    ) -> str:
        """
        Save the trained model and (optionally) configuration and logs.

        Parameters
        ----------
        include_config : bool, default=True
            Whether to include the KOOPConfig YAML in the saved folder.
        include_logs : bool, default=True
            Whether to include the logs directory in the bundle.
        zip_bundle : bool, default=True
            If True, compresses the run folder into a .zip archive.
        cleanup_after_bundle : bool, default=True
            If True, removes the original model directory (`base_dir`)
            after the zip archive was successfully created.

        Returns
        -------
        str
            Path to the saved model weights or final zip archive.
        """
        import os, shutil, torch

        if not hasattr(self, "config") or self.config is None:
            raise ValueError("❌ Cannot save model — no config attached.")

        paths = getattr(self.config, "paths", None)
        if not paths:
            raise ValueError("❌ Config has no valid paths attached (use KOOPConfig.attach_run_id()).")

        # ✅ Ensure target directory exists
        os.makedirs(paths.base_dir, exist_ok=True)

        # 💾 Save model weights
        torch.save(self.model.state_dict(), paths.model_weights)
        logger.info(f"💾 Model weights saved → {paths.model_weights}")

        # 🗂️ Save enriched configuration
        if include_config:
            self.config.save(paths.config_file)
            logger.info(f"🗂️ Config saved → {paths.config_file}")

        # 📦 Create zip bundle if requested
        if zip_bundle:
            try:
                zip_path = self._bundle_run(paths, include_config, include_logs)
                logger.info(f"📦 Run successfully bundled → {zip_path}")

                # 🧹 Optionally clean up original folder after successful bundling
                if cleanup_after_bundle and os.path.exists(paths.base_dir):
                    shutil.rmtree(paths.base_dir, ignore_errors=True)
                    logger.info(f"🧹 Cleaned up run folder → {paths.base_dir}")

                return zip_path

            except Exception as e:
                logger.warning(f"⚠️ Bundling failed, keeping original folder. Error: {e}")
                return paths.model_weights

        # Return path to raw weights if no zipping requested
        return paths.model_weights


    # ------------------------------------------------------------------
    # 📦 3. Bundle Helper
    # ------------------------------------------------------------------
    def _bundle_run(self, paths, include_config: bool, include_logs: bool) -> str:
        """
        Create a zip archive with model weights, config, and logs.
        """
        import zipfile
        
        zip_path = paths.bundle_file
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            if os.path.exists(paths.model_weights):
                zf.write(paths.model_weights, os.path.basename(paths.model_weights))
            if include_config and os.path.exists(paths.config_file):
                zf.write(paths.config_file, os.path.basename(paths.config_file))

            if include_logs and os.path.isdir(paths.logs_file):
                for root, _, files in os.walk(paths.logs_file):
                    for f in files:
                        full_path = os.path.join(root, f)
                        arc_name = os.path.relpath(full_path, os.path.dirname(paths.logs_file))
                        zf.write(full_path, arc_name)
        return zip_path

    # ------------------------------------------------------------------
    # 🔁 4. Model Loading
    # ------------------------------------------------------------------
    def load_model(self, source: Optional[str] = None) -> None:
        """
        Load a model checkpoint and (optionally) its configuration.

        Parameters
        ----------
        source : str, optional
            Path to a directory or zip file containing model weights.
        """
        if source is None and hasattr(self, "config"):
            paths = getattr(self.config, "paths", None)
            if not paths:
                raise ValueError("❌ No source provided and config has no paths.")
            source = paths.model_weights

        if not source:
            raise ValueError("❌ No model source path provided.")

        src_path = Path(source)
        if not src_path.exists():
            raise FileNotFoundError(f"❌ Model source not found: {source}")

        # --- Handle .zip bundles ---
        if src_path.suffix == ".zip":
            self._load_from_bundle(src_path)
            return

        # --- Handle directory with config + weights ---
        if src_path.is_dir():
            self._load_from_directory(src_path)
            return

        # --- Direct .pth file case ---
        if src_path.suffix == ".pth":
            if not hasattr(self, "model") or self.model is None:
                logger.info("🏗️ No model instance found — rebuilding architecture first...")
                self.build_model()
            state = torch.load(str(src_path), map_location="cpu")
            self.model.load_state_dict(state)
            logger.info(f"✅ Model weights loaded → {src_path}")
            return

        raise ValueError(f"❌ Unsupported source type: {source}")

    # ------------------------------------------------------------------
    # 📦 Load From Bundle or Directory
    # ------------------------------------------------------------------
    def _load_from_bundle(self, bundle_path: Path) -> None:
        """Extract and load model + config from a .zip bundle."""
        import tempfile
        from koopomics.config import ConfigManager

        logger.info(f"📦 Loading model bundle from {bundle_path}")
        with zipfile.ZipFile(bundle_path, "r") as zf, tempfile.TemporaryDirectory() as tmpdir:
            zf.extractall(tmpdir)
            tmpdir = Path(tmpdir)

            weights = tmpdir / "weights.pth"
            config_file = next((tmpdir / f for f in ["config.yaml", "config.json"] if (tmpdir / f).exists()), None)

            if not weights.exists():
                raise FileNotFoundError("❌ Missing weights.pth inside bundle.")
            if config_file:
                self.config = ConfigManager(str(config_file))
                logger.info(f"🗂️ Config loaded from bundle → {config_file}")

            if not hasattr(self, "model") or self.model is None:
                self.build_model()

            self.model.load_state_dict(torch.load(weights, map_location="cpu"))
            logger.info("✅ Model weights loaded successfully from bundle.")

    def _load_from_directory(self, dir_path: Path) -> None:
        """Load model + config from a run directory."""
        from koopomics.config import ConfigManager

        logger.info(f"📂 Loading model directory → {dir_path}")
        weights = dir_path / "weights.pth"
        config_file = dir_path / "config.yaml"

        if not weights.exists() or not config_file.exists():
            raise FileNotFoundError("❌ Missing weights.pth or config.yaml in directory.")

        self.config = ConfigManager(str(config_file))
        if not hasattr(self, "model") or self.model is None:
            self.build_model()

        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        logger.info("✅ Model + config loaded successfully from directory.")

# ==========================================
# ============= TRAINING MIXIN =============
# ==========================================
class TrainingMixin:
    """
    Provides training utilities for the Koopman model.

    Includes:
      • Standard and stepwise/progressive training
      • W&B integration
      • SLURM job submission
      • Sweep (hyperparameter optimization) setup
    """

    # ------------------------------------------------------------------
    # 🚀 Main Training Routine
    # ------------------------------------------------------------------
    def train(
        self,
        data: Optional[Union["pd.DataFrame", "torch.Tensor"]] = None,
        feature_list: Optional[List[str]] = None,
        replicate_id: Optional[str] = None,
        save: bool = True,
    ) -> float:
        """
        🚀 Train the Koopman model using the configured mode (full, embed, tuned, stepwise, etc.).

        This high-level routine handles:
        - Dataset loading (if provided)
        - Dynamic dataloader reconfiguration for stepwise modes
        - Trainer creation via `create_trainer()`
        - Model training, checkpoint loading, and summary logging
        - Optional saving of the final model checkpoint

        Parameters
        ----------
        data : pd.DataFrame | torch.Tensor, optional
            Input dataset. If provided, replaces any previously loaded data.
        feature_list : list[str], optional
            Feature names (required if `data` is a DataFrame).
        replicate_id : str, optional
            Column name for replicate IDs (required if `data` is a DataFrame).
        save : bool, default=True
            If True, automatically saves the final trained model after training.

        Returns
        -------
        tuple(float, float, float)
            Best validation metrics: (baseline_ratio, fwd_loss, bwd_loss)
        """
        from koopomics.training import create_trainer

        # --------------------------------------------------------------
        # 1️⃣ Prepare dataset
        # --------------------------------------------------------------
        if data is not None:
            logger.info("📦 Loading new dataset...")
            self.load_data(data, feature_list, replicate_id)

        if not (self.train_loader and self.test_loader):
            raise ValueError("❌ No dataloaders available — call `load_data()` or provide `data`.")

        # --------------------------------------------------------------
        # 2️⃣ Build model (if not yet built)
        # --------------------------------------------------------------
        if self.model is None:
            logger.info("🏗️ Building Koopman model...")
            self.build_model()

        logger.info(f"🧠 Training mode → {self.config.training.training_mode}")
        logger.info(f"💾 Checkpoints → {self.config.paths.model_weights}")

        # --------------------------------------------------------------
        # 3️⃣ Optional stepwise dataloader reconfiguration
        # --------------------------------------------------------------
        data_loader_fn = None
        if "stepwise" in self.config.training.training_mode.lower():
            logger.info("🔁 Stepwise training mode detected — enabling dynamic dataloader rebuilds.")

            def data_loader_fn(max_Kstep: int):
                logger.info(f"⚙️ Reconfiguring dataloaders for K-step = {max_Kstep}")
                self.reconfigure_data(max_Kstep=max_Kstep)
                return self.train_loader, self.test_loader

        # --------------------------------------------------------------
        # 4️⃣ Create trainer
        # --------------------------------------------------------------
        self.trainer = create_trainer(
            model=self.model,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            config=self.config,
            data_loader_fn=data_loader_fn,
        )

        # --------------------------------------------------------------
        # 5️⃣ Execute training
        # --------------------------------------------------------------
        logger.info("🚀 Starting training...")
        best_metrics = self.trainer.train()
        logger.info("✅ Training completed successfully.")
        logger.info(f"🏆 Best validation metrics: {best_metrics}")

        # --------------------------------------------------------------
        # 6️⃣ Optionally save final model
        # --------------------------------------------------------------
        if save:
            self.save_model()
            logger.info("💾 Final model saved successfully.")
        else:
            logger.info("⚙️ Skipped final model saving (save=False).")

        return best_metrics



    # ------------------------------------------------------------------
    # 🧑‍💻 2. SLURM Job Submission
    # ------------------------------------------------------------------
    def submit_train(
        self,
        yaml_path,
        train_idx,
        test_idx,
        job_name: str = "koopman_train",
        model_dict_save_dir: Optional[str] = None,
        use_wandb: bool = False,
        slurm_mem: str = "8G",
        slurm_cpus: int = 4,
        slurm_time: str = "4:00:00",
    ) -> "submitit.Job":
        """
        Submit a Koopman training job via SLURM using Submitit.

        Parameters
        ----------
        yaml_path : str
            Path to the YAML file containing dataset configuration.
        train_idx : list[int]
            Indices for training samples.
        test_idx : list[int]
            Indices for test samples.
        job_name : str, default='koopman_train'
            Job name on SLURM.
        model_dict_save_dir : str, optional
            Directory to save trained models.
        use_wandb : bool, default=False
            Whether to enable Weights & Biases logging.
        slurm_mem : str, default='8G'
            Memory allocation for the job.
        slurm_cpus : int, default=4
            Number of CPU cores.
        slurm_time : str, default='4:00:00'
            Time limit for the job.

        Returns
        -------
        submitit.Job
            The SLURM job handle.
        """
        if not hasattr(self.__class__, "_submitit"):
            try:
                import submitit
                self.__class__._submitit = submitit
            except ImportError:
                raise ImportError(
                    "❌ submitit is required for job submission. Install it via `pip install submitit`."
                )

        submitit = self.__class__._submitit

        # Setup executor
        executor = submitit.AutoExecutor(folder="training_logs")
        executor.update_parameters(
            name=job_name,
            slurm_mem=slurm_mem,
            slurm_cpus_per_task=slurm_cpus,
            slurm_time=slurm_time,
            slurm_gres="",
        )

        # Prepare save directory
        if model_dict_save_dir is None:
            model_dict_save_dir = "trained_models"
        os.makedirs(model_dict_save_dir, exist_ok=True)

        # Submit lightweight job (model will be built inside job)
        job = executor.submit(
            self._run_training,
            config=self.config,
            yaml_path=yaml_path,
            train_idx=train_idx,
            test_idx=test_idx,
            model_dict_save_dir=model_dict_save_dir,
            use_wandb=use_wandb,
        )

        return job, model_dict_save_dir

    @staticmethod
    def _run_training(
        config,
        yaml_path,
        train_idx,
        test_idx,
        model_dict_save_dir: str,
        use_wandb: bool,
    ) -> float:
        """
        Internal method executed inside a SLURM job.
        Handles CPU-safe Koopman training workflow.
        """
        import os
        import torch
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        torch.backends.cudnn.enabled = False
        config.device = "cpu"

        from koopomics import KOOP

        current_model = KOOP(config)
        current_model.model = current_model.model.to("cpu")

        # Load data
        current_model.load_data(
            yaml_path=yaml_path,
            train_idx=train_idx,
            test_idx=test_idx,
        )

        # Train
        current_model.train(
            data=None,
            feature_list=None,
            replicate_id=None,
            use_wandb=use_wandb,
            model_dict_save_dir=model_dict_save_dir,
        )

        run_id = current_model.trainer.wandb_manager.run_id
        return run_id

    # ------------------------------------------------------------------
    # 🎯 3. Hyperparameter Sweeps (W&B)
    # ------------------------------------------------------------------
    def sweep_params(
        self,
        project_name: str,
        entity: Optional[str] = None,
        CV_save_dir: Optional[str] = None,
        sweep_method: str = "bayes",
    ) -> Union['GridSweepManager', 'BayesSweepManager']:
        """
        Create a SweepManager for hyperparameter tuning using W&B.

        Parameters
        ----------
        project_name : str
            Name of the W&B project.
        entity : str, optional
            W&B team or organization name.
        CV_save_dir : str, optional
            Directory for saving cross-validation results.
        sweep_method : str, default='bayes'
            Sweep type ('grid' or 'bayes').

        Returns
        -------
        SweepManager
            Initialized sweep manager for hyperparameter search.
        """
        from ..wandb_utils import GridSweepManager, BayesSweepManager

        logger.info(f"🧪 Creating Koopman SweepManager (method: {sweep_method})")

        sweep_args = {
            "project_name": project_name,
            "entity": entity,
            "CV_save_dir": CV_save_dir,
        }

        # Add data-related parameters
        if self.train_loader is not None and self.test_loader is not None:
            sweep_args.update({
                "data": self.data,
                "condition_id": self.condition_id,
                "time_id": self.time_id,
                "replicate_id": self.replicate_id,
                "feature_list": self.feature_list,
                "mask_value": self.mask_value,
                "parent_yaml": self.yaml_path,
            })

        # Instantiate sweep manager
        if sweep_method == "grid":
            sweep = GridSweepManager(**sweep_args)
        elif sweep_method == "bayes":
            sweep = BayesSweepManager(**sweep_args)
        else:
            raise ValueError(
                f"❌ Unknown sweep method: {sweep_method}. Use 'grid' or 'bayes'."
            )

        logger.info(f"✅ {type(sweep).__name__} created successfully.")
        return sweep

# =========================================================
# ============= PREDICTION & EVALUATION MIXIN =============
# =========================================================
class PredictionEvaluationMixin:
    """
    Provides prediction and evaluation utilities for trained Koopman models.

    Includes:
      • Forward/backward trajectory prediction
      • Baseline comparison and performance metrics
      • Feature-wise error breakdown
      • Embedding evaluation and visualization support
    """

    # ------------------------------------------------------------------
    # 🔮 1. Prediction Interface
    # ------------------------------------------------------------------
    def predict(
        self,
        data: Union['pd.DataFrame', 'torch.Tensor'],
        feature_list: Optional[List[str]] = None,
        replicate_id: Optional[str] = None,
        steps_forward: int = 1,
        steps_backward: int = 0,
    ) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """
        Run predictions using the trained Koopman model.

        Parameters
        ----------
        data : pd.DataFrame | torch.Tensor
            Input dataset for prediction.
        feature_list : list[str], optional
            Feature names (required if `data` is a DataFrame).
        replicate_id : str, optional
            Column name containing replicate IDs (required if `data` is a DataFrame).
        steps_forward : int, default=1
            Number of time steps to predict forward.
        steps_backward : int, default=0
            Number of time steps to predict backward.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (backward_predictions, forward_predictions)
        """
        from ..data_prep import OmicsDataloader

        # --- Sanity checks ---
        if self.model is None:
            raise ValueError("❌ Model not trained. Call train() first or load a trained model.")

        if steps_forward < 0 or steps_backward < 0:
            raise ValueError("❌ steps_forward and steps_backward must be non-negative.")
        if steps_forward == 0 and steps_backward == 0:
            raise ValueError("❌ At least one of steps_forward or steps_backward must be positive.")

        logger.info(f"🔮 Predicting with steps_forward={steps_forward}, steps_backward={steps_backward}")

        # --- Convert input to tensor ---
        if isinstance(data, pd.DataFrame):
            if feature_list is None or replicate_id is None:
                raise ValueError("feature_list and replicate_id are required when using DataFrame input.")

            logger.info("🧩 Creating temporary data loader for prediction...")
            temp_loader = OmicsDataloader(
                df=data,
                feature_list=feature_list,
                replicate_id=replicate_id,
                batch_size=1,
                max_Kstep=max(steps_forward, steps_backward),
                dl_structure="random",
                shuffle=False,
                mask_value=self.mask_value,
                train_ratio=0,
                delay_size=self.config.delay_size,
                random_seed=self.config.random_seed,
            )

            data_loader = temp_loader.get_dataloaders()[0]
            for batch in data_loader:
                input_data = batch[0].to(self.config.device)
                break
        else:
            input_data = data.to(self.config.device)

        # --- Run model prediction ---
        logger.info("🚀 Running Koopman model prediction...")
        self.model.eval()
        with torch.no_grad():
            backward_predictions, forward_predictions = self.model.predict(
                input_data, fwd=steps_forward, bwd=steps_backward
            )

        logger.info("✅ Prediction completed successfully.")
        return backward_predictions, forward_predictions

    # ------------------------------------------------------------------
    # 🧪 2. Evaluation Interface
    # ------------------------------------------------------------------
    def evaluate(
        self,
        data: Optional[Union['pd.DataFrame', 'torch.Tensor']] = None,
        feature_list: Optional[List[str]] = None,
        replicate_id: Optional[str] = None,
        compare_to_baseline: bool = True,
        feature_wise: bool = False,
        evaluate_embedding: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on test or custom data.

        Parameters
        ----------
        data : pd.DataFrame | torch.Tensor, optional
            Dataset for evaluation. If None, uses the test loader.
        feature_list : list[str], optional
            Feature names (required if `data` is a DataFrame).
        replicate_id : str, optional
            Column name for replicate IDs (required if `data` is a DataFrame).
        compare_to_baseline : bool, default=True
            Whether to compare model results to a naive baseline predictor.
        feature_wise : bool, default=False
            Compute per-feature prediction errors.
        evaluate_embedding : bool, default=False
            Evaluate embedding quality (identity and orthogonality metrics).

        Returns
        -------
        dict
            Evaluation metrics and optional feature/embedding statistics.
        """
        from ..data_prep import OmicsDataloader
        from ..test.test_utils import Evaluator, NaiveMeanPredictor

        if self.model is None:
            raise ValueError("❌ Model not trained. Call train() first or load a trained model.")

        logger.info("🧮 Starting model evaluation...")

        # --- Load test data ---
        if data is None:
            if self.test_loader is None:
                raise ValueError("❌ No test data loaded. Call load_data() first or provide `data`.")
            logger.info("📊 Using existing test data.")
            test_loader = self.test_loader
        else:
            logger.info("🧩 Creating temporary data loader for custom evaluation...")
            temp_loader = OmicsDataloader(
                df=data,
                feature_list=feature_list,
                replicate_id=replicate_id,
                batch_size=self.config.batch_size,
                max_Kstep=self.config.max_Kstep,
                dl_structure=self.config.dl_structure,
                shuffle=False,
                mask_value=self.mask_value,
                train_ratio=0,
                delay_size=self.config.delay_size,
                random_seed=self.config.random_seed,
            )
            test_loader = temp_loader.get_dataloaders()[0]

        # --- Baseline setup ---
        baseline = None
        if compare_to_baseline:
            logger.info("📈 Creating baseline (NaiveMeanPredictor)...")
            baseline = NaiveMeanPredictor(self.train_loader, mask_value=self.mask_value)

        # --- Evaluator initialization ---
        evaluator = Evaluator(
            self.model,
            self.train_loader,
            test_loader,
            mask_value=self.mask_value,
            max_Kstep=self.config.max_Kstep,
            baseline=baseline,
            model_name=self.model.__class__.__name__,
            criterion=None,
            loss_weights=self.config.loss_weights,
        )

        # --- Run evaluation ---
        logger.info("⚙️ Running evaluator...")
        _, test_metrics, baseline_metrics = evaluator()
        result_metrics = {**test_metrics}

        # --- Compute baseline ratio if available ---
        if compare_to_baseline and baseline_metrics:
            result_metrics.update({f"baseline_{k}": v for k, v in baseline_metrics.items()})
            combined_test_loss = (test_metrics["forward_loss"] + test_metrics["backward_loss"]) / 2
            combined_base_loss = (baseline_metrics["forward_loss"] + baseline_metrics["backward_loss"]) / 2
            baseline_ratio = (combined_base_loss - combined_test_loss) / combined_base_loss
            result_metrics["baseline_ratio"] = baseline_ratio
            logger.info(f"✅ Evaluation completed (baseline ratio={baseline_ratio:.6f})")
        else:
            logger.info("✅ Evaluation completed (no baseline).")

        # --- Feature-wise prediction errors ---
        if feature_wise:
            logger.info("🔍 Computing feature-wise prediction errors...")
            self.feature_errors = evaluator.compute_prediction_errors(test_loader)
            result_metrics["feature_errors"] = self.feature_errors

            if len(self.feature_errors["fwd_feature_errors"]) > 0:
                fwd_errors = self.feature_errors["fwd_feature_errors"]
                best = min(fwd_errors, key=fwd_errors.get)
                worst = max(fwd_errors, key=fwd_errors.get)
                result_metrics.update({
                    "best_predicted_feature": best,
                    "best_feature_error": fwd_errors[best],
                    "worst_predicted_feature": worst,
                    "worst_feature_error": fwd_errors[worst],
                })

                # Map feature indices → names
                if hasattr(self, "feature_list"):
                    self.feature_names_to_fwd_errors = {
                        self.feature_list[i]: err
                        for i, err in fwd_errors.items()
                        if i < len(self.feature_list)
                    }
                    if "bwd_feature_errors" in self.feature_errors:
                        bwd_errors = self.feature_errors["bwd_feature_errors"]
                        self.feature_names_to_bwd_errors = {
                            self.feature_list[i]: err
                            for i, err in bwd_errors.items()
                            if i < len(self.feature_list)
                        }

                    # Precompute sorted feature list
                    self.sorted_features_by_fwd_error = sorted(
                        self.feature_names_to_fwd_errors.items(), key=lambda x: x[1]
                    )

        # --- Embedding evaluation ---
        if evaluate_embedding:
            logger.info("🧬 Evaluating embedding quality...")
            self.embedding_metrics, baseline_embedding_metrics = evaluator.metrics_embedding()
            result_metrics["embedding_metrics"] = self.embedding_metrics

            if compare_to_baseline and baseline_embedding_metrics:
                self.baseline_embedding_metrics = baseline_embedding_metrics
                result_metrics["baseline_embedding_metrics"] = baseline_embedding_metrics
                emb_ratio = (
                    (baseline_embedding_metrics["identity_loss"] - self.embedding_metrics["identity_loss"])
                    / baseline_embedding_metrics["identity_loss"]
                )
                result_metrics["embedding_improvement_ratio"] = emb_ratio

        # Store results
        self.last_evaluation_results = result_metrics
        return result_metrics

    # ------------------------------------------------------------------
    # 📊 3. Feature Error Analysis
    # ------------------------------------------------------------------
    def get_feature_errors(
        self,
        direction: str = "forward",
        top_n: Optional[int] = None,
        threshold: Optional[float] = None,
        sort_ascending: bool = True,
    ) -> Dict[str, float]:
        """
        Retrieve and filter feature-level prediction errors.

        Parameters
        ----------
        direction : str, default='forward'
            Direction of prediction error ('forward' or 'backward').
        top_n : int, optional
            Return only the top N features by smallest/largest error.
        threshold : float, optional
            Filter by error threshold (≤ for ascending, ≥ for descending).
        sort_ascending : bool, default=True
            Sort errors in ascending order (best features first).

        Returns
        -------
        dict
            Mapping of feature names → error values.
        """
        if self.feature_errors is None:
            raise ValueError("❌ No feature errors available. Run evaluate() with feature_wise=True first.")

        # --- Select direction ---
        direction = direction.lower()
        if direction == "forward":
            error_dict = getattr(self, "feature_names_to_fwd_errors", None)
            if error_dict is None:
                error_dict = {str(i): e for i, e in self.feature_errors["fwd_feature_errors"].items()}
        elif direction == "backward":
            error_dict = getattr(self, "feature_names_to_bwd_errors", None)
            if error_dict is None:
                error_dict = {str(i): e for i, e in self.feature_errors["bwd_feature_errors"].items()}
        else:
            raise ValueError("Direction must be 'forward' or 'backward'.")

        # --- Sort and filter ---
        sorted_items = sorted(error_dict.items(), key=lambda x: x[1], reverse=not sort_ascending)
        if threshold is not None:
            if sort_ascending:
                sorted_items = [(k, v) for k, v in sorted_items if v <= threshold]
            else:
                sorted_items = [(k, v) for k, v in sorted_items if v >= threshold]
        if top_n is not None:
            sorted_items = sorted_items[:top_n]

        return dict(sorted_items)

        
# ===============================================
# ============= VISUALIZATION MIXIN =============
# ===============================================
class VisualizationMixin:
    """
    Provides visualization tools for analyzing model performance,
    including feature-level prediction error plots.
    """

    def plot_feature_errors(
        self,
        n_features: int = 20,
        direction: str = "forward",
        show_feature_names: bool = True,
    ) -> None:
        """
        Plot the top feature prediction errors as a bar chart.

        Parameters
        ----------
        n_features : int, default=20
            Number of features to display.
        direction : str, default='forward'
            Error direction — one of {'forward', 'backward', 'both'}.
        show_feature_names : bool, default=True
            Whether to display feature names on the x-axis.
        """
        import matplotlib.pyplot as plt

        # --- Validate input ---
        if self.feature_errors is None:
            raise ValueError("❌ No feature errors available. Run evaluate() with feature_wise=True first.")

        direction = direction.lower()
        if direction not in {"forward", "backward", "both"}:
            raise ValueError("❌ 'direction' must be one of: 'forward', 'backward', 'both'.")

        logger.info(f"📊 Plotting {direction} feature prediction errors (top {n_features})...")

        # --- Retrieve errors ---
        fwd_errors = bwd_errors = {}
        if direction in {"forward", "both"}:
            fwd_errors = self.get_feature_errors("forward", top_n=n_features)
        if direction in {"backward", "both"}:
            bwd_errors = self.get_feature_errors("backward", top_n=n_features)

        # --- Initialize figure ---
        plt.figure(figsize=(12, 6))

        # --- Plot both forward and backward errors ---
        if direction == "both":
            indices = range(len(fwd_errors))
            width = 0.35

            plt.bar(
                [i - width / 2 for i in indices],
                list(fwd_errors.values()),
                width=width,
                label="Forward Prediction Error",
            )
            plt.bar(
                [i + width / 2 for i in indices],
                list(bwd_errors.values()),
                width=width,
                label="Backward Prediction Error",
            )

            plt.legend()
            xtick_labels = list(fwd_errors.keys())

        # --- Plot single direction ---
        else:
            errors = fwd_errors if direction == "forward" else bwd_errors
            plt.bar(range(len(errors)), list(errors.values()))
            xtick_labels = list(errors.keys())

        # --- Axis formatting ---
        if show_feature_names:
            plt.xticks(range(len(xtick_labels)), xtick_labels, rotation=90)
        else:
            plt.xticks(range(len(xtick_labels)))

        plt.title(f"{direction.capitalize()} Prediction Errors by Feature", fontsize=14, pad=10)
        plt.xlabel("Features", fontsize=12)
        plt.ylabel("Prediction Error (MSE)", fontsize=12)
        plt.tight_layout()

        # --- Display ---
        plt.show()
        logger.info("✅ Feature error plot displayed successfully.")

# ================================================
# ============= INTERPRETATION MIXIN =============
# ================================================
class InterpretationMixin:
    """
    Provides model interpretation utilities such as:
    - Extracting embeddings
    - Accessing Koopman matrices and eigendecompositions
    - Visualizing system dynamics with KoopmanDynamics
    """

    # -------------------------------------------------------------------------
    # 🧩 Embedding Extraction
    # -------------------------------------------------------------------------
    def get_embeddings(
        self,
        data: Union['pd.DataFrame', 'torch.Tensor'],
        feature_list: Optional[List[str]] = None,
        replicate_id: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Generate embeddings from input data using the trained model.

        Parameters
        ----------
        data : pd.DataFrame | torch.Tensor
            Input data to encode.
        feature_list : list[str], optional
            List of feature names (required if `data` is a DataFrame).
        replicate_id : str, optional
            Column name for replicate IDs (required if `data` is a DataFrame).

        Returns
        -------
        torch.Tensor
            Model embeddings for the given input.
        """
        from ..data_prep import OmicsDataloader
        import torch 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model is None:
            raise ValueError("❌ Model not trained. Call train() first or load a trained model.")

        logger.info("🔍 Generating embeddings...")

        # --- Convert data to tensor if needed ---
        if isinstance(data, pd.DataFrame):
            if feature_list is None or replicate_id is None:
                raise ValueError("feature_list and replicate_id are required for DataFrame input.")

            temp_loader = OmicsDataloader(
                df=data,
                feature_list=feature_list,
                replicate_id=replicate_id,
                batch_size=1,
                max_Kstep=1,
                dl_structure="random",
                shuffle=False,
                mask_value=self.config.mask_value,
                train_ratio=0,
                delay_size=self.config.delay_size,
                random_seed=self.config.random_seed,
            )
            data_loader = temp_loader.get_dataloaders()[0]
            input_data = next(iter(data_loader))[0].to(self.device)
        else:
            input_data = data.to(self.device)

        # --- Encode embeddings ---
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.embedding.encode(input_data)

        logger.info("✅ Embeddings generated successfully.")
        return embeddings

    # -------------------------------------------------------------------------
    # 🧮 Koopman Matrices
    # -------------------------------------------------------------------------
    def get_koopman_matrix(self) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Retrieve the Koopman matrix (or matrices) from the trained model.

        Returns
        -------
        np.ndarray | tuple[np.ndarray, np.ndarray]
            Forward Koopman matrix, or (forward, backward) matrices if available.
        """
        if self.model is None:
            raise ValueError("❌ Model not trained. Call train() first or load a trained model.")

        logger.info("🧮 Extracting Koopman matrix...")

        kmatrices = self.model.kmatrix(detach=True)
        if isinstance(kmatrices, tuple):
            logger.info("✅ Forward and backward Koopman matrices extracted.")
        else:
            logger.info("✅ Forward Koopman matrix extracted.")

        return kmatrices

    # -------------------------------------------------------------------------
    # ⚙️ Eigendecomposition
    # -------------------------------------------------------------------------
    def get_eigenvalues(self, plot: bool = False) -> Tuple:
        """
        Compute eigenvalues and eigenvectors of the Koopman operator.

        Parameters
        ----------
        plot : bool, default=False
            Whether to plot the eigenvalue spectrum.

        Returns
        -------
        tuple
            (eigenvalues, eigenvectors)
        """
        if self.model is None:
            raise ValueError("❌ Model not trained. Call train() first or load a trained model.")

        logger.info("🔢 Computing Koopman eigendecomposition...")
        result = self.model.eigen(plot=plot)
        logger.info("✅ Eigendecomposition completed.")
        return result

    # -------------------------------------------------------------------------
    # 🌊 Dynamics Interpretation
    # -------------------------------------------------------------------------
    def get_dynamics(
        self,
        dataset_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> "KoopmanDynamics":
        """
        Create a `KoopmanDynamics` interpreter to analyze and visualize learned dynamics.

        Parameters
        ----------
        dataset_df : pd.DataFrame, optional
            Full dataset to analyze (default: uses loaded data).
        test_df : pd.DataFrame, optional
            Test dataset for evaluation and comparison.

        Returns
        -------
        KoopmanDynamics
            Object providing tools for dynamic mode decomposition,
            trajectory visualization, and interpretability analyses.
        """
        from ..interpret.interpret import KoopmanDynamics

        if self.model is None:
            raise ValueError("❌ Model not trained. Call train() first or load a trained model.")
        if self.data_loader is None and dataset_df is None:
            raise ValueError("❌ No data available. Call load_data() first or provide dataset_df.")

        logger.info("🌊 Creating KoopmanDynamics interpreter...")

        dynamics = KoopmanDynamics(
            model=self.model,
            dataset_df=dataset_df,
            feature_list=self.feature_list,
            replicate_id=self.replicate_id,
            time_id=self.time_id,
            condition_id=self.condition_id,
            mask_value=self.mask_value,
            device=self.config.device,
            test_df=test_df,
        )

        logger.info("✅ KoopmanDynamics interpreter created successfully.")
        return dynamics
