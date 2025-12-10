from __future__ import annotations

from pathlib import Path
from typing import List, Union, Optional, Dict, Any, Tuple
import logging
logger = logging.getLogger("koopomics")

from koopomics.utils import torch, pd, np, wandb

# ================================================================
# 🧠 DATA MANAGEMENT MIXIN
# ================================================================
class DataManagementMixin:
    """
    Mixin providing data handling, preprocessing, and dataloader creation
    for OMICS experiments. It interfaces with `DataRegistry` and 
    `DataPreprocessor` for flexible, configuration-driven workflows.
    """

    # ------------------------------------------------------------------
    # ⚙️ Initialization
    # ------------------------------------------------------------------
    def __init__(self):
        """Initialize the mixin with default registry and loader attributes."""
        self._ensure_data_registry()

        # Preprocessing
        self.preprocessor = None
        self.preprocessing_info = None

        # Data loaders
        self.data_loader = None
        self.train_loader = None
        self.test_loader = None

        # Metadata
        self.yaml_path = None


    # ------------------------------------------------------------------
    # 🧩 Core setup utilities
    # ------------------------------------------------------------------
    def _ensure_data_registry(self):
        """Ensure the `DataRegistry` exists, creating it if necessary."""
        from ..data_prep import DataRegistry

        if not hasattr(self, "_data_registry"):
            self._data_registry = DataRegistry()
            logging.info("🗂️ DataRegistry initialized.")

    def _get_time_structure_params(self):
        """Shortcut to get relevant temporal config fields."""
        tr = self.config["training"]
        data_cfg = self.config["data"]
        max_Kstep = getattr(tr, "max_Kstep", 0)
        delay_size = getattr(data_cfg, "delay_size", 0)
        dl_structure = getattr(data_cfg, "dl_structure", "temporal")

        return max_Kstep, delay_size, dl_structure

    # ------------------------------------------------------------------
    # 🧪 Data preprocessing
    # ------------------------------------------------------------------
    def preprocess_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Preprocess raw data without creating loaders.

        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        kwargs : dict
            Arguments for `DataPreprocessor.preprocess_data`.

        Returns
        -------
        pd.DataFrame
            Preprocessed dataframe.
        """
        from ..data_prep.data_prep import DataPreprocessor

        ids = {k: kwargs.pop(k, getattr(self, k, None)) for k in ["time_id", "condition_id", "replicate_id"]}
        self.preprocessor = DataPreprocessor(**ids)
        processed, info = self.preprocessor.preprocess_data(data, **kwargs)

        self.preprocessing_info = info
        return processed


    # ------------------------------------------------------------------
    # 📥 Data loading (main entrypoint)
    # ------------------------------------------------------------------
    def load_data(
        self,
        data: Union[str, Path, pd.DataFrame, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
        yaml_path: Optional[Path] = None,
        registry_dir: Optional[Path] = None,
        feature_list: Optional[List[str]] = None,
        replicate_id: Optional[str] = None,
        time_id: Optional[str] = None,
        condition_id: Optional[str] = None,
        mask_value: Optional[float] = None,
        train_replicates: Optional[List[int]] = None,
        test_replicates: Optional[List[int]] = None,
        split_by_timepoints: bool = False,
        selected_replicates: Optional[List[int]] = None,
        train_timepoints: Optional[List[int]] = None,
        test_timepoints: Optional[List[int]] = None,
    ) -> None:
        """
        Load and prepare OMICS data for model training and testing.

        This function serves as the unified entrypoint for loading data
        either from in-memory structures (DataFrame or Tensor) or from a
        YAML-based configuration and registry directory. It also sets up
        training and testing splits — either across replicates or within
        replicates along the time dimension.

        Parameters
        ----------
        data : str | Path | pd.DataFrame | torch.Tensor | Tuple[torch.Tensor, torch.Tensor], optional
            Input data source. Can be one of:
            - A file path (CSV/TSV/HDF5, depending on implementation)
            - A pandas DataFrame (structured OMICS table)
            - A PyTorch tensor (preprocessed dataset)
            - A tuple of tensors `(train_tensor, test_tensor)` for pre-split data

        yaml_path : Path, optional
            Path to a YAML configuration file that defines dataset parameters
            (e.g., feature names, identifiers, and preprocessing rules).

        registry_dir : Path, optional
            Directory containing auxiliary files (feature registries, metadata, etc.).

        feature_list : list of str, optional
            Features (columns) to extract from the DataFrame. Required if `data` is a DataFrame.

        replicate_id : str, optional
            Column name in the DataFrame identifying biological or experimental replicates.

        time_id : str, optional
            Column name in the DataFrame identifying timepoints.

        condition_id : str, optional
            Column name in the DataFrame identifying treatment or condition groups.

        mask_value : float, optional
            Value to use for missing or masked data during tensor preparation.

        train_replicates : list of int, optional
            Indices of replicates to include in the training set.
            Mutually exclusive with `train_ratio` if present in downstream code.

        test_replicates : list of int, optional
            Indices of replicates to include in the test set.

        split_by_timepoints : bool, default=False
            Whether to split within replicates along the time dimension
            instead of splitting across replicates.

        selected_replicates : list of int, optional
            Replicates to apply the timepoint split on (only used if `split_by_timepoints=True`).

        train_timepoints : list of int, optional
            Indices of timepoints to include in the training segment
            (used when `split_by_timepoints=True`).

        test_timepoints : list of int, optional
            Indices of timepoints to include in the test segment
            (used when `split_by_timepoints=True`).

        Notes
        -----
        - If both `train_replicates`/`test_replicates` and `split_by_timepoints=True`
        are provided, the replicate-level split is applied first.
        - Automatically detects tensor dimensionality and logs data structure.
        - All inputs are converted to tensors before downstream processing.
        """
        from math import ceil
        logger.info("🚀 Starting data loading...")

        self._ensure_data_registry()
        if yaml_path is None and data is None:
            raise ValueError("Either `yaml_path` or `data` must be provided.")

        data_config = self.config.get_data_config()

        # --- Load from registry or dataframe ---
        self._load_from_registry_or_dataframe(
            data, yaml_path, registry_dir, feature_list, replicate_id, time_id, condition_id, mask_value
        )

        # --- Correct feature-dependent dimensions ---
        self._adjust_model_input_dim()

        # --- Correct temporal configuration ---
        self._adjust_temporal_config()

        # --- Create appropriate dataloaders ---
        if isinstance(data, tuple) and all(isinstance(t, torch.Tensor) for t in data):
            self._init_tensor_dataloaders(data)
        else:
            self._init_dataframe_dataloaders(
                data_cfg=data_config,
                split_by_timepoints=split_by_timepoints,

                # ---- split by replicates ----
                train_replicates=train_replicates,
                test_replicates=test_replicates,

                # ----- split_by_timepoints -----
                selected_replicates=selected_replicates,
                train_timepoints=train_timepoints,
                test_timepoints=test_timepoints,
            )
        logger.info("✅ Data loading completed successfully.")


    # ------------------------------------------------------------------
    # 🔧 Helpers for data loading
    # ------------------------------------------------------------------
    def _load_from_registry_or_dataframe(
        self, data, yaml_path, registry_dir, feature_list, replicate_id, time_id, condition_id, mask_value
    ):
        """Load data either from YAML registry or direct DataFrame."""
        if yaml_path:
            self._data_registry.load_from_yaml(yaml_path)
            self.yaml_path = yaml_path
        elif isinstance(data, (pd.DataFrame, str)):
            self._data_registry.validate_direct_input(data, feature_list, replicate_id, time_id, condition_id, mask_value)
            yaml_path = self._data_registry.create_data_input_file(
                input=data,
                feature_list=feature_list,
                replicate_id=replicate_id,
                condition_id=condition_id,
                time_id=time_id,
                mask_value=mask_value,
                output_dir=registry_dir,
            )
            self._data_registry.load_from_yaml(yaml_path)
            self.yaml_path = yaml_path

        for key, value in self._data_registry.get_data_attributes().items():
            setattr(self, key, value)


    def _adjust_model_input_dim(self):
        """
        🔧 Ensure that the encoder input dimension matches the actual number of features.

        This method updates the first entry in `E_layer_dims` to equal the current
        feature count and automatically triggers revalidation of the configuration.
        """
        num_features = len(self.feature_list)

        # Access Pydantic model directly for live update
        model_cfg = self.config.model

        # Update encoder dimensions safely (auto-validates)
        e_dims = list(model_cfg.E_layer_dims)
        if not e_dims:
            raise ValueError("E_layer_dims is empty or not defined in config.")

        e_dims[0] = num_features
        model_cfg.E_layer_dims = e_dims  # ✅ auto-revalidated due to validate_assignment=True

        logging.info(f"📏 Updated input dimension → num_features={num_features}, E_layer_dims={e_dims}")

        # Optionally persist corrected config
        if getattr(self.config, "file_path", None):
            self.config.save(self.config.file_path)

    def _adjust_temporal_config(self):
        """
        ⏱️ Automatically correct `max_Kstep`, `delay_size`, and `dl_structure`
        based on the number of available timepoints and training mode.
        """
        from math import ceil

        num_timepoints = len(self.data[self.time_id].unique())
        tr = self.config.training
        data_cfg = self.config.data

        max_Kstep, delay_size, dl_structure = self._get_time_structure_params()
        max_possible_Kstep = num_timepoints - 2

        # === 🧭 Training mode correction ===
        if tr.training_mode == "embed_tuned_stepwise":
            tr.max_Kstep = max_possible_Kstep
            data_cfg.dl_structure = "temporal"

            logger.info(
                f"🪜 training_mode='embed_tuned_stepwise' → "
                f"dl_structure='temporal', max_Kstep={max_possible_Kstep} "
                f"for {num_timepoints} timepoints."
            )

        # === 🕒 Delay and segment structure corrections ===
        if data_cfg.dl_structure in {"temp_delay", "temp_segm"}:
            if tr.max_Kstep >= max_possible_Kstep:
                logger.warning(
                    f"⚠️ max_Kstep {tr.max_Kstep} too high for {num_timepoints} timepoints → switching to temporal mode."
                )
                tr.max_Kstep = max_possible_Kstep
                data_cfg.dl_structure = "temporal"
                data_cfg.delay_size = 0

            elif data_cfg.dl_structure == "temp_segm":
                seg_size = 3
                limit = ceil(num_timepoints / seg_size) - 1
                if tr.max_Kstep > limit:
                    logger.warning(
                        f"Reducing max_Kstep from {tr.max_Kstep} → {limit} for temp_segm (segment size={seg_size})."
                    )
                    tr.max_Kstep = limit

            elif data_cfg.dl_structure == "temp_delay":
                data_cfg.delay_size = min(delay_size, num_timepoints - 1)
                tr.max_Kstep = num_timepoints - data_cfg.delay_size

        # === 🔄 Re-validate and optionally persist ===
        self.config.correct()  # ensures consistency across model/training/data
        if getattr(self.config, "file_path", None):
            self.config.save(self.config.file_path)

        logger.info(f"🧩 Temporal configuration updated and revalidated for {num_timepoints} timepoints.")



    def _init_tensor_dataloaders(self, data_tuple):
        """Initialize dataloaders directly from pre-split tensors."""
        from torch.utils.data import TensorDataset
        from koopomics.training.data_loader import PermutedDataLoader

        train_tensor, val_tensor = data_tuple
        data_cfg = self.config.get_data_config()

        num_delays = train_tensor.shape[-2]
        self.config["data"]["delay_size"] = num_delays

        train_dataset = TensorDataset(train_tensor)
        val_dataset = TensorDataset(val_tensor)

        self.train_loader = PermutedDataLoader(train_dataset, batch_size=data_cfg["batch_size"], shuffle=False, permute_dims=(1, 0, 2, 3))
        self.test_loader = PermutedDataLoader(val_dataset, batch_size=600, shuffle=False, permute_dims=(1, 0, 2, 3))
        self.data_loader = None

        logging.info(f"📦 Pre-split tensors loaded ({len(self.train_loader)} train, {len(self.test_loader)} val).")


    def _init_dataframe_dataloaders(
        self,
        data_cfg,
        split_by_timepoints: bool,
        selected_replicates: Optional[List[int]] = None,
        train_replicates: Optional[List[int]] = None,
        test_replicates: Optional[List[int]] = None,
        train_timepoints: Optional[List[int]] = None,
        test_timepoints: Optional[List[int]] = None,
    ):
        """
        Initialize OMICS dataloaders from a structured DataFrame.

        This function prepares and instantiates the OmicsDataloader
        depending on whether splitting is performed across replicates
        or within replicates by timepoints.

        Parameters
        ----------
        data_cfg : dict
            Configuration dictionary defining loader parameters
            (e.g., batch size, train ratio, augmentation, etc.).
        split_by_timepoints : bool
            Whether to perform timepoint-level splitting instead of replicate-level.
        selected_replicates : list of int, optional
            Replicates to process (used if split_by_timepoints=True).
        train_replicates : list of int, optional
            Replicates to include in the training set (replicate-level split).
        test_replicates : list of int, optional
            Replicates to include in the test set (replicate-level split).
        train_timepoints : list of int, optional
            Timepoints to include in the training set (timepoint-level split).
        test_timepoints : list of int, optional
            Timepoints to include in the test set (timepoint-level split).
        """
        from ..data_prep import OmicsDataloader


        # --- Core attributes ---
        self.split_by_timepoints = split_by_timepoints
        self.selected_replicates = selected_replicates

        # --- Instantiate dataloader ---
        self.data_loader = OmicsDataloader(
            df=self.data,
            feature_list=self.feature_list,
            replicate_id=self.replicate_id,
            time_id=self.time_id,
            condition_id=self.condition_id,
            batch_size=data_cfg["batch_size"],
            max_Kstep=self.config.training.max_Kstep,
            dl_structure=data_cfg["dl_structure"],
            shuffle=True,
            mask_value=self.mask_value,
            train_ratio=data_cfg["train_ratio"],
            delay_size=data_cfg["delay_size"],
            random_seed=data_cfg["random_seed"],
            concat_delays=data_cfg["concat_delays"],
            augment_by=data_cfg["augment_by"],
            num_augmentations=data_cfg["num_augmentations"],

            # --- Split configuration ---
            split_by_timepoints=self.split_by_timepoints,
            selected_replicates=selected_replicates,
            train_replicates=train_replicates,
            test_replicates=test_replicates,
            train_timepoints=train_timepoints,
            test_timepoints=test_timepoints,
        )

        self.train_loader, self.test_loader = self.data_loader.get_dataloaders()


    # ------------------------------------------------------------------
    # 🔄 Reconfiguration
    # ------------------------------------------------------------------
    def reconfigure_data(self, max_Kstep: Optional[int] = None):
        """
        Rebuild dataloaders dynamically for progressive/stepwise training 
        without reloading from disk. 
        It is mainly used for the training mode: "Embed_Tuned_Stepwise".
        """
        from ..data_prep import OmicsDataloader

        if self.data is None:
            raise ValueError("No data available. Call `load_data()` first.")

        logging.info(f"♻️ Reconfiguring dataloader (max_Kstep={max_Kstep}).")

        data_cfg = self.config.get_data_config()
        self.data_loader = OmicsDataloader(
            df=self.data,
            feature_list=self.feature_list,
            replicate_id=self.replicate_id,
            time_id=self.time_id,
            condition_id=self.condition_id,
            batch_size=data_cfg["batch_size"],
            max_Kstep=max_Kstep,
            dl_structure=data_cfg["dl_structure"],
            shuffle=True,
            mask_value=self.mask_value,
            train_ratio=data_cfg["train_ratio"],
            delay_size=data_cfg["delay_size"],
            random_seed=data_cfg["random_seed"],
            concat_delays=data_cfg["concat_delays"],
            augment_by=data_cfg["augment_by"],
            num_augmentations=data_cfg["num_augmentations"],
            split_by_timepoints=self.split_by_timepoints,
            replicate_idx=self.replicate_idx,
        )

        self.train_loader, self.test_loader = self.data_loader.get_dataloaders()
        logging.info(f"✅ Dataloaders reconfigured: {len(self.train_loader)} train, {len(self.test_loader)} test.")


    # ------------------------------------------------------------------
    # 📤 Data reconstruction
    # ------------------------------------------------------------------
    def get_data_dfs(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Reconstruct full, train, and test DataFrames from loaders."""
        if self.train_loader is None or self.test_loader is None:
            raise ValueError("No data loaded — call `load_data()` first.")

        if self.data_loader:
            train_idx, test_idx = self.data_loader.get_indices()
            return (
                self.data_loader.reconstruct_original_dataframe(),
                self.data_loader.reconstruct_original_dataframe(train_idx),
                self.data_loader.reconstruct_original_dataframe(test_idx),
            )

        logging.warning("⚠️ Using synthetic DataFrame reconstruction (no original data).")
        return self._create_synthetic_dfs()

    def _create_synthetic_dfs(self):
        """Create synthetic dataframes if original data is unavailable."""
        train_size, test_size = len(self.train_loader.dataset), len(self.test_loader.dataset)
        total_size = train_size + test_size
        feature_names = getattr(self, "feature_list", [f"feature_{i}" for i in range(self.train_loader.dataset.tensors[0].shape[-1])])

        df = pd.DataFrame({
            "replicate_id": [f"sample_{i}" for i in range(total_size)],
            "time_id": [0] * total_size,
            "condition_id": ["unknown"] * total_size,
            **{f: [0.0] * total_size for f in feature_names},
        })
        return df, df.iloc[:train_size], df.iloc[train_size:]


    # ------------------------------------------------------------------
    # 🧾 Registry utilities
    # ------------------------------------------------------------------
    def create_data_input_file(self, **kwargs):
        """Wrapper around `DataRegistry.create_data_input_file()`."""
        self._ensure_data_registry()
        return self._data_registry.create_data_input_file(**kwargs)
