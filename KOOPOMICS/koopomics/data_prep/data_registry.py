import yaml
import json
import datetime
from koopomics.utils import torch, pd, np, wandb
from pathlib import Path
from typing import List, Union, Optional, Dict, Any, Tuple

import logging
# Configure logging
logger = logging.getLogger("koopomics")

class DataRegistry:
    """
    🧬 **DataRegistry — Centralized Manager for OMICS Dataset I/O and Metadata**

    The `DataRegistry` class provides a unified interface for managing OMICS datasets, 
    handling their input/output, validation, and standardized configuration management.
    It serves as the bridge between raw experimental data (e.g. CSVs or DataFrames) 
    and model-ready representations (HDF5 + YAML metadata).

    ---
    ### ⚙️ Core Responsibilities
    - 📦 **Standardization** – Converts raw data into structured formats (HDF5 + YAML)
    - 📂 **Loading** – Reads and reconstructs datasets from YAML/HDF5 configurations
    - 🔢 **Mapping** – Encodes categorical variables (replicates, conditions, times) as integer IDs
    - 🧪 **Validation** – Ensures all required metadata, features, and columns are consistent
    - 🧾 **Metadata Management** – Tracks preprocessing history, mappings, and dimensions

    ---
    ### 🧠 Internal Logic Overview
    The registry ensures reproducibility and interoperability across all analysis stages:
    1. **Input Handling:**  
       Accepts input as a DataFrame, CSV file, or tensor, validating structure and required columns.
    2. **Conversion to HDF5:**  
       Saves standardized datasets to HDF5 with attribute-encoded mappings and metadata.
    3. **YAML Configuration:**  
       Stores references, mappings, and preprocessing metadata in a portable YAML file.
    4. **Loading & Reconstruction:**  
       Rebuilds the original dataset and all relevant mappings for use in dataloaders or analysis.
    5. **Split Handling:**  
       Supports datasets that have been partitioned into train/test splits with inheritance of parent metadata.

    ---
    ### 🧩 Method Reference

    **`__init__()`**  
    Initializes the registry, setting up empty containers for data, mappings, and metadata.

    **`create_data_input_file()`**  
    🏗️ Converts a DataFrame or CSV into an HDF5 + YAML dataset registry.  
    - Applies mapping for replicate/time/condition columns → integer IDs  
    - Auto-detects preprocessing keywords in filenames  
    - Stores dataset attributes, mappings, and optional split metadata  
    - Returns the path to the generated YAML config  

    **`validate_direct_input()`**  
    🔍 Validates that user-provided inputs (data, IDs, mask values, and feature list) are complete and consistent.  
    Logs a concise summary of data type, shape, and identifiers.  
    - Supports DataFrames, Tensors, tuples of tensors, or file paths.  
    - Raises detailed errors for missing or incompatible parameters.

    **`load_from_yaml()`**  
    📥 Loads an existing OMICS dataset configuration from YAML and reconstructs the associated HDF5 data.  
    - Loads numeric + original categorical columns  
    - Recreates mapping dictionaries and metadata  
    - Automatically detects split datasets and stores train/test indices  
    - Returns the fully reconstructed DataFrame  

    **`get_data_attributes()`**  
    📦 Returns all core data-related attributes and mappings as a dictionary.  
    Useful for programmatic inspection, serialization, or debug logging.

    ---
    ### 🧭 Typical Workflow Example

    ```python
    registry = DataRegistry()

    # Step 1️⃣: Create registry files from a DataFrame
    yaml_path = registry.create_data_input_file(
        condition_id="Treatment",
        time_id="Day",
        replicate_id="Plant_ID",
        feature_list=my_features,
        mask_value=-1,
        input=my_dataframe,
        output_dir="./registry_output"
    )

    # Step 2️⃣: Load dataset from YAML for downstream analysis
    df = registry.load_from_yaml(yaml_path)

    # Step 3️⃣: Inspect mappings or attributes
    attrs = registry.get_data_attributes()
    print(attrs["feature_list"])
    ```

    ---
    ### 🧾 Notes
    - The registry ensures **consistent encoding** across all datasets.
    - Supports both **full datasets** and **split configurations** (e.g., train/test).
    - Auto-detection of preprocessing flags (normalized, scaled, etc.) improves traceability.
    - Designed to integrate seamlessly with `OmicsDataloader` and Koopman modeling pipelines.
    """
    
    def __init__(self):
        """Initialize the DataRegistry."""
        # --- Data storage ---
        self.data = None
        self.feature_list = None
        self.replicate_id = None
        self.time_id = None
        self.condition_id = None
        self.mask_value = None
        
        # --- Mappings ---
        self._condition_map = None
        self._time_map = None
        self._replicate_map = None
        self._inv_condition_map = None
        self._inv_time_map = None
        self._inv_replicate_map = None
        
        # --- Metadata ---
        self.preprocessing_info = None
        
    # ------------------------------------------------------------------
    # 🧾 Create YAML + HDF5 input data registry
    # ------------------------------------------------------------------
    def create_data_input_file(
        self,
        condition_id: str,
        time_id: str,
        replicate_id: str,
        feature_list: List[str],
        mask_value: float,
        input: Union[pd.DataFrame, Path, str] = None,
        output_dir: Union[Path, str] = None,
        data_name: Optional[str] = "dataframe_input",
        original_path: Optional[Union[Path, str]] = None,
        is_split: bool = False,
        parent_yaml: Optional[Union[Path, str]] = None,
        split_indices: Optional[Dict[str, List[int]]] = None,
        split_metadata: Optional[Dict] = None,
    ) -> str:
        """
        🧩 Creates a YAML configuration file with HDF5 data storage for OMICS data.
        Supports both full datasets and train/test splits.
        
        Args:
            condition_id: Column name for condition labels.
            time_id: Column name for timepoints.
            replicate_id: Column name for replicate IDs.
            feature_list: List of feature column names.
            mask_value: Value to use for masking missing data.
            input: Input data (DataFrame or file path).
            output_dir: Directory to save outputs.
            data_name: Identifier for DataFrame inputs.
            original_path: Original source path for DataFrame inputs.
            is_split: Flag indicating this is a split dataset.
            parent_yaml: Path to original dataset's YAML config.
            split_indices: {'train': [indices], 'test': [indices]}.
            split_metadata: Additional split info.

        Returns:
            Path to the created YAML file.
        
        Raises:
            ValueError: If input validation fails.
            FileNotFoundError: If input file doesn't exist.
            IOError: If file operations fail.
        """
        import h5py

        try:
            # ============================================================
            # 🧮 Validate input and parameters
            # ============================================================
            if input is None:
                raise ValueError("❌ Input must be provided (DataFrame or file path).")
                
            # Handle split dataset logic
            if is_split:
                if not parent_yaml:
                    raise ValueError("❌ 'parent_yaml' required for split datasets.")
                if not split_indices:
                    raise ValueError("❌ 'split_indices' required for split datasets.")
                parent_yaml = Path(parent_yaml)
                if not parent_yaml.exists():
                    raise FileNotFoundError(f"Parent config not found: {parent_yaml}")

            # ============================================================
            # 📂 Prepare output directory
            # ============================================================
            output_path = Path(output_dir).absolute() if output_dir else Path.cwd() / "input_data_registry"
            output_path.mkdir(parents=True, exist_ok=True)

            # ============================================================
            # 📥 Load and preprocess input data
            # ============================================================
            if isinstance(input, (Path, str)):
                input_path = Path(input).absolute()
                if not input_path.exists():
                    raise FileNotFoundError(f"Input file not found: {input_path}")
                df = pd.read_csv(input_path)
                base_name = input_path.stem
                original_source = str(input_path)
                
                # 🧠 Auto-detect preprocessing from filename
                filename = input_path.stem.lower()
                preprocessing = {
                    "interpolated": "interpolated" in filename,
                    "normalized": "normalized" in filename,
                    "scaled": any(x in filename for x in ["scaled", "zscore"]),
                    "log_transformed": "log" in filename,
                }

            else:
                df = input.copy()
                base_name = data_name
                original_source = original_path if original_path is not None else "Provided DataFrame"
                
                if original_path is not None:
                    filename = Path(original_source).stem.lower()
                    preprocessing = {
                        "interpolated": "interpolated" in filename,
                        "normalized": "normalized" in filename,
                        "scaled": any(x in filename for x in ["scaled", "zscore"]),
                        "log_transformed": "log" in filename,
                    }
                else:
                    preprocessing = {
                        "interpolated": False,
                        "normalized": False,
                        "scaled": False,
                        "log_transformed": False,
                    }
            
            # ============================================================
            # 🧪 Validate data structure and features
            # ============================================================
            required_cols = {condition_id, time_id, replicate_id}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            missing_features = set(feature_list) - set(df.columns)
            if missing_features:
                raise ValueError(f"Features not found in data: {missing_features}")
            
            # Identify start index of feature block
            all_columns = df.columns.tolist()
            try:
                feature_start_col = min(all_columns.index(feat) for feat in feature_list)
            except ValueError:
                raise ValueError("Could not determine feature start column - feature_list items not found in DataFrame.")
            
            # ============================================================
            # 🔢 Create mappings from original values to integer IDs
            # ============================================================
            condition_map = {str(v): int(k) for k, v in enumerate(df[condition_id].astype(str).unique())}
            time_map = {str(v): int(k) for k, v in enumerate(df[time_id].astype(str).unique())}
            replicate_map = {str(v): int(k) for k, v in enumerate(df[replicate_id].astype(str).unique())}

            # Apply mappings to new columns
            df["condition_id"] = df[condition_id].astype(str).map(condition_map).astype(int)
            df["time_id"] = df[time_id].astype(str).map(time_map).astype(int)
            df["replicate_id"] = df[replicate_id].astype(str).map(replicate_map).astype(int)
            
            # ============================================================
            # 💾 Save data to HDF5 file
            # ============================================================
            h5_path = output_path / f"{base_name}.h5"
            yaml_path = output_path / f"{base_name}_config.yaml"
            
            try:
                with h5py.File(h5_path, "w") as hf:
                    # Store numeric feature data
                    hf.create_dataset(
                        "data", data=df[feature_list].values.astype("float32"), compression="gzip"
                    )
                    
                    # Store ID arrays
                    hf.create_dataset("condition_ids", data=df["condition_id"].values)
                    hf.create_dataset("time_ids", data=df["time_id"].values)
                    hf.create_dataset("replicate_ids", data=df["replicate_id"].values)
                    
                    # Store metadata mappings
                    hf.attrs["condition_map"] = json.dumps(condition_map)
                    hf.attrs["time_map"] = json.dumps(time_map)
                    hf.attrs["replicate_map"] = json.dumps(replicate_map)
                    hf.attrs["mask_value"] = float(mask_value)
                    hf.attrs["creation_date"] = datetime.datetime.now().isoformat()
                    
                    # Add split info if applicable
                    if is_split:
                        hf.attrs["is_split"] = True
                        hf.attrs["parent_config"] = str(parent_yaml)
            except Exception as e:
                raise IOError(f"Failed to create HDF5 file: {str(e)}")
            
            # ============================================================
            # 🧾 Build YAML configuration file
            # ============================================================
            config = {
                "data_files": {
                    "hdf5": str(h5_path),
                    "original_source": original_source,
                    "is_split": is_split,
                },
                "notes": None,
                "split_info": None,
                "generated_on": datetime.datetime.now().isoformat(),
                "columns": {
                    "condition": condition_id,
                    "time": time_id,
                    "replicate": replicate_id,
                    "features_start": feature_start_col,
                },
                "preprocessing": preprocessing,
                "metadata": {            
                    "dimensions": {
                        "n_samples": len(df),
                        "n_features": len(feature_list),
                        "n_timepoints": len(time_map),
                        "n_conditions": len(condition_map),
                        "n_replicates": len(replicate_map),
                    },
                    "mappings": {
                        "condition": {str(k): str(v) for v, k in condition_map.items()},
                        "time": {str(k): str(v) for v, k in time_map.items()},
                        "replicate": {str(k): str(v) for v, k in replicate_map.items()},
                    },
                    "mask_value": float(mask_value),
                    "feature_names": list(feature_list),
                },
            }

            # ============================================================
            # ✂️ Handle split dataset metadata
            # ============================================================
            if is_split:
                # Inherit preprocessing from parent config if available
                try:
                    with open(parent_yaml) as f:
                        parent_config = yaml.safe_load(f)
                    config["preprocessing"] = parent_config.get("preprocessing", config["preprocessing"])
                except Exception as e:
                    logger.warning(f"⚠️ Could not read parent config: {str(e)}")
                
                config["split_info"] = {
                    "parent_config": str(parent_yaml),
                    "indices": {k: [int(i) for i in v] for k, v in split_indices.items()},
                    "metadata": split_metadata or {},
                }
                
                # Add descriptive notes
                config["notes"] = f"Split from {parent_yaml} with strategy: {split_metadata.get('strategy', 'unknown')}"
            
            # ============================================================
            # 📝 Save YAML configuration
            # ============================================================
            try:
                with open(yaml_path, "w") as f:
                    yaml.dump(config, f, sort_keys=False, default_flow_style=False)
            except Exception as e:
                raise IOError(f"Failed to create YAML file: {str(e)}")
            
            # ============================================================
            # ✅ Summary logging
            # ============================================================
            logger.info(
                f"\n✅ Successfully created OMICS data registry"
                f"\n   📄 Config: {yaml_path}"
                f"\n   💾 Data:   {h5_path}"
                f"\n   🧬 Samples: {len(df)}, Features: {len(feature_list)}, Replicates: {len(replicate_map)}"
            )
            if is_split:
                logger.info(f"✂️ Split from: {parent_yaml}")
            
            return str(yaml_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to create input files: {str(e)}")
            raise

    def validate_direct_input(
        self,
        data,
        feature_list,
        replicate_id,
        time_id,
        condition_id,
        mask_value
    ) -> None:
        """
        🧪 Validate parameters for direct OMICS data input.

        Ensures that all required parameters are provided and that 
        the data object is of a supported type (DataFrame, Tensor, tuple, or path).

        Args:
            data: Input data (DataFrame, Tensor, tuple of Tensors, or path)
            feature_list: List of feature names (must not be None)
            replicate_id: Column name for replicate identifiers
            time_id: Column name for timepoints
            condition_id: Column name for conditions
            mask_value: Value used to represent masked/missing data

        Raises:
            ValueError: If required parameters are missing or data type is unsupported.
        """
        # ============================================================
        # 🔍 Parameter presence validation
        # ============================================================
        missing = []
        if feature_list is None: missing.append("feature_list")
        if replicate_id is None: missing.append("replicate_id")
        if time_id is None: missing.append("time_id")
        if condition_id is None: missing.append("condition_id")
        if mask_value is None: missing.append("mask_value")

        if missing:
            raise ValueError(
                f"❌ Missing required parameters for data input: {', '.join(missing)}"
            )

        # ============================================================
        # 🧩 Validate data type and provide summary
        # ============================================================
        if isinstance(data, pd.DataFrame):
            logger.info(
                f"\n📊 Data input detected: Pandas DataFrame"
                f"\n   • Shape: {data.shape}"
                f"\n   • Features: {len(feature_list)} ({', '.join(feature_list[:5])}...)"
                f"\n   • Replicate ID: '{replicate_id}'"
                f"\n   • Time ID: '{time_id}'"
                f"\n   • Condition ID: '{condition_id}'"
            )

        elif isinstance(data, torch.Tensor):
            logger.info(
                f"\n🔢 Data input detected: PyTorch Tensor"
                f"\n   • Shape: {tuple(data.shape)}"
            )

        elif isinstance(data, tuple) and all(isinstance(x, torch.Tensor) for x in data):
            logger.info(
                f"\n🧮 Data input detected: Tuple of PyTorch Tensors (train/validation)"
                f"\n   • Train tensor shape: {tuple(data[0].shape)}"
                f"\n   • Validation tensor shape: {tuple(data[1].shape)}"
            )

        elif isinstance(data, (Path, str)):
            logger.info(
                f"\n📂 Data input detected: File path"
                f"\n   • Loaded from: {data}"
            )

        else:
            raise ValueError(
                "❌ Unsupported data type: must be DataFrame, Tensor, tuple of Tensors, or file path (str/Path)."
            )

    def load_from_yaml(self, yaml_path: Path) -> pd.DataFrame:
        """
        📥 Load OMICS dataset from a YAML configuration file and corresponding HDF5 data.

        Loads both integer-encoded identifiers (for computation) and original values (for interpretation),
        setting class attributes such as mappings, preprocessing info, and feature metadata.

        Args:
            yaml_path: Path to the YAML configuration file.
                
        Returns:
            pd.DataFrame: DataFrame containing both integer IDs and original values.
                
        Raises:
            FileNotFoundError: If YAML or HDF5 file doesn't exist.
            ValueError: If required sections or data are missing.
            IOError: If there are problems reading files.
        """
        import h5py

        try:
            # ============================================================
            # 🧾 Load and validate YAML configuration
            # ============================================================
            with open(yaml_path, "r") as f:
                config = yaml.safe_load(f)

            required_sections = ["data_files", "columns", "metadata"]
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"❌ Missing required section in YAML: {section}")

            # Extract paths
            hdf5_path = Path(config["data_files"]["hdf5"])
            if not hdf5_path.exists():
                raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

            # ============================================================
            # 💾 Load data from HDF5
            # ============================================================
            with h5py.File(hdf5_path, "r") as hf:
                # --- Validate presence of datasets and attributes ---
                required_datasets = ["data", "condition_ids", "time_ids", "replicate_ids"]
                required_attrs = ["condition_map", "time_map", "replicate_map"]

                for ds in required_datasets:
                    if ds not in hf:
                        raise ValueError(f"Missing required dataset in HDF5: {ds}")
                for attr in required_attrs:
                    if attr not in hf.attrs:
                        raise ValueError(f"Missing required attribute in HDF5: {attr}")

                # --- Load mappings ---
                condition_map = json.loads(hf.attrs["condition_map"])
                time_map = json.loads(hf.attrs["time_map"])
                replicate_map = json.loads(hf.attrs["replicate_map"])

                # --- Reverse mappings ---
                inv_condition_map = {v: k for k, v in condition_map.items()}
                inv_time_map = {v: k for k, v in time_map.items()}
                inv_replicate_map = {v: k for k, v in replicate_map.items()}

                # --- Create DataFrame with feature data ---
                data = pd.DataFrame(
                    data=hf["data"][:],
                    columns=config["metadata"]["feature_names"],
                )

                # --- Add integer ID columns ---
                data["condition_id"] = hf["condition_ids"][:]
                data["time_id"] = hf["time_ids"][:]
                data["replicate_id"] = hf["replicate_ids"][:]

                # --- Add original value columns using reverse maps ---
                data[config["columns"]["condition"]] = data["condition_id"].map(inv_condition_map)
                data[config["columns"]["time"]] = data["time_id"].map(inv_time_map)
                data[config["columns"]["replicate"]] = data["replicate_id"].map(inv_replicate_map)

            # ============================================================
            # 🧱 Store attributes in the DataRegistry instance
            # ============================================================
            self.data = data
            self.feature_list = config["metadata"]["feature_names"]
            self.replicate_id = config["columns"]["replicate"]
            self.time_id = config["columns"]["time"]
            self.condition_id = config["columns"]["condition"]

            # --- Store forward and inverse mappings ---
            self._condition_map = condition_map
            self._time_map = time_map
            self._replicate_map = replicate_map
            self._inv_condition_map = inv_condition_map
            self._inv_time_map = inv_time_map
            self._inv_replicate_map = inv_replicate_map

            # --- Store metadata ---
            self.mask_value = config["metadata"].get("mask_value", None)
            self.preprocessing_info = config.get("preprocessing", {})

            # ============================================================
            # ✂️ Handle split dataset information (if applicable)
            # ============================================================
            self.is_split = config["data_files"]["is_split"]

            if self.is_split:
                self.train_indices = config["split_info"]["indices"]["train"]
                self.test_indices = config["split_info"]["indices"]["test"]

            # ============================================================
            # ✅ Logging summary
            # ============================================================
            logger.info(
                f"\n✅ Successfully loaded OMICS dataset"
                f"\n   📄 YAML: {yaml_path}"
                f"\n   💾 HDF5: {hdf5_path}"
                f"\n   📊 Data shape: {data.shape}"
                f"\n   🧬 Features: {len(self.feature_list)}"
                f"\n   🧫 Replicates: {len(replicate_map)}, Timepoints: {len(time_map)}, Conditions: {len(condition_map)}"
                f"\n   ✂️ Split: {self.is_split}"
            )

            if self.is_split:
                logger.info(
                    f"   🔹 Train indices: {len(self.train_indices)} | 🔹 Test indices: {len(self.test_indices)}"
                )

            return data

        # ============================================================
        # ⚠️ Error handling
        # ============================================================
        except yaml.YAMLError as e:
            logger.error(f"❌ Error parsing YAML file: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error decoding JSON mappings in HDF5: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load data from YAML: {str(e)}")
            raise
            
    def get_data_attributes(self) -> Dict[str, Any]:
        """
        📦 Retrieve all core attributes of the loaded OMICS dataset.

        This method returns a comprehensive dictionary of stored data elements,
        including mappings, metadata, preprocessing information, and raw data.

        Returns:
            dict: A dictionary containing the dataset’s core attributes.
        """
        return {
            # --- 🔬 Core Data ---
            "data": self.data,
            "feature_list": self.feature_list,

            # --- 🧫 Identifiers ---
            "replicate_id": self.replicate_id,
            "time_id": self.time_id,
            "condition_id": self.condition_id,

            # --- 🩸 Masking ---
            "mask_value": self.mask_value,

            # --- 🔁 Mappings (encoded → integer) ---
            "condition_map": self._condition_map,
            "time_map": self._time_map,
            "replicate_map": self._replicate_map,

            # --- 🔄 Inverse Mappings (integer → original) ---
            "inv_condition_map": self._inv_condition_map,
            "inv_time_map": self._inv_time_map,
            "inv_replicate_map": self._inv_replicate_map,

            # --- 🧪 Preprocessing Info ---
            "preprocessing_info": self.preprocessing_info,
        }

 