# IVUS–EM Segmentation & Reconstruction

Offline reconstruction software accompanying:

> **Learned ultrasound segmentation and deformable CT fusion for augmented reality endovascular surgery**  
> Dillon et al.

This repository performs learned intravascular ultrasound (IVUS) segmentation and three-dimensional reconstruction by combining IVUS images with synchronized electromagnetic (EM) tracking transforms. The command-line workflow can automatically download the associated datasets from Zenodo, run segmentation and TSDF fusion, display the reconstruction, and save reviewer-facing outputs.

![Example IVUS–EM reconstruction](images/setup.png)

## Repository contents

The files have the following roles:

- `run_mapping.py` — command-line entry point, dataset download, replay, reconstruction, visualization, and output export.
- `segmentation_helpers_runtime.py` — DeepLumen model definition, inference, and segmentation post-processing.
- `reconstruction_helpers_runtime.py` — reconstruction, point-cloud, TSDF, transform, and visualization utilities used at runtime.
- `aortascope_mapping_params.yaml` — mapping and model configuration.
- `calibration_parameters_ivus.yaml` — Dataset-specific calibration is loaded during replay.
- `requirements.txt` — pinned Python dependencies.

## Data

The associated IVUS–EM datasets are available from:

https://zenodo.org/records/20737792

The following named datasets are generated from imaging of the 7 in-vitro aortic models developed from the [aortic-vessel-tree (AVT) dataset](https://figshare.com/articles/dataset/Aortic_Vessel_Tree_AVT_CTA_Datasets_and_Segmentations/14806362), as well as 3 in-vivo ovine studies:

```text
patient_1
patient_2
patient_3
patient_4
patient_5
patient_6
patient_7
sheep_1
sheep_2
sheep_3
```

When a named dataset is requested and is not already present, `run_mapping.py` downloads the corresponding ZIP archive and extracts it automatically.


## System requirements

The software was tested in the following system requirements:

- Ubuntu/Linux
- Python 3.9
- A graphical desktop session capable of displaying Open3D and OpenCV windows
- A C++ compiler and CMake for building the Voxblox Python bindings
- At least ~500 MB of open disk space for the selected Zenodo dataset and generated outputs

A CUDA-compatible GPU is optional for TensorFlow inference. TensorFlow uses an available compatible GPU when one is visible to the installed TensorFlow build; otherwise it runs supported operations on the CPU. CPU execution is expected to be slower.

## Python environment

Create and activate a Python 3.9 virtual environment:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the pinned Python packages:

```bash
python -m pip install -r requirements.txt
```

The tested requirements are:

```text
numpy==1.26.4
open3d==0.18.0
PyYAML==6.0.1
opencv-python==4.9.0.80
tensorflow==2.16.1
scipy==1.13.1
```

The remaining imports used by the code are part of the Python standard library and do not require separate installation.

## Voxblox Python bindings

The reconstruction requires the Python bindings from:

https://github.com/PRBonn/voxblox_pybind

These bindings provide:

```python
from voxblox import FastTsdfIntegrator
```

Install the required Ubuntu build dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libprotobuf-dev \
    protobuf-compiler \
    python3-dev \
    python3-pip \
    git
```

Clone the repository with its submodules and install it while the project virtual environment is active:

```bash
git clone --recurse-submodules https://github.com/PRBonn/voxblox_pybind.git
cd voxblox_pybind
make install
cd ..
```

Verify that the binding is available to the same Python interpreter used for this project:

```bash
python -c "from voxblox import FastTsdfIntegrator; print('Voxblox import successful')"
python -m pip show voxblox
```

Then check out that commit before installation:

```bash
git checkout <COMMIT_HASH>
git submodule update --init --recursive
make install
```

## Model weights

The trained DeepLumen model weights are required but are not downloaded from Zenodo by the current script.

Provide the model in either of two ways:

1. Set `model_path` in `aortascope_mapping_params.yaml`.
2. Supply the weights on the command line with `--model`.

For example:

```bash
python run_mapping.py \
    --dataset-name patient_1 \
    --model /absolute/path/to/model.weights.h5
```

The command-line argument takes precedence over `model_path` in the YAML configuration.

Before sharing the repository, make sure the weights are available to reviewers through the private review package, a stable repository, or a permanent data archive. Do not commit large model files directly to ordinary Git history when they exceed GitHub's file-size limits.


## Quick start

From the repository root:

```bash
source .venv/bin/activate
python run_mapping.py --dataset-name patient_1
```

The short form is equivalent:

```bash
python run_mapping.py -d patient_1
```

This command:

1. checks for `data/patient_1`;
2. downloads the archive from Zenodo if the dataset is absent;
3. verifies the archive checksum;
4. extracts and resolves the replay directory;
5. loads the model and calibration;
6. runs DeepLumen segmentation and IVUS–EM reconstruction;
7. displays the segmentation and 3D reconstruction;
8. saves the reconstruction outputs.

## Command-line options

Show all options with:

```bash
python run_mapping.py --help
```

### Run a named Zenodo dataset

```bash
python run_mapping.py --dataset-name patient_1
```

### Use an already extracted dataset

```bash
python run_mapping.py \
    --dataset-path /absolute/path/to/extracted/dataset
```

The supplied directory, or one of its subdirectories, must contain an accepted image folder and an accepted EM-transform folder.


## Configuration

The primary mapping parameters are stored in:

```text
aortascope_mapping_params.yaml
```

Important parameters include:

- `deeplumen_on` should enable the released DeepLumen model.
- `tsdf_map` should enable TSDF reconstruction.
- `model_path` must point to the released model weights unless `--model` is supplied.
- `voxel_size` TSDF voxel value resolution

The released YAML file should contain the exact settings used for the manuscript results.

## Using a local dataset

User datasets may also be used. A minimal expected dataset layout is:

```text
dataset/
├── image_numpys/
│   ├── image_0.npy
│   ├── image_1.npy
│   └── ...
├── EM_data/
│   ├── transform_0.npy
│   ├── transform_1.npy
│   └── ...
└── calibration_parameters_ivus.yaml
```

Alternate accepted image and transform folder names are listed in the Data section.

Image and transform files are sorted naturally by their numeric frame indices. The number of image files must equal the number of transform files, and every transform must have shape `(4, 4)`.


## Troubleshooting

### `ModuleNotFoundError: No module named 'voxblox'`

The compiled Voxblox bindings are not installed for the active Python interpreter.

Check:

```bash
which python
python -m pip show voxblox
python -c "from voxblox import FastTsdfIntegrator"
```

Rebuild `voxblox_pybind` while the intended virtual environment is active.

### Model weights not found

Supply an existing model path:

```bash
python run_mapping.py \
    --dataset-name patient_1 \
    --model /absolute/path/to/model.weights.h5
```

Also verify `model_path` in `aortascope_mapping_params.yaml`.

### Calibration file not found

Make sure `calibration_parameters_ivus.yaml` is present beside the Python scripts and that the selected dataset contains its acquisition-specific calibration file when required.


### TensorFlow does not detect the GPU

Inspect TensorFlow visibility:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

An empty list means TensorFlow will use the CPU. Verify the TensorFlow installation and system GPU runtime separately.

### Output-saving error

The script saves outputs only after a successful run produces:

- a non-empty lumen mesh;
- a non-empty lumen point cloud;
- a non-empty branch point cloud;
- a segmentation preview.

Review the preceding console output and configuration when any required geometry is empty.



## Citation

When using this software or the associated datasets, cite the accompanying manuscript and the Zenodo record.

Manuscript citation:

```text
Dillon, T. M. et al. Learned ultrasound segmentation and deformable CT fusion
for augmented reality endovascular surgery. Manuscript under review.
```


## Intended use

This software is research code and is not a medical device. It is not intended for clinical diagnosis, treatment, procedural decision-making, or patient care.

## License and review status

This repository is currently prepared for confidential editorial and peer review.

No open-source license is granted unless and until public-release and licensing terms are approved by the relevant institutional intellectual-property office. Review access does not grant permission for redistribution, commercial use, sublicensing, or creation of derivative products beyond what is necessary for manuscript evaluation.

Third-party dependencies remain subject to their respective licenses.
