# Tractography masterclass: from whole-brain tractography to bundle microstructure

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chiaramff/Tractography_IST_Masterclass/blob/main/tractography_masterclass.ipynb)

A Jupyter notebook that takes a whole-brain tractogram from bundle segmentation to bundle-specific FA, comparing:

1. **Bundle extraction approaches**: ROI-based filtering with `tckedit` (MRtrix3) · TRACULA (FreeSurfer) · RecoBundles (DIPY); deterministic vs probabilistic tractography
2. **Interactive visualization**: 3D viewer (pick a tract, rotate, zoom) + slice scroller
3. **Mean FA per bundle**: TRACULA `pathstats` · MRtrix3 (`tcksample`, `tckmap` + `mrstats`) · DIPY; unweighted vs weighted
4. **Along-tract profiles**: orientation → resampling → sampling → weighting → averaging
5. **Bonus**: a structural connectome from the same whole-brain tractogram

## Data

The notebook uses the data from one subject of the [FreeSurfer TRACULA tutorial data](https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/Tracula) (BANDA study). All inputs are downloaded by the notebook the first time it runs:

| what | source |
|---|---|
| FA map, FreeSurfer parcellation and brain mask in diffusion space, TRACULA outputs for 8 bundles (`sub-BANDA001`) | `tracula_tutorial_subset.tar.gz` (~4 MB) in this repository's GitHub release |
| HCP842 bundle atlas (RecoBundles) | downloaded from DIPY (`fetch_bundle_atlas_hcp842`, ~200 MB) |
| Lookup tables for the connectome (`FreeSurferColorLUT.txt`, MRtrix3 `fs_default.txt`) | downloaded from the FreeSurfer and MRtrix3 GitHub repositories |
| Whole-brain tractograms, 200k streamlines each: probabilistic `wholebrain.tck` (iFOD2) and deterministic `wholebrain_det.tck` (SD_STREAM), ~170 MB each | GitHub release of this repository; regenerate with `scripts/01_whole_brain_tractogram.sh` |

## Run it

* **Online, no install (Google Colab)**: click the Colab badge and run the notebook. The first code cell sets everything up on Colab (copies this repository, installs the Python packages and MRtrix3; ~3 min). Needs a Google account. 
* **Locally (Linux, macOS)**:
  ```bash
  conda env create -f environment.yml
  conda activate tractography-masterclass
  jupyter lab tractography_masterclass.ipynb      # start it from this folder
  ```
  Option to specify your paths for add their paths in the notebook's "Optional: your paths" cell (or set `MRTRIX_DIR`, `FREESURFER_HOME`, `TRACULA_TUTORIAL_DIR`).

All downloads go to `data/` and all results to `outputs/` inside the repository folder.

## Regenerating the whole-brain tractogram

Needs the full TRACULA tutorial subject (DWI ~2.5 GB) and MRtrix3:

```bash
bash scripts/01_whole_brain_tractogram.sh tracula_tutorial/trc/sub-BANDA001 derivatives/sub-BANDA001 16
```

MSMT-CSD → ACT (5TT from FreeSurfer `aparc+aseg`) → iFOD2 and SD_STREAM, 2M streamlines each, 0.75 mm step, max angle 45° → SIFT to 200k each.
