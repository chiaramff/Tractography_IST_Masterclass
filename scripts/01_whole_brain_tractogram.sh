#!/usr/bin/env bash
# Whole-brain tractogram for the masterclass (sub-BANDA001, FreeSurfer TRACULA tutorial data).
#
# Input : TRACULA-preprocessed DWI (eddy/topup corrected, diffusion space) + aparc+aseg in diffusion space
# Output: <out_dir>/tractography/wholebrain.tck      (probabilistic, iFOD2, SIFT-filtered)
#         <out_dir>/tractography/wholebrain_det.tck  (deterministic, SD_STREAM, same settings, SIFT-filtered)
#         <out_dir>/mrtrix/                      (MRtrix3 intermediates: FODs, 5TT, full tractograms; safe to delete)
#
# Usage : bash scripts/01_whole_brain_tractogram.sh <tracula_subject_dir> <out_dir> [nthreads]
#   e.g.  bash scripts/01_whole_brain_tractogram.sh tracula_tutorial/trc/sub-BANDA001 derivatives/sub-BANDA001 32
#
# Needs MRtrix3 >= 3.0 and a FreeSurferColorLUT.txt ($FREESURFER_HOME) for 5ttgen.
set -euo pipefail

TRC=$(readlink -f "$1")
mkdir -p "$2"
OUT=$(readlink -f "$2")
NTHREADS=${3:-8}
N_GEN=${N_GEN:-2000000}     # streamlines generated
N_KEEP=${N_KEEP:-200000}    # streamlines kept after SIFT (keeps the file small enough to share)

export MRTRIX_NTHREADS=$NTHREADS
mkdir -p "$OUT/tractography" "$OUT/mrtrix"
cd "$OUT/mrtrix"

# 1. DWI with gradient table (FSL bvecs/bvals written by TRACULA)
[ -f dwi.mif ] || mrconvert "$TRC/dmri/dwi.nii.gz" dwi.mif -fslgrad "$TRC/dmri/bvecs" "$TRC/dmri/bvals"
[ -f mask.mif ] || mrconvert "$TRC/dlabel/diff/lowb_brain_mask.nii.gz" mask.mif -datatype bit

# 2. Multi-shell multi-tissue CSD (data has b=1500 and b=3000 shells)
[ -f wm_response.txt ] || dwi2response dhollander dwi.mif wm_response.txt gm_response.txt csf_response.txt -mask mask.mif
[ -f wmfod_norm.mif ] || {
  dwi2fod msmt_csd dwi.mif -mask mask.mif wm_response.txt wmfod.mif gm_response.txt gm.mif csf_response.txt csf.mif
  mtnormalise wmfod.mif wmfod_norm.mif gm.mif gm_norm.mif csf.mif csf_norm.mif -mask mask.mif
}

# 3. Anatomically-constrained tractography (ACT) from the FreeSurfer segmentation already in diffusion space
[ -f 5tt.mif ] || 5ttgen freesurfer "$TRC/dlabel/diff/aparc+aseg.bbr.nii.gz" 5tt.mif -nocrop
[ -f gmwmi.mif ] || 5tt2gmwmi 5tt.mif gmwmi.mif

# 4. Tractography: probabilistic (iFOD2) and deterministic (SD_STREAM) with the same settings
#    step = half the voxel size (0.75 mm), max angle 45 deg (iFOD2's default; SD_STREAM's would be 60).
#    -backtrack only works for probabilistic algorithms, so only iFOD2 uses it.
TRACK_OPTS=(-step 0.75 -angle 45 -act 5tt.mif -crop_at_gmwmi -seed_gmwmi gmwmi.mif
            -select "$N_GEN" -minlength 20 -maxlength 250 -cutoff 0.06)
[ -f tracks_full.tck ] || tckgen wmfod_norm.mif tracks_full.tck -algorithm iFOD2 -backtrack "${TRACK_OPTS[@]}"
[ -f tracks_full_det.tck ] || tckgen wmfod_norm.mif tracks_full_det.tck -algorithm SD_STREAM "${TRACK_OPTS[@]}"

# 5. SIFT each down to a biologically more plausible (and shareable) subset of the same size
[ -f "$OUT/tractography/wholebrain.tck" ] || \
  tcksift tracks_full.tck wmfod_norm.mif "$OUT/tractography/wholebrain.tck" -act 5tt.mif -term_number "$N_KEEP"
[ -f "$OUT/tractography/wholebrain_det.tck" ] || \
  tcksift tracks_full_det.tck wmfod_norm.mif "$OUT/tractography/wholebrain_det.tck" -act 5tt.mif -term_number "$N_KEEP"

mrconvert 5tt.mif "$OUT/tractography/5tt.nii.gz" -force
tckinfo "$OUT/tractography/wholebrain.tck" -count
tckinfo "$OUT/tractography/wholebrain_det.tck" -count
