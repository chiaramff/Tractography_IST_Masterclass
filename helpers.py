"""Helpers for tractography_masterclass.ipynb.

The analysis steps are in the notebook. This file holds the code that is not the point of the
lesson: running command-line tools, FreeSurfer label numbers, and the figures.
"""
import base64
import io
import shutil
import subprocess
import tarfile
from urllib.request import urlretrieve
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import to_rgb
from plotly.subplots import make_subplots
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.measure import marching_cubes

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.tracking.streamline import length, transform_streamlines


# ---- command-line tools ---------------------------------------------------------------------------

def _echo(cmd, note=""):
    root = str(Path.cwd()) + "/"
    print("$", " ".join(str(c).replace(root, "") for c in cmd) + note)


def run(cmd, show=True):
    # Run a command, echo it (so you can copy it to a terminal) and return stdout.
    cmd = [str(c) for c in cmd]
    if show:
        _echo(cmd)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return r.stdout


# ---- data and files -------------------------------------------------------------------------------

def fetch_tutorial_data(data_dir, release_url, local_dir=None, subject="sub-BANDA001"):
    # Folder with the TRACULA tutorial files the notebook uses (contains trc/<subject>/...).
    # Uses `local_dir` if it has them (a local copy of the tutorial data), otherwise downloads the subset
    # archive from this repository's release once and unpacks it into <data_dir>/tracula_tutorial.
    check = Path("trc") / subject / "dmri" / "dtifit_FA.nii.gz"
    if local_dir and (Path(local_dir) / check).exists():
        return Path(local_dir)
    root = Path(data_dir) / "tracula_tutorial"
    if not (root / check).exists():
        archive = Path(data_dir) / "tracula_tutorial_subset.tar.gz"
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            print("downloading the TRACULA tutorial data (~4 MB) ...")
            urlretrieve(f"{release_url}/{archive.name}", archive)
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            tar.extractall(root)
    return root


def save_tck(streamlines, path, reference):
    # Write streamlines (RAS mm) to a .tck file; `reference` is the image that defines the voxel grid.
    save_tractogram(StatefulTractogram(streamlines, reference, Space.RASMM), str(path), bbox_valid_check=False)


def save_mm(streamlines, path):
    # Write streamlines in mm without a reference image (e.g. in atlas space), as .tck.
    nib.streamlines.save(nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4)), str(path))


def load_mm(path):
    return nib.streamlines.load(str(path)).streamlines


def atlas_name(b):
    # TRACULA bundle name -> HCP842 atlas name, e.g. lh.af -> AF_L
    h, t = b.split(".")
    return f"{t.upper()}_{'L' if h == 'lh' else 'R'}"


# ---- FreeSurfer labels ----------------------------------------------------------------------------
# Label numbers from FreeSurferColorLUT.txt. Cortex (aparc): 1000 + index (lh), 2000 + index (rh).

CORTEX = dict(bankssts=1, caudalmiddlefrontal=3, cuneus=5, entorhinal=6, fusiform=7, inferiorparietal=8,
              inferiortemporal=9, lateraloccipital=11, lateralorbitofrontal=12, lingual=13, medialorbitofrontal=14,
              middletemporal=15, parahippocampal=16, paracentral=17, parsopercularis=18, parsorbitalis=19,
              parstriangularis=20, pericalcarine=21, postcentral=22, precentral=24, rostralmiddlefrontal=27,
              superiorfrontal=28, superiorparietal=29, superiortemporal=30, supramarginal=31, frontalpole=32,
              temporalpole=33, insula=35)
SUBCORTICAL = {"lh": dict(cerebral_wm=2, cerebellum_wm=7, cerebellum_ctx=8, thalamus=10, caudate=11, putamen=12,
                          pallidum=13, amygdala=18),
               "rh": dict(cerebral_wm=41, cerebellum_wm=46, cerebellum_ctx=47, thalamus=49, caudate=50, putamen=51,
                          pallidum=52, amygdala=54)}
BRAINSTEM = 16


def fs_labels(hemi, names):
    # FreeSurfer label numbers for structure names, e.g. fs_labels("lh", ["precentral", "thalamus"]).
    # "brainstem" has no hemisphere; "hemisphere" is all cortex + subcortical labels of that hemisphere.
    ctx = 1000 if hemi == "lh" else 2000
    out = []
    for n in names:
        if n == "brainstem":
            out.append(BRAINSTEM)
        elif n == "hemisphere":
            out += [ctx + i for i in range(36)] + list(SUBCORTICAL[hemi].values())
        elif n in CORTEX:
            out.append(ctx + CORTEX[n])
        else:
            out.append(SUBCORTICAL[hemi][n])
    return out


def extract_label(seg, labels, out, dilate=0, show=True):
    # Binary mask of the voxels of `seg` that have any of `labels`, dilated `dilate` times.
    # Uses FreeSurfer's mri_extract_label if it is installed, otherwise does the same in numpy
    # (mri_extract_label -dilate grows the mask by a 3x3x3 cube per iteration).
    cmd = ["mri_extract_label"] + (["-dilate", dilate] if dilate else []) + [seg, *labels, out]
    if shutil.which("mri_extract_label"):
        run(cmd, show)
        return out
    if show:
        _echo(cmd, "   # FreeSurfer not found: same operation in numpy")
    img = nib.load(seg)
    mask = np.isin(np.asarray(img.dataobj), labels)
    if dilate:
        mask = binary_dilation(mask, np.ones((3, 3, 3)), iterations=dilate)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), img.affine), out)
    return out


def endpoint_roi(seg, labels, wm, out, show=True):
    # Endpoint ROI: the labels, grown by one voxel (3x3x3) but only into white matter (`wm` mask). This catches
    # streamlines that stop just below the cortex (ACT, -crop_at_gmwmi) without spreading into neighbouring gyri.
    out = Path(out)
    steps = out.parent / "steps"
    steps.mkdir(exist_ok=True)
    label, grown = steps / out.name.replace(".nii", "_label.nii"), steps / out.name.replace(".nii", "_dil.nii")
    extract_label(seg, labels, label, show=show)
    extract_label(seg, labels, grown, dilate=1, show=show)
    run(["mrcalc", grown, wm, "-and", label, "-or", out, "-datatype", "bit", "-force", "-quiet"], show)
    return out


# ---- TRACULA ----------------------------------------------------------------------------------------

def read_pathstats(folder):
    # pathstats.overall.txt of one TRACULA bundle as a dict, e.g. {"Volume": 3610, "FA_Avg": 0.494, ...}
    rows = [l.split() for l in open(Path(folder) / "pathstats.overall.txt") if l.strip() and not l.startswith("#")]
    return {k: float(v) for k, v in rows}


def read_byvoxel(folder):
    # pathstats.byvoxel.txt of one TRACULA bundle as a table (one row per voxel of the reference path)
    import pandas as pd
    return pd.read_csv(Path(folder) / "pathstats.byvoxel.txt", sep=r"\s+", comment="#")


def above_robust_max(d, frac=0.2):
    # TRACULA's rule: voxels above 20% of the robust maximum (99th percentile of the non-zero voxels)
    return d > frac * np.percentile(d[d > 0], 99)


def bundle_length(method, streamlines, pathstats, voxel_mm):
    # Median streamline length in mm; TRACULA: Len_Avg from pathstats (in voxels) x voxel size
    if method == "TRACULA":
        return pathstats["Len_Avg"] * voxel_mm
    return np.median(length(streamlines)) if len(streamlines) else np.nan


# ---- 3D views -------------------------------------------------------------------------------------

SCENE = dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
             camera=dict(eye=dict(x=-1.6, y=0.3, z=0.3)))


def lines3d(sl, color, name, max_n=150, seed=0, **kw):
    # Up to max_n streamlines as ONE Scatter3d trace (NaN rows break the line between streamlines).
    if len(sl) == 0:
        return go.Scatter3d(x=[], y=[], z=[], mode="lines", name=name, line=dict(color=color), **kw)
    idx = np.random.default_rng(seed).choice(len(sl), min(max_n, len(sl)), replace=False)
    xyz = np.vstack([np.vstack([sl[i], [np.nan] * 3]) for i in idx])
    kw.setdefault("opacity", 0.6)
    return go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines", name=name,
                        line=dict(color=color, width=2), hoverinfo="name", **kw)


def isosurface(vol, level, affine, color, name, step=1, **kw):
    verts, faces, _, _ = marching_cubes(vol, level, step_size=step)
    v = nib.affines.apply_affine(affine, verts)
    kw.setdefault("hoverinfo", "name")
    return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                     color=color, name=name, **kw)


def pd_surface(pd, affine, color, name="TRACULA path.pd", level=0.1, **kw):
    # TRACULA's output as it is computed: a surface around the voxels where the path probability
    # distribution is above `level` x its maximum (lightly smoothed so the surface is not blocky).
    s = gaussian_filter(pd, 0.7)
    kw.setdefault("opacity", 0.35)
    return isosurface(s, level * s.max(), affine, color, name, **kw)


def glass_brain(mask, affine):
    return isosurface(gaussian_filter(mask.astype(float), 2), 0.5, affine, "#b5b4ad", "brain", step=3,
                      opacity=0.07, hoverinfo="skip", showlegend=False)


def _bundle_traces(m, b, bundles, pd_vol, affine, color, **kw):
    # (trace, default visibility) pairs for one bundle from one method.
    # TRACULA: the path distribution surface is shown, the path samples (.trk) are one click away.
    if m == "TRACULA":
        return [(pd_surface(pd_vol[b], affine, color, legendgroup="pd", **kw), True),
                (lines3d(bundles[m][b], color, "TRACULA path.pd.trk", legendgroup="trk", **kw), "legendonly")]
    return [(lines3d(bundles[m][b], color, m, legendgroup=m, **kw), True)]


def _bundle_menu(fig, owner, names, **pos):
    # Dropdown that shows one bundle at a time; owner[i] = (bundle or None, default visibility) of trace i.
    vis = lambda b: [v if o is None or o == b else False for o, v in owner]
    for trace, v in zip(fig.data, vis(names[0])):
        trace.visible = v
    fig.update_layout(updatemenus=[dict(buttons=[dict(label=b, method="update", args=[{"visible": vis(b)}])
                                                 for b in names], xanchor="left", **pos)])


def plot_bundles_3d(bundles, pd_vol, brain_mask, affine, colors):
    # All methods in one 3D scene, one bundle at a time.
    names = list(pd_vol)
    fig = go.Figure([glass_brain(brain_mask, affine)])
    owner = [(None, True)]
    for b in names:
        for m, c in colors.items():
            for trace, v in _bundle_traces(m, b, bundles, pd_vol, affine, c):
                fig.add_trace(trace)
                owner.append((b, v))
    _bundle_menu(fig, owner, names, x=0, y=1.0, yanchor="top")
    fig.update_layout(height=650, template="plotly_white", scene=SCENE, margin=dict(l=0, r=0, t=40, b=0),
                      title="Bundles by method (drag to rotate, scroll to zoom, click the legend to toggle)",
                      legend=dict(itemsizing="constant", y=0.9))
    return fig


def plot_side_by_side(bundles, pd_vol, brain_mask, affine, colors):
    # One 3D panel per method (each panel rotates independently). pd_vol is only needed for TRACULA.
    methods = list(colors)
    names = list(bundles[methods[0]])
    fig = make_subplots(rows=1, cols=len(methods), specs=[[{"type": "scene"}] * len(methods)],
                        subplot_titles=methods, horizontal_spacing=0.0)
    owner = []
    for c, m in enumerate(methods, start=1):
        fig.add_trace(glass_brain(brain_mask, affine), row=1, col=c)
        owner.append((None, True))
        for b in names:
            for trace, v in _bundle_traces(m, b, bundles, pd_vol, affine, colors[m]):
                trace.showlegend = m == "TRACULA"                        # legend only for TRACULA's two views
                fig.add_trace(trace, row=1, col=c)
                owner.append((b, v))
    _bundle_menu(fig, owner, names, x=0, y=1.15)
    fig.update_layout(height=450, template="plotly_white", margin=dict(l=0, r=0, t=60, b=0),
                      legend=dict(orientation="h", y=-0.05))
    fig.update_scenes(**SCENE)
    return fig


# one colour per bundle, left/right pairs in two shades of the same colour
BUNDLE_COLORS = {"lh.af": "#2a78d6", "rh.af": "#7fb0ec", "lh.cst": "#1baf7a", "rh.cst": "#79d6b1",
                 "lh.ilf": "#eb6834", "rh.ilf": "#f4a983", "lh.uf": "#8e4fd1", "rh.uf": "#c3a0ea"}


def _scene_fig(height=600, title=None):
    fig = go.Figure()
    fig.update_layout(height=height, template="plotly_white", scene=SCENE, margin=dict(l=0, r=0, t=40, b=0),
                      title=title, legend=dict(itemsizing="constant", y=0.9))
    return fig


def plot_tracula(pd_vol, brain_mask, affine):
    # All TRACULA path distributions together (surface at 10% of each path.pd's maximum); click the legend to hide one.
    fig = _scene_fig(650, "TRACULA: path distributions (path.pd) of all bundles")
    fig.add_trace(glass_brain(brain_mask, affine))
    for b, pd_ in pd_vol.items():
        fig.add_trace(pd_surface(pd_, affine, BUNDLE_COLORS.get(b, "#2a78d6"), name=b, opacity=0.6, showlegend=True))
    return fig


def plot_rois(bundle, roi_dir, seg_file, streamlines, brain_mask, affine):
    # Endpoint ROIs A (green) and B (blue), the exclusion ROI X in the bundle's hemisphere (faint red), and the bundle.
    h = bundle.split(".")[0]
    other = "rh" if h == "lh" else "lh"
    seg = np.asarray(nib.load(seg_file).dataobj)
    other_hemi = np.isin(seg, fs_labels(other, ["hemisphere"]))
    load = lambda k: nib.load(Path(roi_dir) / f"{bundle}_{k}.nii.gz").get_fdata() > 0
    A, B, X = load("A"), load("B"), load("X") & ~other_hemi
    fig = _scene_fig(650, f"{bundle}: endpoint ROIs A and B, exclusion ROI X (this hemisphere), streamlines")
    fig.add_trace(glass_brain(brain_mask, affine))
    for mask, color, name, op in ((X, "#e5484d", "X (exclude)", 0.08), (A, "#1baf7a", "A", 0.5), (B, "#2a78d6", "B", 0.5)):
        fig.add_trace(isosurface(gaussian_filter(mask.astype(float), 0.5), 0.5, affine, color, name,
                                 opacity=op, showlegend=True))
    fig.add_trace(lines3d(streamlines, "#0b0b0b", f"streamlines ({len(streamlines)})"))
    return fig


def plot_atlas_bundles(models):
    # The HCP842 model bundles (in MNI space), one colour per bundle.
    fig = _scene_fig(600, "HCP842 atlas: model bundles (MNI space)")
    for b, sl in models.items():
        fig.add_trace(lines3d(sl, BUNDLE_COLORS.get(b, "#8a8984"), atlas_name(b), max_n=200))
    return fig


def plot_slr(atlas_centroids, subject_centroids, xfm):
    # Whole-brain streamline-based registration (SLR): cluster centroids before and after registration.
    moved = transform_streamlines(subject_centroids, xfm)
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}] * 2],
                        subplot_titles=["before registration", "after registration (SLR)"], horizontal_spacing=0.0)
    for c, sub in ((1, subject_centroids), (2, moved)):
        fig.add_trace(lines3d(atlas_centroids, "#8a8984", "atlas centroids (MNI)", max_n=len(atlas_centroids),
                              showlegend=(c == 1), legendgroup="atlas"), row=1, col=c)
        fig.add_trace(lines3d(sub, "#eb6834", "our centroids", max_n=len(sub), showlegend=(c == 1), legendgroup="ours"),
                      row=1, col=c)
    fig.update_scenes(**SCENE)
    fig.update_layout(height=480, template="plotly_white", margin=dict(l=0, r=0, t=60, b=0),
                      legend=dict(orientation="h", y=-0.05, itemsizing="constant"))
    return fig


def plot_reco_thresholds(results, model, title):
    # results: {(reduction_thr, pruning_thr): recognized streamlines in MNI space}. One slider step per setting;
    # the model bundle (faint) is always shown.
    fig = _scene_fig(600)
    fig.add_trace(lines3d(model, "#8a8984", "model bundle (atlas)", max_n=200, opacity=0.25))
    settings = list(results)
    for setting in settings:
        fig.add_trace(lines3d(results[setting], "#eb6834", "recognized", visible=(setting == (15, 7))))
    steps = []
    for k, (red, prune) in enumerate(settings):
        n = len(results[(red, prune)])
        steps.append(dict(method="update", label=f"{red} / {prune}",
                          args=[{"visible": [True] + [j == k for j in range(len(settings))]},
                                {"title.text": f"{title}: reduction_thr {red}, pruning_thr {prune} → {n} streamlines"}]))
    start = settings.index((15, 7)) if (15, 7) in settings else 0
    fig.update_layout(title=steps[start]["args"][1]["title.text"],
                      sliders=[dict(active=start, steps=steps, pad=dict(t=30),
                                    currentvalue=dict(prefix="reduction / pruning threshold (mm): "))])
    return fig


def plot_wholebrain(wb, bin_mm=5, per_bin=100, n_all=2000, seed=1):
    # Whole-brain tractogram explorer: a slider picks a length bin, the histogram shows where it is.
    # Colour = direction (red: left-right, green: anterior-posterior, blue: superior-inferior).
    from dipy.tracking.streamline import length
    wb_len = length(wb)
    edges = np.arange(20, wb_len.max() + bin_mm, bin_mm)
    counts, hist_edges = np.histogram(wb_len, bins=np.arange(0, wb_len.max() + 2, 2))
    pts = wb.get_data()
    lo, hi = pts.min(0), pts.max(0)
    rng = np.random.default_rng(seed)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.06,
                        specs=[[{"type": "scene"}, {"type": "xy"}]])
    owner = []                                                        # slider step of each 3D trace

    def add_streamlines(idx, step):
        if len(idx) == 0:
            return
        ends = np.array([np.abs(wb[i][-1] - wb[i][0]) for i in idx])
        rgb = np.round(ends / np.linalg.norm(ends, axis=1, keepdims=True) * 2) / 2
        for col in np.unique(rgb, axis=0):
            sel = idx[(rgb == col).all(1)]
            xyz = np.vstack([np.vstack([wb[i], [np.nan] * 3]) for i in sel])
            fig.add_scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines", visible=(step == 0), opacity=0.6,
                              line=dict(color="rgb({},{},{})".format(*(col * 255).astype(int)), width=1.5),
                              hoverinfo="skip", showlegend=False, row=1, col=1)
            owner.append(step)

    groups = [("all", 0, np.inf, rng.choice(len(wb), n_all, replace=False))]
    for a, b in zip(edges[:-1], edges[1:]):
        in_bin = np.flatnonzero((wb_len >= a) & (wb_len < b))
        groups.append((f"{a:.0f}", a, b, rng.choice(in_bin, min(per_bin, len(in_bin)), replace=False)))
    for step, (_, _, _, idx) in enumerate(groups):
        add_streamlines(idx, step)

    fig.add_bar(x=hist_edges[:-1] + 1, y=counts, width=2, marker_color="#b5b4ad", showlegend=False,
                hovertemplate="%{x:.0f} mm: %{y:,} streamlines<extra></extra>", row=1, col=2)

    steps = []
    for step, (label, a, b, idx) in enumerate(groups):
        if label == "all":
            shapes, title = [], f"Whole-brain tractogram: {n_all:,} of {len(wb):,} streamlines"
        else:
            n = int(((wb_len >= a) & (wb_len < b)).sum())
            shapes = [dict(type="rect", xref="x", yref="y domain", x0=a, x1=b, y0=0, y1=1,
                           fillcolor="#6250d6", opacity=0.2, line_width=0)]
            title = f"{a:.0f}–{b:.0f} mm: {n:,} streamlines (showing {len(idx)})"
        steps.append(dict(method="update", label=label,
                          args=[{"visible": [o == step for o in owner] + [True]}, {"shapes": shapes, "title.text": title}]))

    fig.update_xaxes(title_text="streamline length (mm)", row=1, col=2)
    fig.update_yaxes(title_text="count", row=1, col=2)
    ext = hi - lo
    fig.update_scenes(aspectmode="manual", aspectratio=dict(zip("xyz", ext / ext.max())),
                      xaxis=dict(range=[lo[0], hi[0]], visible=False), yaxis=dict(range=[lo[1], hi[1]], visible=False),
                      zaxis=dict(range=[lo[2], hi[2]], visible=False), camera=dict(eye=dict(x=-1.6, y=0.3, z=0.3)))
    fig.update_layout(height=550, template="plotly_white", margin=dict(l=0, r=20, t=50, b=0),
                      title=steps[0]["args"][1]["title.text"],
                      sliders=[dict(active=0, steps=steps, pad=dict(t=40),
                                    currentvalue=dict(prefix="length (mm): "))])
    return fig


# ---- slice scroller -------------------------------------------------------------------------------

PLANES = {"axial": 2, "coronal": 1, "sagittal": 0}


def _slice_png(fa_rgb, d, color, scale, ax, i):
    # One slice of FA with a density map on top, as an inline PNG. Every voxel the bundle touches is at least
    # 70% opaque; colour goes from a light tint (low density) to the full colour (>= scale, the robust maximum).
    rgb = np.rot90(np.take(fa_rgb, i, axis=ax))
    d = np.rot90(np.take(d, i, axis=ax))
    v = np.clip(d / scale, 0, 1)[..., None]
    a = np.where(d > 0, 0.7 + 0.3 * v[..., 0], 0)[..., None]
    col = np.array(to_rgb(color))
    c = (1 - v) * (0.7 * col + 0.3) + v * col
    rgb = rgb * (1 - a) + c * a
    buf = io.BytesIO()
    Image.fromarray((rgb * 255).astype(np.uint8)).save(buf, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def slice_scroller(density, fa, colors, bundle="lh.af", plane="axial"):
    # density[method][bundle]: volume on the FA grid. One panel per method, a slider scrolls the slices.
    methods = list(colors)
    ax = PLANES[plane]
    fa_rgb = 0.65 * plt.get_cmap("gray")(np.clip(fa / 0.9, 0, 1))[..., :3]          # dimmed so the colours stand out
    scale = {m: np.percentile(density[m][bundle][density[m][bundle] > 0], 99) for m in methods}   # robust max
    total = sum(density[m][bundle] for m in methods)
    other = tuple(a for a in range(3) if a != ax)
    hit = np.nonzero(total.sum(axis=other))[0]
    idx = list(range(max(hit[0] - 3, 0), min(hit[-1] + 4, fa.shape[ax])))     # only slices the bundle touches
    com = np.array(np.nonzero(total)).mean(1)                                 # start at the centre of mass
    start = min(idx, key=lambda i: abs(i - com[ax]))
    src = {i: [_slice_png(fa_rgb, density[m][bundle], colors[m], scale[m], ax, i) for m in methods] for i in idx}

    fig = make_subplots(rows=1, cols=len(methods), subplot_titles=[f"{m}: {bundle}" for m in methods],
                        horizontal_spacing=0.02)
    for k in range(len(methods)):
        fig.add_trace(go.Image(source=src[start][k], hoverinfo="skip"), 1, k + 1)
        if plane != "sagittal":   # array is LAS: after rot90 the subject's right is on the image left
            for x, t in ((3, "R"), (fa.shape[0] - 8, "L")):
                fig.add_annotation(x=x, y=6, text=t, showarrow=False, font=dict(color="white"),
                                   xref=f"x{k + 1}", yref=f"y{k + 1}", xanchor="left")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=470, margin=dict(l=10, r=10, t=40, b=10),
        sliders=[dict(active=idx.index(start), pad=dict(t=10),
                      currentvalue=dict(prefix=f"{plane} slice "),
                      steps=[dict(method="restyle", label=str(i), args=[{"source": src[i]}, list(range(len(methods)))])
                             for i in idx])])
    return fig


# ---- mean FA --------------------------------------------------------------------------------------

def fa_table(fa_df, measures, methods, bundle_names):
    # mean FA results as a table: one row per bundle; columns: measure / method / tool
    t = fa_df[fa_df.measure.isin(measures)].pivot_table(index="bundle", columns=["measure", "source", "tool"], values="FA")
    return t.reindex(columns=measures, level="measure").reindex(columns=methods, level="source").loc[bundle_names].round(3)


def plot_mean_fa(fa_df, colors, bundle_names):
    # Unweighted vs weighted mean FA per bundle. TRACULA: its own pathstats; others: MRtrix3 voxel values.
    def pick(measure):
        sel = fa_df[(fa_df.measure == measure) &
                    (((fa_df.source == "TRACULA") & (fa_df.tool == "TRACULA pathstats")) |
                     ((fa_df.source != "TRACULA") & (fa_df.tool == "MRtrix3")))]
        return sel.set_index(["source", "bundle"]).FA

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=["unweighted (each voxel once)", "weighted (streamline density / path probability)"])
    for c, meas in enumerate(["voxel mean", "voxel weighted"], start=1):
        s = pick(meas)
        for m, color in colors.items():
            y = [s.get((m, b), np.nan) for b in bundle_names]
            fig.add_bar(x=bundle_names, y=y, name=m, marker_color=color, marker_cornerradius=4,
                        showlegend=(c == 1), legendgroup=m, row=1, col=c,
                        hovertemplate=f"{m}<br>%{{x}}<br>FA = %{{y:.3f}}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, height=430, template="plotly_white",
                      title="Mean FA per bundle", legend=dict(orientation="h", y=-0.15))
    fig.update_yaxes(range=[0.3, 0.65], title_text="mean FA", col=1)
    return fig


def plot_mrtrix_vs_dipy(fa_df, colors):
    # One point per bundle and method: MRtrix3 value (x) vs DIPY value (y), for each definition of the mean.
    meas_list = ["point mean", "streamline mean", "voxel mean", "voxel weighted"]
    wide = fa_df[fa_df.tool.isin(["MRtrix3", "DIPY"])].pivot_table(
        index=["source", "bundle", "measure"], columns="tool", values="FA").reset_index()
    fig = make_subplots(rows=1, cols=len(meas_list), subplot_titles=meas_list, horizontal_spacing=0.06)
    for c, meas in enumerate(meas_list, start=1):
        fig.add_scatter(x=[0.35, 0.65], y=[0.35, 0.65], mode="lines", line=dict(color="#8a8984", dash="dash", width=1),
                        showlegend=False, hoverinfo="skip", row=1, col=c)
        for m, color in colors.items():
            w = wide[(wide.measure == meas) & (wide.source == m)]
            if w.empty:                                     # e.g. TRACULA: no streamline-based values
                continue
            fig.add_scatter(x=w.MRtrix3, y=w.DIPY, mode="markers", name=m, legendgroup=m, showlegend=(c == 1),
                            marker=dict(color=color, size=9, line=dict(color="white", width=2)),
                            text=w.bundle, hovertemplate="%{text}<br>MRtrix3 %{x:.3f}<br>DIPY %{y:.3f}<extra>" + m + "</extra>",
                            row=1, col=c)
    fig.update_xaxes(title_text="MRtrix3", range=[0.35, 0.65])
    fig.update_yaxes(title_text="DIPY", range=[0.35, 0.65])
    fig.update_layout(height=400, template="plotly_white", legend=dict(orientation="h", y=-0.25))
    return fig


# ---- along-tract profiles -------------------------------------------------------------------------

INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"


def plot_orientation(raw, oriented, color):
    # First point of every streamline, before and after orienting.
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}] * 2], subplot_titles=["raw", "oriented"])
    for c, sl in enumerate([raw, oriented], start=1):
        fig.add_trace(lines3d(sl, "#b5b4ad", "streamlines", showlegend=False), row=1, col=c)
        starts = np.array([s[0] for s in sl])
        fig.add_scatter3d(x=starts[:, 0], y=starts[:, 1], z=starts[:, 2], mode="markers", name="first point",
                          marker=dict(size=3, color=color), showlegend=(c == 1), row=1, col=c)
    fig.update_scenes(**SCENE)
    fig.update_layout(height=420, template="plotly_white", margin=dict(l=0, r=0, t=40, b=0))
    return fig


def plot_profile_steps(raw_values, fa_oriented, prof_unoriented, prof_oriented, prof_weighted, color, along):
    # (a) raw FA along streamlines, (b) effect of orientation, (c) effect of Gaussian weighting.
    x = np.linspace(0, 100, fa_oriented.shape[1])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for v in raw_values:
        ax[0].plot(v, color=MUTED, lw=0.6, alpha=0.6)
    ax[0].set(title="(a) raw: different number of points", xlabel="point index", ylabel="FA")
    for a in ax[1:]:
        for v in fa_oriented[:60]:
            a.plot(x, v, color=GRID, lw=0.6)
    ax[1].plot(x, prof_unoriented, color=MUTED, lw=2, ls="--", label="resampled, NOT oriented")
    ax[1].plot(x, prof_oriented, color=color, lw=2, label="resampled + oriented")
    ax[1].set(title="(b) orientation matters", xlabel=f"% along tract ({along})")
    ax[2].plot(x, prof_oriented, color=MUTED, lw=2, ls="--", label="unweighted mean")
    ax[2].plot(x, prof_weighted, color=color, lw=2, label="Gaussian-weighted (AFQ)")
    ax[2].set(title="(c) weighting", xlabel="% along tract")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    for a in ax[1:]:
        a.axvspan(0, 10, color=GRID, alpha=0.5)
        a.axvspan(90, 100, color=GRID, alpha=0.5)
        a.legend(frameon=False, loc="lower center")
    ax[0].set_ylim(0.1, 0.9)
    plt.tight_layout()
    plt.show()


def plot_profiles(prof_df, colors, start_label, end_label):
    # Along-tract FA profiles: one panel per bundle, colour = method, dash = tool.
    dash = {"DIPY afq_profile (weighted)": "solid", "MRtrix3 (unweighted)": "dot", "TRACULA pathstats (FA_Avg)": "dash"}
    tracts = list(start_label)
    fig = make_subplots(rows=len(tracts), cols=2, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.05,
                        horizontal_spacing=0.04, subplot_titles=[f"{h}.{t}" for t in tracts for h in ("lh", "rh")])
    seen = set()
    for r, t in enumerate(tracts, start=1):
        for c, h in enumerate(("lh", "rh"), start=1):
            b = f"{h}.{t}"
            for src in colors:
                for tool, g in prof_df[(prof_df.bundle == b) & (prof_df.source == src)].groupby("tool", sort=False):
                    key = (src, tool)
                    fig.add_scatter(x=g.pct, y=g.FA, mode="lines", name=f"{src}: {tool}", legendgroup=f"{src}{tool}",
                                    showlegend=key not in seen, line=dict(color=colors[src], dash=dash[tool], width=2),
                                    hovertemplate=f"{src}<br>{tool}<br>%{{x:.0f}}% → FA %{{y:.3f}}<extra>{b}</extra>",
                                    row=r, col=c)
                    seen.add(key)
            fig.add_vrect(x0=0, x1=10, fillcolor=GRID, opacity=0.5, line_width=0, row=r, col=c)
            fig.add_vrect(x0=90, x1=100, fillcolor=GRID, opacity=0.5, line_width=0, row=r, col=c)
        fig.update_xaxes(title_text=f"% along ({start_label[t]} → {end_label[t]})" if r == len(tracts) else None, row=r)
    fig.update_yaxes(title_text="FA", col=1)
    fig.update_layout(height=1100, template="plotly_white", hovermode="x unified",
                      title="Along-tract FA profiles (grey bands: ends that are usually trimmed)",
                      legend=dict(orientation="h", y=-0.08, font=dict(size=11)))
    return fig


# ---- connectome -----------------------------------------------------------------------------------

def read_node_names(lut_file):
    # Node names from an MRtrix3 labelconvert table (e.g. fs_default.txt): index, short name, full name, RGBA.
    # (fs_default.txt lists two FreeSurfer names for the thalamus; keep the first)
    names = {}
    for r in (l.split() for l in open(lut_file) if l.strip() and not l.startswith("#")):
        names.setdefault(int(r[0]), r[2])
    names.pop(0, None)
    return names


def plot_connectome(C, names):
    # Connectivity matrix as a heatmap (log scale), hover shows the two regions and the streamline count.
    labels = [names[i + 1] for i in range(C.shape[0])]
    text = [[f"{labels[i]} – {labels[j]}: {int(C[i, j]):,} streamlines" for j in range(C.shape[1])] for i in range(C.shape[0])]
    fig = go.Figure(go.Heatmap(z=np.log10(1 + C), x=labels, y=labels, text=text, hoverinfo="text",
                               colorscale="Viridis", colorbar=dict(title="log10(1 + streamlines)")))
    fig.update_layout(height=750, width=800, template="plotly_white", title="Structural connectome (84 regions)",
                      xaxis=dict(showticklabels=False, title="region"), margin=dict(l=40, r=20, t=50, b=40),
                      yaxis=dict(showticklabels=False, title="region", autorange="reversed", scaleanchor="x"))
    return fig
