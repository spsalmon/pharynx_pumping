import itertools
import json
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import skan
import skimage
import tifffile
from matplotlib.gridspec import GridSpec
from scipy import interpolate, sparse
from scipy.sparse import csgraph
from scipy.spatial import distance

import worm_backbone

# pixel scales for ZYX axes as seen in raw image metadata
DEFAULT_SCALE = np.asarray((2.0, 0.325, 0.325))


class Worm:
    def __init__(
        self, splines: list[interpolate.BSpline], length, scale=DEFAULT_SCALE
    ) -> None:
        self.splines = splines
        self.length = length
        self.scale = scale

    def to_json(self):
        ...

    def imgspace_to_wormspace(self, coords):
        # each row is a point with 3 dimensions
        # coords.shape indices
        POINTS_AXIS, DIM_AXIS = 0, 1
        # col indices
        DIM_Z, DIM_Y, DIM_X = 0, 1, 2
        # each unique Z refers to a distinct spline
        zs = coords[:, DIM_Z].astype(int)
        wormspace_coords = []
        for z in np.unique(zs):
            indices = np.ix_(zs == z, [DIM_Y, DIM_X])
            yx_coords = coords[indices]
            wormspace_coords.append(self._map_onto_spline(z, yx_coords))
        wormspace_coords = np.vstack(wormspace_coords)
        return wormspace_coords

    def _map_onto_spline(self, z, coords):
        # TODO: possible optimisations are to create a KDTree of worm backbone for faster distance queries; create a worm class that contains (and caches) certain calculations like worm length and backbone pixels, and aforementioned KDTree (only want to initialise the tree once)
        spline = self.splines[z]
        parameter_u = np.linspace(0, 1, 1000)
        backbone_pixels = spline(parameter_u)
        non_na_mask = np.all(~np.isnan(backbone_pixels), axis=1)
        backbone_pixels = backbone_pixels[non_na_mask]
        parameter_u = parameter_u[non_na_mask]
        backbone_derivs = spline.derivative(1)(parameter_u)

        coords = coords / self.scale[[1, 2]]

        (
            closest_backbone_pixels,
            closest_backbone_indices,
            distance_to_backbone,
        ) = closest_points(
            coords,
            backbone_pixels,
            metric="euclidean",
            return_indices=True,
            return_distances=True,
        )

        cross_product = np.cross(
            backbone_derivs[closest_backbone_indices],
            coords - closest_backbone_pixels,
        )
        signs = np.sign(cross_product)
        distance_to_backbone = distance_to_backbone * signs / self.length

        height = z / self.scale[0] / self.length
        height = np.full(len(distance_to_backbone), height)

        return np.c_[
            height, distance_to_backbone, parameter_u[closest_backbone_indices]
        ]

    def wormspace_to_pixelspace(self, coords):
        # each row is a point with 3 dimensions
        # coords.shape indices
        POINTS_AXIS, DIM_AXIS = 0, 1
        # col indices
        DIM_Z, DIM_Y, DIM_X = 0, 1, 2

        zs = coords[:, DIM_Z]
        img_z = np.round(zs * self.length * self.scale[DIM_Z]).astype(int)

        img_coords = []
        for z in np.unique(img_z):
            spline = self.splines[z]
            indices = np.ix_(img_z == z, [DIM_Y, DIM_X])
            vs, us = coords[indices].T
            origins = spline(us)
            offsets = vs * self.length
            derivs = spline(us)
            normals = derivs[:, [1, 0]] * np.array([1, -1])
            # unit length normals
            normals = normals / np.linalg.norm(normals, axis=1)
            img_yxs = origins + normals * offsets
            img_yxs = img_yxs * self.scale[[DIM_Y, DIM_X]]
            img_zyxs = np.c_[np.full(len(img_yxs), z), img_yxs]
            img_coords.append(img_zyxs)
        img_coords = np.vstack(img_coords)
        return img_coords

        spline = self.splines[np.round(img_z).astype(int)]
        # parameter_u = np.linspace(0, 1, 1000)
        # backbone_pixels = spline(parameter_u)
        # non_na_mask = np.all(~np.isnan(backbone_pixels), axis=1)
        # backbone_pixels = backbone_pixels[non_na_mask]
        # parameter_u = parameter_u[non_na_mask]
        # backbone_derivs = spline.derivative(1)(parameter_u)

    @classmethod
    def from_json(cls, json_file):
        data = json.load(json_file)
        return cls(**data)

    @classmethod
    def from_img(cls, img, scale=DEFAULT_SCALE):
        backbones = worm_backbone.generate_worm_backbone(img)
        return cls.from_backbones(backbones)

    @classmethod
    def from_backbones(cls, backbones: list[np.ndarray], scale=DEFAULT_SCALE):
        DIM_Y, DIM_X = 1, 2
        scaled_backbones = [backbone / scale[[DIM_Y, DIM_X]] for backbone in backbones]
        backbone_lengths = np.asarray(
            [calculate_worm_length(backbone) for backbone in scaled_backbones]
        )
        max_length = backbone_lengths.max()
        # for every pixel in real space, move u_per_pixel in parameter space
        us_per_pixel = 1 / max_length

        splines = []
        for backbone in scaled_backbones:
            ys, xs = backbone.T
            us = generate_parametrisation(backbone, max_length)
            tck, u = interpolate.splprep([ys, xs], u=us, ub=us[0], ue=us[-1], s=0)
            t, c, k = tck
            c = np.asarray(c).T
            spline = interpolate.BSpline(t, c, k, extrapolate=False)
            splines.append(spline)

        return cls(splines, max_length, scale)


# @profile
def closest_points(
    source_points, target_points, metric, return_indices=False, return_distances=False
):
    # pair_wise_dists[i, j] is distance between source_points[i] and target_points[j]
    SOURCE = 0
    TARGET = 1
    pair_wise_dists = distance.cdist(source_points, target_points, metric=metric)
    min_dist_indices = np.argmin(pair_wise_dists, axis=TARGET)

    closest_target_points = target_points[min_dist_indices]
    distances_to_closest = np.min(pair_wise_dists, axis=TARGET)
    if return_indices and return_distances:
        return closest_target_points, min_dist_indices, distances_to_closest
    elif return_indices:
        return closest_target_points, min_dist_indices
    elif return_distances:
        return closest_target_points, distances_to_closest
    else:
        return closest_target_points


def generate_parametrisation(backbone, max_length):
    backbone_inter_pixel_dist = np.linalg.norm(backbone[:-1] - backbone[1:], axis=1)
    length = backbone_inter_pixel_dist.sum()
    u_per_pixel = 1 / max_length

    backbone_inter_u_dist = backbone_inter_pixel_dist * u_per_pixel
    start = (1 - length * u_per_pixel) / 2
    us = np.zeros(len(backbone))
    us[0] = start
    us[1:] = np.cumsum(backbone_inter_u_dist) + start

    return us


# @profile
def map_pixelspace_coords_to_worm_coords(
    worm_backbone_spline: interpolate.BSpline, pixelspace_coords, u_per_pixel_conversion
):
    # TODO: possible optimisations are to create a KDTree of worm backbone for faster distance queries; create a worm class that contains (and caches) certain calculations like worm length and backbone pixels, and aforementioned KDTree (only want to initialise the tree once)
    parameter_u = np.linspace(0, 1, 1000)
    backbone_pixels = worm_backbone_spline(parameter_u)
    non_na = np.all(~np.isnan(backbone_pixels), axis=1)
    backbone_pixels = backbone_pixels[non_na]
    parameter_u = parameter_u[non_na]
    backbone_derivs = worm_backbone_spline.derivative(1)(parameter_u)

    (
        closest_backbone_pixels,
        closest_backbone_indices,
        distance_to_backbone,
    ) = closest_points(
        pixelspace_coords,
        backbone_pixels,
        metric="euclidean",
        return_indices=True,
        return_distances=True,
    )

    cross_product = np.cross(
        backbone_derivs[closest_backbone_indices],
        pixelspace_coords - closest_backbone_pixels,
    )
    signs = np.sign(cross_product)
    # distance_to_backbone = distance_to_backbone * signs
    distance_to_backbone = distance_to_backbone * signs * u_per_pixel_conversion
    print("distdelta:", distance_to_backbone.max() - distance_to_backbone.min())
    return np.c_[distance_to_backbone, parameter_u[closest_backbone_indices]]


# @profile
def calculate_worm_length(backbone):
    return np.linalg.norm(backbone[:-1] - backbone[1:], axis=1).sum()


if __name__ == "__main__":
    ...
    # worms = [
    #     {"Time": "00241", "Point": "0017", "Seq": "16446"},
    #     {"Time": "00243", "Point": "0017", "Seq": "16582"},
    #     {"Time": "00250", "Point": "0017", "Seq": "17058"},
    #     {"Time": "00256", "Point": "0017", "Seq": "17466"},
    #     {"Time": "00258", "Point": "0017", "Seq": "17602"},
    #     {"Time": "00264", "Point": "0017", "Seq": "18010"},
    # ]

    # worm = worms[-1]

    # for worm in worms:
    #     plots_path = Path(
    #         f"plots/Time{worm['Time']}_Point{worm['Point']}_Seq{worm['Seq']}"
    #     )
    #     (plots_path / "medial_axis").mkdir(parents=True, exist_ok=True)

    #     ch1 = tifffile.imread(
    #         f"/Users/borisgusev/Bioinformatics/master_project/data/nuclei_tracking/daf16_ch1_examples/Time{worm['Time']}_Point{worm['Point']}_GFP_Seq{worm['Seq']}.tiff"
    #     )

    #     ch1_mask = tifffile.imread(
    #         f"/Users/borisgusev/Bioinformatics/master_project/data/nuclei_tracking/daf16_ch1_masks_examples/Time{worm['Time']}_Point{worm['Point']}_GFP_Seq{worm['Seq']}.tiff"
    #     )
    #     ch2 = tifffile.imread(
    #         f"/Users/borisgusev/Bioinformatics/master_project/data/nuclei_tracking/daf16_ch2_examples/Time{worm['Time']}_Point{worm['Point']}_Channel470 nm,470 nm_Seq{worm['Seq']}.tiff"
    #     )

    #     ch2_mask = tifffile.imread(
    #         f"/Users/borisgusev/Bioinformatics/master_project/data/nuclei_tracking/daf16_ch2_stardist_masks_examples/Time{worm['Time']}_Point{worm['Point']}_Channel470 nm,470 nm_Seq{worm['Seq']}.tiff"
    #     )

    #     ch2_mask_centroids = []
    #     for slc1 in ch2_mask:
    #         labels = skimage.measure.label(slc1, connectivity=2)
    #         regions = skimage.measure.regionprops(labels)
    #         centroids = np.array([region.centroid for region in regions])
    #         ch2_mask_centroids.append(centroids)

    #     splines, us_per_pixel = generate_worm_backbone_spline(ch1)

    #     mask_zs, mask_ys, mask_xs = np.nonzero(ch1_mask)
    #     xlims = mask_xs.min(), mask_xs.max()
    #     ylims = mask_ys.max(), mask_ys.min()
    #     for i, (slc1, spline, slc2, nuclei_mask, centroids) in enumerate(
    #         zip(ch1, splines, ch2, ch2_mask, ch2_mask_centroids)
    #     ):
    #         # fig, ax = plt.subplots(1, 2, figsize=(12, 3), width_ratios=(1, 3))
    #         fig = plt.figure(figsize=(6, 9), dpi=600, layout="constrained")
    #         grid = GridSpec(3, 2, figure=fig)
    #         ax0 = fig.add_subplot(grid[0, 0])
    #         ax1 = fig.add_subplot(grid[0, 1])
    #         ax2 = fig.add_subplot(grid[1, 0])
    #         ax3 = fig.add_subplot(grid[1, 1])
    #         ax4 = fig.add_subplot(grid[2, :])

    #         mask = worm_mask(slc1)
    #         mask_ys, mask_xs = np.nonzero(mask)

    #         spline_ys, spline_xs = spline(np.linspace(0, 1, 1000)).T

    #         # xlims = mask_xs.min(), mask_xs.max()
    #         # ylims = mask_ys.max(), mask_ys.min()
    #         [(ax.set_xlim(*xlims), ax.set_ylim(*ylims)) for ax in (ax0, ax1, ax2, ax3)]
    #         ax4.set_xlim(0, 1)
    #         ax4.set_ylim(-0.02, 0.02)

    #         ax0.imshow(slc1)
    #         ax1.imshow(mask)
    #         ax2.imshow(slc2)
    #         ax3.imshow(nuclei_mask)

    #         if len(centroids) > 0:
    #             ys, xs = centroids.T
    #             [ax.scatter(xs, ys, s=2, c="red") for ax in (ax0, ax1, ax2, ax3)]

    #             worm_centroids = map_pixelspace_coords_to_worm_coords(
    #                 spline, centroids, us_per_pixel
    #             )
    #             ys, xs = worm_centroids.T
    #             ax4.scatter(xs, ys, s=0.5, c="blue")

    #         [
    #             ax.plot(
    #                 spline_xs,
    #                 spline_ys,
    #                 "-",
    #                 linewidth=0.5,
    #                 c="red",
    #             )
    #             for ax in (ax0, ax1, ax2, ax3)
    #         ]
    #         fig.savefig(plots_path / f"medial_axis/slice_{i}.png", dpi=600)
    #         plt.close()

    #     n_colors = len(list(filter(lambda x: len(x) > 0, ch2_mask_centroids)))
    #     colors = matplotlib.colormaps["gist_rainbow"](np.linspace(0, 1, n_colors))
    #     fig, ax = plt.subplots(1, 2, figsize=(12, 3), dpi=600)
    #     ax[0].set_title("Raw")
    #     ax[1].set_title("Worm Coord Transform")
    #     color_index = 0
    #     for spline, centroids in zip(splines, ch2_mask_centroids):
    #         if len(centroids) == 0:
    #             continue
    #         ys, xs = centroids.T
    #         ax[0].scatter(xs, ys, s=0.5, color=colors[color_index])

    #         ys, xs = map_pixelspace_coords_to_worm_coords(
    #             spline, centroids, us_per_pixel
    #         ).T
    #         ax[1].scatter(xs, ys, s=0.5, color=colors[color_index])
    #         color_index += 1

    #     fig.savefig(plots_path / "point_cloud.png")
    #     plt.close()

    #     print("done!")
