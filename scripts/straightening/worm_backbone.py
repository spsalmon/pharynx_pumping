import itertools

import cv2
import networkx as nx
import numpy as np
import skan
import skimage
from scipy.spatial import distance


def worm_mask(img):
    mask = img > skimage.filters.threshold_yen(img)
    mask = cv2.medianBlur(skimage.util.img_as_ubyte(mask), ksize=15)
    return mask


def main_branch_paths(skeleton: skan.Skeleton):
    branch_data = skan.summarize(skeleton, find_main_branch=True)
    # get the largest subskeleton
    subskeletons = branch_data.groupby(by="skeleton-id", as_index=True)
    largest_subskeleton = subskeletons.get_group(
        subskeletons["branch-distance"].sum().idxmax()
    )
    # get the nodes/edges for skeleton's branch-graph
    main_branch_edges = largest_subskeleton.loc[
        largest_subskeleton.main == True, ["node-id-src", "node-id-dst"]
    ].to_numpy()
    main_branch_nodes, counts = np.unique(
        main_branch_edges.flatten(), return_counts=True
    )

    main_branch_graph = nx.Graph()
    main_branch_graph.add_edges_from(main_branch_edges)
    end1, end2 = main_branch_nodes[np.nonzero(counts == 1)]
    # order the nodes
    main_branch_nodes = np.asarray(nx.shortest_path(main_branch_graph, end1, end2))

    PATH_AXIS = 0
    NODE_AXIS = 1
    ALL_PATHS = np.s_[:]
    # main_branch_mask has True if path is part of main branch
    # path is part of main branch if both of its nodes are in main_branch_nodes, hence check for sum(NODE_AXIS) == 2
    main_branch_mask = (
        skeleton.paths[ALL_PATHS, main_branch_nodes].sum(axis=NODE_AXIS) == 2
    )
    main_branch_paths = np.flatnonzero(main_branch_mask)
    # subset skeleton.paths with unordered main_branch_paths and the ordered main_branch_nodes to give something like the following:
    # 0 1 1 0 0
    # 1 1 0 0 0
    # 0 0 1 1 0
    # 0 0 0 1 1
    # argmax for each row will give the col-index for the first "1" in that row
    # argsorting the col-index gives the path order
    # i.e. read the 1s in matrix from left to right, recording the order of the rows
    # for the example above, the desired result is [1, 0, 2, 3], which is the row order that would reorder the matrix to:
    # 1 1 0 0 0
    # 0 1 1 0 0
    # 0 0 1 1 0
    # 0 0 0 1 1
    path_order = (
        skeleton.paths[np.ix_(main_branch_paths, main_branch_nodes)]
        .argmax(axis=NODE_AXIS)
        .argsort(axis=PATH_AXIS)
        .A1  # A1 converts np.matrix to 1D np.array
    )
    main_branch_paths = main_branch_paths[path_order]
    return main_branch_paths


def paths_to_coordinates(skeleton: skan.Skeleton, paths):
    # axis 0 -> points; axis 1 -> xy coords
    POINTS = 0

    path_coords = [skeleton.path_coordinates(path) for path in paths]
    ordered_path_coords = []
    for path1_coords, path2_coords in zip(path_coords[:-1], path_coords[1:]):
        path1_end = path1_coords[-1]
        path2_tip1, path2_tip2 = path2_coords[[0, -1]]
        # path1_end is indeed the end
        if np.all(path1_end == path2_tip1) or np.all(path1_end == path2_tip2):
            ordered_path_coords.append(path1_coords)
        # path1_end is actually the start, so flip
        else:
            flipped = np.flip(path1_coords, axis=POINTS)
            ordered_path_coords.append(flipped)

    last_path = path_coords[-1]
    last_path_start = last_path[0]
    if len(ordered_path_coords) == 0:
        ordered_path_coords.append(last_path)
    else:
        last_ordered_coord = ordered_path_coords[-1][-1]
        if np.all(last_ordered_coord == last_path_start):
            # if end and start align, append as is
            ordered_path_coords.append(last_path)
        else:
            # if end and start don't align, flip first then append
            ordered_path_coords.append(np.flip(last_path, axis=POINTS))

    ordered_path_coords = np.concatenate(ordered_path_coords, axis=POINTS)
    # coords need to be unique because splprep will throw an error otherwise
    _, uniq_indices = np.unique(ordered_path_coords, return_index=True, axis=POINTS)
    uniq_indices.sort()
    ordered_path_coords = ordered_path_coords[uniq_indices]
    return ordered_path_coords


def are_tips_flipped(tip_pair1, tip_pair2):
    pair_wise_dists = distance.cdist(tip_pair1, tip_pair2, metric="euclidean")
    unflipped_dist = np.sum(pair_wise_dists[[0, 1], [0, 1]])
    flipped_dist = np.sum(pair_wise_dists[[0, 1], [1, 0]])
    return flipped_dist < unflipped_dist


def generate_worm_backbone(img):
    mask = np.zeros_like(img)
    skeleton_img = np.zeros_like(img)
    for i, slc in enumerate(img):
        slc_mask = worm_mask(slc)
        mask[i] = slc_mask
        skeleton_img[i] = skimage.morphology.medial_axis(slc_mask)

    skeletons = [skan.Skeleton(skel_img) for skel_img in skeleton_img]
    slcs_main_branches = [main_branch_paths(skeleton) for skeleton in skeletons]
    backbones = [
        paths_to_coordinates(skeleton, slc_paths)
        for skeleton, slc_paths in zip(skeletons, slcs_main_branches)
    ]
    backbone_tips = [backbone[[0, -1]] for backbone in backbones]
    relative_flips = [
        are_tips_flipped(tips1, tips2)
        for tips1, tips2 in zip(backbone_tips[:-1], backbone_tips[1:])
    ]
    absolute_flips = list(
        itertools.accumulate(relative_flips, func=lambda x, y: x != y, initial=False)
    )
    backbones = [
        np.flip(backbone, axis=0) if is_flipped else backbone
        for backbone, is_flipped in zip(backbones, absolute_flips)
    ]
    return backbones
