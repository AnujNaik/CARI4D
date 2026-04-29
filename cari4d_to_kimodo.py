"""
Convert CARI4D output .pth file to Kimodo constraints JSON format.
"""

import argparse
import json
import torch
import pytorch3d.transforms as T

# SMPL 24 joints -> SMPLXSkeleton22
SMPL_TO_SMPLX22 = [
    0, 1, 2, 3, 4, 5, 6,
    7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17,
    18, 19, 20, 21,
]


def load_pth(pth_path, source='pr'):
    d = torch.load(pth_path, map_location='cpu', weights_only=False)
    assert source in d, f"Source '{source}' not found. Available: {list(d.keys())}"
    return d[source]


# ✅ Proper 180° flip (fix upside-down issue ONLY)
def flip_root_upside_down(smpl_pose):
    root_aa = smpl_pose[:, :3]

    root_rot = T.axis_angle_to_matrix(root_aa)

    R = torch.tensor([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ], dtype=torch.float32)

    # ✅ FIX: multiply on the RIGHT
    root_rot_new = torch.matmul(root_rot, R)

    root_aa_new = T.matrix_to_axis_angle(root_rot_new)

    smpl_pose[:, :3] = root_aa_new
    return smpl_pose


def to_kimodo_constraints(data, indices, crop_start=0, crop_end=None):
    smpl_pose = data['smpl_pose']  # [T, 72]
    smpl_t = data['smpl_t']        # [T, 3]
    T_total = smpl_pose.shape[0]

    if crop_end is None:
        crop_end = T_total

    assert 0 <= crop_start < crop_end <= T_total

    # Crop
    smpl_pose = smpl_pose[crop_start:crop_end].clone()
    smpl_t = smpl_t[crop_start:crop_end].clone()
    T_crop = smpl_pose.shape[0]
    print(f"  Cropped to frames [{crop_start}, {crop_end}) -> {T_crop} frames")

    # 🔥 FIX: flip orientation ONLY (leave translation untouched)
    smpl_pose = flip_root_upside_down(smpl_pose)

    # Re-index keyframes
    rel_indices = [i - crop_start for i in indices]
    for orig, rel in zip(indices, rel_indices):
        assert 0 <= rel < T_crop, \
            f"Keyframe {orig} (relative: {rel}) is outside crop window"

    # Convert joints
    local_joints_rot_24 = smpl_pose[:, :72].reshape(T_crop, 24, 3)
    local_joints_rot = local_joints_rot_24[:, SMPL_TO_SMPLX22, :]

    # Canonicalize XZ (Y stays unchanged)
    origin_xz = smpl_t[0, [0, 2]].clone()
    smpl_t_canon = smpl_t.clone()
    smpl_t_canon[:, 0] -= origin_xz[0]
    smpl_t_canon[:, 2] -= origin_xz[1]

    local_joints_rot_sub = local_joints_rot[rel_indices]
    root_positions_sub = smpl_t_canon[rel_indices]
    smooth_root_2d_sub = smpl_t_canon[rel_indices][:, [0, 2]]

    return [{
        "type": "fullbody",
        "frame_indices": rel_indices,
        "local_joints_rot": local_joints_rot_sub.numpy().tolist(),
        "root_positions": root_positions_sub.numpy().tolist(),
        "smooth_root_2d": smooth_root_2d_sub.numpy().tolist(),
    }]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', required=True)
    parser.add_argument('--out', default='constraints.json')
    parser.add_argument('--source', default='pr', choices=['pr', 'gt', 'in'])
    parser.add_argument('--keyframes', type=int, nargs='+')
    parser.add_argument('--crop_start', type=int, default=0)
    parser.add_argument('--crop_end', type=int, default=None)
    parser.add_argument('--every', type=int, default=None)
    args = parser.parse_args()

    print(f"Loading {args.pth} (source='{args.source}')...")
    data = load_pth(args.pth, source=args.source)

    T_total = data['smpl_pose'].shape[0]
    print(f"  {T_total} total frames")

    crop_start = args.crop_start
    crop_end = args.crop_end if args.crop_end is not None else T_total
    window = crop_end - crop_start

    print(f"  Crop window: [{crop_start}, {crop_end}) = {window} frames ({window/30:.1f}s)")

    if args.keyframes:
        indices = sorted(args.keyframes)
        print(f"  Keyframes (original): {indices}")
        print(f"  Keyframes (relative): {[i - crop_start for i in indices]}")
    elif args.every:
        indices = list(range(crop_start, crop_end, args.every))
        print(f"  Using every {args.every} frames -> {len(indices)} keyframes")
    else:
        raise ValueError("Provide either --keyframes or --every")

    constraints = to_kimodo_constraints(
        data,
        indices,
        crop_start=crop_start,
        crop_end=crop_end
    )

    n_joints = len(constraints[0]['local_joints_rot'][0])
    print(f"  joint count: {n_joints} (should be 22)")
    print(f"  root frame 0 (canon): {constraints[0]['root_positions'][0]}")

    with open(args.out, 'w') as f:
        json.dump(constraints, f, indent=2)

    print(f"\nSaved: {args.out}")
    print(f"Relative keyframe indices: {constraints[0]['frame_indices']}")
    print(f"\nkimodo_gen \"A person picks up a box.\" --duration {window/30:.1f} --constraints {args.out}")


if __name__ == '__main__':
    main()