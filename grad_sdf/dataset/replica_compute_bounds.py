import argparse
import os.path as osp

from grad_sdf.dataset.replica import compute_bound


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to the Replica dataset")
    parser.add_argument(
        "--max-depth",
        type=float,
        default=-1.0,
        help="Maximum depth to consider, -1 means no limit",
    )
    args = parser.parse_args()

    for scene in ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]:
        data_path = osp.join(args.data_path, scene)
        if osp.exists(data_path):
            bound_min, bound_max = compute_bound(data_path, args.max_depth)
            bound_min = bound_min.tolist()
            bound_max = bound_max.tolist()
            print(f"Scene: {scene}, bound_min: {bound_min}, bound_max: {bound_max}")


if __name__ == "__main__":
    main()
