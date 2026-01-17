import torch
import numpy as np
import matplotlib

matplotlib.use("TkAgg")  # Use TkAgg backend for interactive display
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse

sys.path.append("/home/qihao/workplace/grad-sdf")

from grad_sdf.frame import LiDARFrame


def load_ply(ply_path, depth_min=0.6, depth_max=40.0):
    """Load PLY file"""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    depth = np.linalg.norm(points, axis=-1)
    mask = (depth >= depth_min) & (depth <= depth_max)
    points = points[mask]
    return torch.from_numpy(points).float()


def visualize_depth_image(depth_image, mask, save_path=None):
    """Visualize depth image"""
    # Calculate figure size to maintain 1024:128 aspect ratio
    # 1024:128 = 8:1 ratio
    fig, axes = plt.subplots(1, 3, figsize=(24, 4))

    # Depth image - maintain original aspect ratio
    im1 = axes[0].imshow(depth_image.numpy(), cmap="jet", aspect="equal")
    axes[0].set_title("Depth Image (1024x128)")
    axes[0].set_xlabel("Width (Azimuth 360°)")
    axes[0].set_ylabel("Height (Elevation 90°)")
    plt.colorbar(im1, ax=axes[0], label="Depth (m)", fraction=0.046)

    # Valid mask - maintain original aspect ratio
    axes[1].imshow(mask.numpy(), cmap="gray", aspect="equal")
    axes[1].set_title("Valid Mask (1024x128)")
    axes[1].set_xlabel("Width (Azimuth 360°)")
    axes[1].set_ylabel("Height (Elevation 90°)")

    # Statistics
    valid_pixels = mask.sum().item()
    total_pixels = mask.numel()
    min_depth = depth_image[mask].min().item() if valid_pixels > 0 else 0
    max_depth = depth_image[mask].max().item() if valid_pixels > 0 else 0
    mean_depth = depth_image[mask].mean().item() if valid_pixels > 0 else 0

    stats_text = f"""Statistics:
Total pixels: {total_pixels}
Valid pixels: {valid_pixels}
Coverage: {valid_pixels/total_pixels*100:.2f}%

Depth Range:
Min: {min_depth:.2f} m
Max: {max_depth:.2f} m
Mean: {mean_depth:.2f} m
"""
    axes[2].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment="center", fontproperties={"family": "monospace"})
    axes[2].axis("off")
    axes[2].set_title("Statistics")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Saved to: {save_path}")
        plt.close()
    else:
        plt.show(block=False)  # Non-blocking display
        plt.pause(0.001)  # Small pause to allow window to render


def visualize_noise_filter_comparison(depth_before, mask_before, depth_after, mask_after, save_path=None):
    """Visualize noise filter before/after comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(24, 8))

    # Before filtering - Depth image
    im1 = axes[0, 0].imshow(depth_before.cpu().numpy(), cmap="jet", aspect="equal")
    axes[0, 0].set_title("Before Filter - Depth Image")
    axes[0, 0].set_xlabel("Width (Azimuth 360°)")
    axes[0, 0].set_ylabel("Height (Elevation 90°)")
    plt.colorbar(im1, ax=axes[0, 0], label="Depth (m)", fraction=0.046)

    # Before filtering - Mask
    axes[0, 1].imshow(mask_before.cpu().numpy(), cmap="gray", aspect="equal")
    axes[0, 1].set_title("Before Filter - Valid Mask")
    axes[0, 1].set_xlabel("Width (Azimuth 360°)")
    axes[0, 1].set_ylabel("Height (Elevation 90°)")

    # Before filtering - Statistics
    valid_before = mask_before.sum().item()
    total_pixels = mask_before.numel()
    stats_before = f"""Before Filter Stats:
Total pixels: {total_pixels}
Valid pixels: {valid_before}
Coverage: {valid_before/total_pixels*100:.2f}%

Depth Range:
Min: {depth_before[mask_before].min().item():.2f} m
Max: {depth_before[mask_before].max().item():.2f} m
Mean: {depth_before[mask_before].mean().item():.2f} m
"""
    axes[0, 2].text(
        0.1, 0.5, stats_before, fontsize=11, verticalalignment="center", fontproperties={"family": "monospace"}
    )
    axes[0, 2].axis("off")

    # After filtering - Depth image
    im2 = axes[1, 0].imshow(depth_after.cpu().numpy(), cmap="jet", aspect="equal")
    axes[1, 0].set_title("After Filter - Depth Image")
    axes[1, 0].set_xlabel("Width (Azimuth 360°)")
    axes[1, 0].set_ylabel("Height (Elevation 90°)")
    plt.colorbar(im2, ax=axes[1, 0], label="Depth (m)", fraction=0.046)

    # After filtering - Mask
    axes[1, 1].imshow(mask_after.cpu().numpy(), cmap="gray", aspect="equal")
    axes[1, 1].set_title("After Filter - Valid Mask")
    axes[1, 1].set_xlabel("Width (Azimuth 360°)")
    axes[1, 1].set_ylabel("Height (Elevation 90°)")

    # After filtering - Statistics
    valid_after = mask_after.sum().item()
    removed = valid_before - valid_after
    stats_after = f"""After Filter Stats:
Total pixels: {total_pixels}
Valid pixels: {valid_after}
Coverage: {valid_after/total_pixels*100:.2f}%

Depth Range:
Min: {depth_after[mask_after].min().item():.2f} m
Max: {depth_after[mask_after].max().item():.2f} m
Mean: {depth_after[mask_after].mean().item():.2f} m

Filter Effect:
Removed pixels: {removed}
Retention rate: {valid_after/valid_before*100:.2f}%
"""
    axes[1, 2].text(
        0.1, 0.5, stats_after, fontsize=11, verticalalignment="center", fontproperties={"family": "monospace"}
    )
    axes[1, 2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Comparison saved to: {save_path}")
        plt.close()
    else:
        plt.show(block=False)
        plt.pause(0.001)


def test_single_frame(
    ply_path, frame_id=0, apply_noise_filter=False, noise_threshold=0.5, min_blob_size=30, width=1024, height=128
):
    """Test single point cloud frame
    Args:
        ply_path: Path to point cloud file
        frame_id: Frame ID
        apply_noise_filter: Whether to apply noise filtering
        noise_threshold: Depth jump threshold (meters)
        min_blob_size: Minimum blob size in pixels
        width: Depth image width
        height: Depth image height
    """
    print(f"\nProcessing point cloud: {ply_path}")

    # Load point cloud
    pointcloud = load_ply(ply_path)
    print(f"Number of points: {pointcloud.shape[0]}")

    # Create LiDARFrame
    offset = torch.zeros(3)
    ref_pose = torch.eye(4)

    lidar_frame = LiDARFrame(fid=frame_id, pointcloud=pointcloud, offset=offset, ref_pose=ref_pose)

    # Convert to depth image
    print("Converting to depth image...")
    depth_image_before, mask_before, point_to_pixel, pixel_to_point = lidar_frame.pcd_to_image(
        width=width, height=height
    )

    if apply_noise_filter:
        print(f"Applying noise filter (threshold={noise_threshold}m, min_blob_size={min_blob_size} pixels)...")
        # Apply noise filter
        lidar_frame.apply_noise_filter(noise_filter_threshold=noise_threshold, min_blob_size=min_blob_size)

        # Regenerate depth image
        depth_image_after, mask_after, _, _ = lidar_frame.pcd_to_image(width=width, height=height)

        return depth_image_before, mask_before, depth_image_after, mask_after, point_to_pixel, pixel_to_point
    else:
        return depth_image_before, mask_before, None, None, point_to_pixel, pixel_to_point


def test_all_frames(
    data_dir,
    output_dir=None,
    interval=1,
    apply_noise_filter=False,
    noise_threshold=0.5,
    min_blob_size=30,
    width=1024,
    height=128,
):
    """Iterate through all point cloud files and visualize
    Args:
        data_dir: Point cloud data directory
        output_dir: Output directory, if None display interactively
        interval: Process every N frames
        apply_noise_filter: Whether to apply noise filtering
        noise_threshold: Depth jump threshold (meters)
        min_blob_size: Minimum blob size in pixels
        width: Depth image width
        height: Depth image height
    """
    data_path = Path(data_dir)
    ply_files = sorted(data_path.glob("*.ply"))

    print(f"Found {len(ply_files)} point cloud files")

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {output_path}")

    for i, ply_file in enumerate(ply_files):
        # Process only selected frames
        if i % interval != 0 and i != 0 and i != len(ply_files) - 1:
            continue

        print(f"\nProgress: {i+1}/{len(ply_files)}")

        try:
            result = test_single_frame(
                ply_file,
                frame_id=i,
                apply_noise_filter=apply_noise_filter,
                noise_threshold=noise_threshold,
                min_blob_size=min_blob_size,
                width=width,
                height=height,
            )

            if apply_noise_filter:
                depth_before, mask_before, depth_after, mask_after, _, _ = result

                if output_dir:
                    save_path = output_path / f"noise_filter_comparison_{ply_file.stem}.png"
                    visualize_noise_filter_comparison(
                        depth_before, mask_before, depth_after, mask_after, save_path=save_path
                    )
                else:
                    visualize_noise_filter_comparison(depth_before, mask_before, depth_after, mask_after)
                    input("Press Enter to continue to next frame (or Ctrl+C to exit)...")
                    plt.close("all")
            else:
                depth_image, mask, _, _, _, _ = result

                if output_dir:
                    save_path = output_path / f"depth_{ply_file.stem}.png"
                    visualize_depth_image(depth_image, mask, save_path=save_path)
                else:
                    visualize_depth_image(depth_image, mask)
                    input("Press Enter to continue to next frame (or Ctrl+C to exit)...")
                    plt.close("all")

        except Exception as e:
            print(f"Error processing {ply_file}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\nCompleted! Processed {len(ply_files)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LiDAR point clouds to depth images with optional noise filtering"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/qihao/workplace/grad-sdf/data/newercollege-lidar/ply",
        help="Directory containing PLY files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (if None, display interactively)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=100,
        help="Process every N frames (default: 100)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Depth image width (default: 1024)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Depth image height (default: 128)",
    )
    parser.add_argument(
        "--apply_noise_filter",
        action="store_true",
        help="Apply noise filtering",
    )
    parser.add_argument(
        "--noise_threshold",
        type=float,
        default=0.5,
        help="Depth jump threshold in meters (default: 0.5)",
    )
    parser.add_argument(
        "--min_blob_size",
        type=int,
        default=30,
        help="Minimum blob size in pixels, smaller blobs will be removed (default: 30)",
    )

    args = parser.parse_args()

    # Display settings
    print("=" * 60)
    print(f"LiDAR Point Cloud to Depth Image{' with Noise Filtering' if args.apply_noise_filter else ''}")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir if args.output_dir else 'Interactive display'}")
    print(f"Interval: Every {args.interval} frame(s)")
    print(f"Resolution: {args.width}x{args.height}")
    if args.apply_noise_filter:
        print(f"Noise filtering: Enabled")
        print(f"  - Depth jump threshold: {args.noise_threshold} m")
        print(f"  - Minimum blob size: {args.min_blob_size} pixels")
    else:
        print(f"Noise filtering: Disabled")
    print("=" * 60)
    if not args.output_dir:
        print("Press Enter to view each frame, or Ctrl+C to exit")
    print()

    test_all_frames(
        args.data_dir,
        output_dir=args.output_dir,
        interval=args.interval,
        apply_noise_filter=args.apply_noise_filter,
        noise_threshold=args.noise_threshold,
        min_blob_size=args.min_blob_size,
        width=args.width,
        height=args.height,
    )
