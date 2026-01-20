from dataclasses import dataclass

from numpy import ma
import torch
import torch.nn as nn

from grad_sdf.utils.config_abc import ConfigABC


@dataclass
class CriterionConfig(ConfigABC):
    boundary_loss_weight: float = 1.0
    boundary_loss_type: str = "L1"
    perturbation_loss_weight: float = 1.0
    perturbation_loss_type: str = "L1"
    eikonal_loss_surface_weight: float = 1.0
    eikonal_loss_space_weight: float = 1.0
    eikonal_loss_type: str = "L1"
    projection_loss_weight: float = 1.0
    projection_loss_type: str = "L1"
    heat_loss_weight: float = 0.0
    heat_loss_lambda: float = 0.0
    sign_loss_free_weight: float = 0.0
    sign_loss_occ_weight: float = 0.0
    sign_loss_temperature: float = 100.0
    reconstruction_loss_threshold: float = 0.1
    perturbation_loss_threshold: float = 0.2
    eikonal_loss_surface_threshold: float = 1.0
    eikonal_loss_space_threshold: float = 1.0
    projection_loss_threshold: float = 2.0


class Criterion(nn.Module):
    def __init__(
        self, cfg: CriterionConfig, frames_num: int, num_valid_rays: int, n_stratified: int, n_perturbed: int
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_stratified = n_stratified
        self.n_perturbed = n_perturbed
        # Keep tables on CPU to save GPU memory (only for logging)
        self.boundary_loss_table = torch.full((frames_num, num_valid_rays), -1.0, device="cpu")
        self.eikonal_loss_table = torch.full(
            (frames_num, num_valid_rays, n_stratified + n_perturbed + 1), -1.0, device="cpu"
        )
        self.perturbation_loss_table = torch.full((frames_num, num_valid_rays, n_perturbed), -1.0, device="cpu")
        self.projection_loss_table = torch.full((frames_num, num_valid_rays, n_stratified), -1.0, device="cpu")

        if self.cfg.boundary_loss_type == "L1":
            self.boundary_loss_fn = nn.L1Loss(reduction="none")
        elif self.cfg.boundary_loss_type == "L2":
            self.boundary_loss_fn = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Unknown boundary loss type: {self.cfg.boundary_loss_type}")

        if self.cfg.perturbation_loss_type == "L1":
            self.perturbation_loss_fn = nn.L1Loss(reduction="none")
        elif self.cfg.perturbation_loss_type == "L2":
            self.perturbation_loss_fn = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Unknown perturbation loss type: {self.cfg.perturbation_loss_type}")

        if self.cfg.eikonal_loss_type == "L1":
            self.eikonal_loss_fn = nn.L1Loss(reduction="none")
        elif self.cfg.eikonal_loss_type == "L2":
            self.eikonal_loss_fn = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Unknown eikonal loss type: {self.cfg.eikonal_loss_type}")

        if self.cfg.projection_loss_type == "L1":
            self.projection_loss_fn = nn.L1Loss(reduction="none")
        elif self.cfg.projection_loss_type == "L2":
            self.projection_loss_fn = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Unknown projection loss type: {self.cfg.projection_loss_type}")

    def forward(
        self,
        pred_sdf: torch.Tensor,
        pred_grad: torch.Tensor,
        gt_sdf_perturb: torch.Tensor,
        gt_sdf_stratified: torch.Tensor,
        frames_idx: int,
    ):
        loss = 0
        loss_dict = {}

        boundary_loss, mask_boundary_loss = self.get_boundary_loss(pred_sdf, frames_idx)
        perturbation_loss, mask_perturbation_loss = self.get_perturbation_loss(
            pred_sdf[:, self.n_stratified : self.n_stratified + self.n_perturbed],
            gt_sdf_perturb,
            frames_idx,
        )

        grad_norm = None

        if self.cfg.eikonal_loss_surface_weight > 0 or self.cfg.eikonal_loss_space_weight > 0:
            grad_norm = torch.norm(pred_grad, dim=-1)  # (num_valid_rays, total_samples)
            eikonal_loss_surface, eikonal_loss_space, eikonal_loss_surface_mask, eikonal_loss_space_mask = (
                self.get_eikonal_loss(grad_norm, frames_idx)
            )

        if self.cfg.projection_loss_weight > 0:
            projection_loss, projection_loss_mask = self.get_projection_loss(
                pred_sdf[:, : self.n_stratified],
                gt_sdf_stratified,
                frames_idx,
            )

        if self.cfg.heat_loss_weight > 0:
            if grad_norm is None:
                grad_norm = torch.norm(pred_grad, dim=-1)
            heat_loss = self.get_heat_loss(pred_sdf, grad_norm)

        # mask_total = (
        #     mask_boundary_loss
        #     & mask_perturbation_loss
        #     & eikonal_loss_surface_mask
        #     & eikonal_loss_space_mask
        #     & projection_loss_mask
        # )
        loss_boundary = boundary_loss[mask_boundary_loss].mean()
        loss_perturbation = perturbation_loss[mask_perturbation_loss].mean()
        loss_eikonal_surface = eikonal_loss_surface[eikonal_loss_surface_mask].mean()
        loss_eikonal_space = eikonal_loss_space[eikonal_loss_space_mask].mean()
        loss_projection = projection_loss[projection_loss_mask].mean()
        loss = (
            self.cfg.boundary_loss_weight * loss_boundary
            + self.cfg.perturbation_loss_weight * loss_perturbation
            + self.cfg.eikonal_loss_surface_weight * loss_eikonal_surface
            + self.cfg.eikonal_loss_space_weight * loss_eikonal_space
            + self.cfg.projection_loss_weight * loss_projection
        )
        loss_dict["boundary_loss"] = loss_boundary.item()
        loss_dict["perturbation_loss"] = loss_perturbation.item()
        loss_dict["eikonal_loss_surface"] = loss_eikonal_surface.item()
        loss_dict["eikonal_loss_space"] = loss_eikonal_space.item()
        loss_dict["projection_loss"] = loss_projection.item()

        loss_dict["total_loss"] = loss.item()
        return loss, loss_dict

    def get_boundary_loss(self, pred_sdf: torch.Tensor, frames_idx: int):
        pred_sdf_surface = pred_sdf[:, -1]
        boundary_loss = self.boundary_loss_fn(pred_sdf_surface, torch.zeros_like(pred_sdf_surface))
        self.boundary_loss_table[frames_idx, :] = boundary_loss.cpu().detach()
        if self.cfg.reconstruction_loss_threshold > 0:
            mask_boundary_loss = boundary_loss <= self.cfg.reconstruction_loss_threshold
            return boundary_loss, mask_boundary_loss
        else:
            return boundary_loss, torch.ones_like(boundary_loss, dtype=torch.bool)

    def get_perturbation_loss(self, pred_sdf_perturb: torch.Tensor, gt_sdf_perturb: torch.Tensor, frames_idx: int):
        perturbation_loss = self.perturbation_loss_fn(pred_sdf_perturb, gt_sdf_perturb)
        self.perturbation_loss_table[frames_idx, :, :] = perturbation_loss.cpu().detach()
        mask_perturbation_loss = (perturbation_loss <= self.cfg.perturbation_loss_threshold).all(
            dim=1
        )  # (num_valid_rays,)
        if self.cfg.reconstruction_loss_threshold > 0:
            mask_perturbation_loss = perturbation_loss <= self.cfg.reconstruction_loss_threshold
            return perturbation_loss, mask_perturbation_loss
        else:
            return perturbation_loss, torch.ones_like(perturbation_loss, dtype=torch.bool)

    def get_eikonal_loss(self, grad_norm: torch.Tensor, frames_idx: int):
        grad_norm_surface = grad_norm[:, self.n_stratified :]  # surface & perturbation
        grad_norm_space = grad_norm[:, : self.n_stratified]  # free space
        eikonal_loss_surface = self.eikonal_loss_fn(grad_norm_surface, torch.ones_like(grad_norm_surface))
        eikonal_loss_surface_mask = (eikonal_loss_surface <= self.cfg.eikonal_loss_surface_threshold).all(
            dim=1
        )  # (num_valid_rays,)
        eikonal_loss_space = self.eikonal_loss_fn(grad_norm_space, torch.ones_like(grad_norm_space))
        eikonal_loss_space_mask = (eikonal_loss_space <= self.cfg.eikonal_loss_space_threshold).all(
            dim=1
        )  # (num_valid_rays,)
        self.eikonal_loss_table[frames_idx, :, self.n_stratified :] = eikonal_loss_surface.cpu().detach()
        self.eikonal_loss_table[frames_idx, :, : self.n_stratified] = eikonal_loss_space.cpu().detach()
        if self.cfg.reconstruction_loss_threshold > 0:
            mask_eikonal_loss_surface = eikonal_loss_surface <= self.cfg.reconstruction_loss_threshold
            mask_eikonal_loss_space = eikonal_loss_space <= self.cfg.reconstruction_loss_threshold
            return eikonal_loss_surface, eikonal_loss_space, mask_eikonal_loss_surface, mask_eikonal_loss_space
        else:
            return (
                eikonal_loss_surface,
                eikonal_loss_space,
                torch.ones_like(eikonal_loss_surface, dtype=torch.bool),
                torch.ones_like(eikonal_loss_space, dtype=torch.bool),
            )

    def get_projection_loss(self, pred_sdf: torch.Tensor, gt_sdf_stratified: torch.Tensor, frames_idx: int):
        projection_loss = self.projection_loss_fn(pred_sdf, gt_sdf_stratified)
        projection_loss_mask = (projection_loss <= self.cfg.projection_loss_threshold).all(dim=1)  # (num_valid_rays,)
        self.projection_loss_table[frames_idx, :, :] = projection_loss.cpu().detach()
        if self.cfg.reconstruction_loss_threshold > 0:
            mask_projection_loss = projection_loss <= self.cfg.reconstruction_loss_threshold
            return projection_loss, mask_projection_loss
        else:
            return projection_loss, torch.ones_like(projection_loss, dtype=torch.bool)

    def get_sign_loss(
        self,
        positive_sdf_mask: torch.Tensor,
        negative_sdf_mask: torch.Tensor,
        pred_sdf: torch.Tensor,
    ):
        free_pred = pred_sdf[positive_sdf_mask.squeeze()]
        occ_pred = pred_sdf[negative_sdf_mask.squeeze()]
        sign_loss_free = (torch.tanh(self.cfg.sign_loss_temperature * free_pred) - 1).abs().mean()
        sign_loss_occ = (torch.tanh(self.cfg.sign_loss_temperature * occ_pred) + 1).abs().mean()
        return sign_loss_free, sign_loss_occ

    def get_heat_loss(self, pred_sdf: torch.Tensor, grad_norm: torch.Tensor):
        pred_sdf = pred_sdf[:, : self.n_stratified]  # only consider free space samples
        grad_norm = grad_norm[:, : self.n_stratified]
        heat = torch.exp(-self.cfg.heat_loss_lambda * pred_sdf.abs()).unsqueeze(1)
        heat_loss = 0.5 * heat**2 * (grad_norm**2 + 1)
        return heat_loss

    def visualize_loss_table(self, n_frames: int = 3):
        """
        Visualize loss distributions by randomly sampling n_frames.
        Each frame has different points, so we collect all valid loss values
        from the selected frames and plot their distribution.

        Args:
            n_frames: Number of frames to randomly sample for statistics
        """
        import random
        import numpy as np
        import matplotlib.pyplot as plt
        import io
        import PIL.Image

        # Randomly select n frame indices from available frames
        total_frames = self.boundary_loss_table.shape[0]
        n_frames = min(n_frames, total_frames)  # Don't exceed available frames
        selected_frame_indices = random.sample(range(total_frames), n_frames)

        # Collect loss values from selected frames
        # Use OrderedDict to maintain consistent order
        from collections import OrderedDict

        losses_to_plot = OrderedDict()

        # boundary_loss: (num_valid_rays,) per frame
        boundary_loss_list = []
        for frame_idx in selected_frame_indices:
            frame_loss = self.boundary_loss_table[frame_idx].numpy()
            # Filter out uninitialized values (-1.0)
            valid_loss = frame_loss[frame_loss >= 0]
            if len(valid_loss) > 0:
                boundary_loss_list.append(valid_loss)
        if len(boundary_loss_list) > 0:
            boundary_loss = np.concatenate(boundary_loss_list)
            losses_to_plot["boundary_loss"] = boundary_loss

        # perturbation_loss: (num_valid_rays, n_perturbed) per frame
        perturbation_loss_list = []
        for frame_idx in selected_frame_indices:
            frame_loss = self.perturbation_loss_table[frame_idx].numpy()
            valid_loss = frame_loss[frame_loss >= 0]
            if len(valid_loss) > 0:
                perturbation_loss_list.append(valid_loss)
        if len(perturbation_loss_list) > 0:
            perturbation_loss = np.concatenate(perturbation_loss_list)
            losses_to_plot["perturbation_loss"] = perturbation_loss

        # eikonal_loss: split into surface and space
        # eikonal_loss_table shape: (frames_num, num_valid_rays, n_stratified+n_perturbed+1)
        # [:, :, :n_stratified] -> space (free space)
        # [:, :, n_stratified:] -> surface (surface & perturbation)
        eikonal_loss_space_list = []
        eikonal_loss_surface_list = []
        for frame_idx in selected_frame_indices:
            # Space loss (free space samples)
            frame_loss_space = self.eikonal_loss_table[frame_idx, :, : self.n_stratified].numpy()
            valid_loss_space = frame_loss_space[frame_loss_space >= 0]
            if len(valid_loss_space) > 0:
                eikonal_loss_space_list.append(valid_loss_space)

            # Surface loss (surface & perturbation samples)
            frame_loss_surface = self.eikonal_loss_table[frame_idx, :, self.n_stratified :].numpy()
            valid_loss_surface = frame_loss_surface[frame_loss_surface >= 0]
            if len(valid_loss_surface) > 0:
                eikonal_loss_surface_list.append(valid_loss_surface)

        if len(eikonal_loss_space_list) > 0:
            eikonal_loss_space = np.concatenate(eikonal_loss_space_list)
            losses_to_plot["eikonal_loss_space"] = eikonal_loss_space

        if len(eikonal_loss_surface_list) > 0:
            eikonal_loss_surface = np.concatenate(eikonal_loss_surface_list)
            losses_to_plot["eikonal_loss_surface"] = eikonal_loss_surface

        # projection_loss: (num_valid_rays, n_stratified) per frame
        projection_loss_list = []
        for frame_idx in selected_frame_indices:
            frame_loss = self.projection_loss_table[frame_idx].numpy()
            valid_loss = frame_loss[frame_loss >= 0]
            if len(valid_loss) > 0:
                projection_loss_list.append(valid_loss)
        if len(projection_loss_list) > 0:
            projection_loss = np.concatenate(projection_loss_list)
            losses_to_plot["projection_loss"] = projection_loss

        # Handle case where no valid data exists
        if len(losses_to_plot) == 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f"No valid loss data in {n_frames} sampled frames", ha="center", va="center", fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
        else:
            # Plot each loss distribution as a subplot in one row
            num_losses = len(losses_to_plot)
            fig, axs = plt.subplots(1, num_losses, figsize=(4 * num_losses, 4))

            if num_losses == 1:
                axs = [axs]

            for ax, (loss_name, loss_data) in zip(axs, losses_to_plot.items()):
                ax.hist(loss_data, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
                ax.set_title(f"{loss_name}\n({len(loss_data)} points, {n_frames} frames)", fontsize=10)
                ax.set_xlabel("Loss Value", fontsize=9)
                ax.set_ylabel("Count", fontsize=9)
                ax.tick_params(axis="both", which="major", labelsize=8)
                # Add statistics text
                mean_val = loss_data.mean()
                std_val = loss_data.std()
                median_val = np.median(loss_data)
                ax.text(
                    0.95,
                    0.95,
                    f"Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMedian: {median_val:.4f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)

        # Return RGB array
        rgb = np.array(PIL.Image.open(buf).convert("RGB"))
        return rgb
