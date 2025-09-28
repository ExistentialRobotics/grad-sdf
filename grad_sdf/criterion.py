from dataclasses import dataclass

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


class Criterion(nn.Module):
    def __init__(self, cfg: CriterionConfig, n_stratified: int, n_perturbed: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_stratified = n_stratified
        self.n_perturbed = n_perturbed

        if self.cfg.boundary_loss_type == "L1":
            self.boundary_loss_fn = nn.L1Loss(reduction="mean")
        elif self.cfg.boundary_loss_type == "L2":
            self.boundary_loss_fn = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Unknown boundary loss type: {self.cfg.boundary_loss_type}")

        if self.cfg.perturbation_loss_type == "L1":
            self.perturbation_loss_fn = nn.L1Loss(reduction="mean")
        elif self.cfg.perturbation_loss_type == "L2":
            self.perturbation_loss_fn = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Unknown perturbation loss type: {self.cfg.perturbation_loss_type}")

        if self.cfg.eikonal_loss_type == "L1":
            self.eikonal_loss_fn = nn.L1Loss(reduction="mean")
        elif self.cfg.eikonal_loss_type == "L2":
            self.eikonal_loss_fn = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Unknown eikonal loss type: {self.cfg.eikonal_loss_type}")

        if self.cfg.projection_loss_type == "L1":
            self.projection_loss_fn = nn.L1Loss(reduction="mean")
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
    ):
        loss = 0
        loss_dict = {}

        boundary_loss = self.get_boundary_loss(pred_sdf)
        perturbation_loss = self.get_perturbation_loss(
            pred_sdf[:, self.n_stratified : self.n_stratified + self.n_perturbed],
            gt_sdf_perturb,
        )
        loss += self.cfg.boundary_loss_weight * boundary_loss + self.cfg.perturbation_loss_weight * perturbation_loss
        loss_dict["boundary_loss"] = boundary_loss.item()
        loss_dict["perturbation_loss"] = perturbation_loss.item()

        grad_norm = None

        if self.cfg.eikonal_loss_surface_weight > 0 or self.cfg.eikonal_loss_space_weight > 0:
            grad_norm = torch.norm(pred_grad, dim=-1)  # (num_valid_rays, total_samples)
            eikonal_loss_surface, eikonal_loss_space = self.get_eikonal_loss(grad_norm)
            loss += (
                self.cfg.eikonal_loss_surface_weight * eikonal_loss_surface
                + self.cfg.eikonal_loss_space_weight * eikonal_loss_space
            )
            loss_dict["eikonal_loss_surface"] = eikonal_loss_surface.item()
            loss_dict["eikonal_loss_space"] = eikonal_loss_space.item()

        if self.cfg.projection_loss_weight > 0:
            projection_loss = self.get_projection_loss(
                pred_sdf[:, : self.n_stratified],
                gt_sdf_stratified,
            )
            loss += self.cfg.projection_loss_weight * projection_loss
            loss_dict["projection_loss"] = projection_loss.item()

        if self.cfg.heat_loss_weight > 0:
            if grad_norm is None:
                grad_norm = torch.norm(pred_grad, dim=-1)
            heat_loss = self.get_heat_loss(pred_sdf, grad_norm)
            loss += self.cfg.heat_loss_weight * heat_loss
            loss_dict["heat_loss"] = heat_loss.item()

        loss_dict["total_loss"] = loss.item()
        return loss, loss_dict

    def get_boundary_loss(self, pred_sdf: torch.Tensor):
        pred_sdf_surface = pred_sdf[:, -1]
        boundary_loss = self.boundary_loss_fn(pred_sdf_surface, torch.zeros_like(pred_sdf_surface))
        return boundary_loss

    def get_perturbation_loss(self, pred_sdf_perturb: torch.Tensor, gt_sdf_perturb: torch.Tensor):
        perturbation_loss = self.perturbation_loss_fn(pred_sdf_perturb, gt_sdf_perturb)
        return perturbation_loss

    def get_eikonal_loss(self, grad_norm: torch.Tensor):
        grad_norm_surface = grad_norm[:, self.n_stratified :]  # surface & perturbation
        grad_norm_space = grad_norm[:, : self.n_stratified]  # free space
        eikonal_loss_surface = self.eikonal_loss_fn(grad_norm_surface, torch.ones_like(grad_norm_surface))
        eikonal_loss_space = self.eikonal_loss_fn(grad_norm_space, torch.ones_like(grad_norm_space))
        return eikonal_loss_surface, eikonal_loss_space

    def get_projection_loss(self, pred_sdf: torch.Tensor, gt_sdf_stratified: torch.Tensor):
        return self.projection_loss_fn(pred_sdf, gt_sdf_stratified)

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
        heat_loss = (0.5 * heat**2 * (grad_norm**2 + 1)).mean()
        return heat_loss
