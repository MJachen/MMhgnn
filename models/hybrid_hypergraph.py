from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .backbone3d import Light3DBackbone
from .hgnn import HGNNStack


ANATOMY_HYPEREDGES = [
    [0, 1, 2],  # core + boundary + peri_inner: local lesion evolution
    [1, 2, 3],  # boundary + peri_inner + peri_outer: outward extension
    [2, 3, 4],  # peri_inner + peri_outer + distal_normal: context calibration
    [0, 2, 4],  # core + peri_inner + distal_normal: lesion-background contrast
    [0, 1, 2, 3],  # core + boundary + peri_inner + peri_outer: overall lesion-context
]


def _group_norm(num_channels: int, num_groups: int) -> nn.GroupNorm:
    if num_channels <= 0:
        raise ValueError(f"GroupNorm channels must be positive, got {num_channels}.")
    if num_groups <= 0 or num_channels % num_groups != 0:
        raise ValueError(
            f"GroupNorm groups must be positive and divide channels exactly; "
            f"got groups={num_groups}, channels={num_channels}."
        )
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class AuxiliaryWTSegmentationBranch(nn.Module):
    """Lightweight training-time WT head over modality encoder feature maps."""

    def __init__(self, modalities: Sequence[str], encoder_channels: int, config: Dict):
        super().__init__()
        target = str(config.get("target", "wt")).lower()
        fusion = str(config.get("fusion", "masked_mean")).lower()
        projection_channels = int(config.get("projection_channels", 32))
        hidden_channels = [int(value) for value in config.get("hidden_channels", [32, 16])]
        output_channels = int(config.get("output_channels", 1))
        group_norm_groups = int(config.get("group_norm_groups", 8))

        if target != "wt":
            raise ValueError(f"Only Whole Tumor auxiliary supervision is supported, got target={target!r}.")
        if fusion != "masked_mean":
            raise ValueError(f"Only mask-aware mean fusion is supported, got fusion={fusion!r}.")
        if len(hidden_channels) != 2:
            raise ValueError("auxiliary_segmentation.hidden_channels must contain exactly two values.")
        if output_channels != 1:
            raise ValueError("Whole Tumor auxiliary segmentation requires output_channels=1.")

        self.modalities = list(modalities)
        self.projections = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Conv3d(encoder_channels, projection_channels, kernel_size=1, bias=False),
                    _group_norm(projection_channels, group_norm_groups),
                    nn.ReLU(inplace=True),
                )
                for modality in self.modalities
            }
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv3d(projection_channels, hidden_channels[0], kernel_size=3, padding=1, bias=False),
            _group_norm(hidden_channels[0], group_norm_groups),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels[0], hidden_channels[1], kernel_size=3, padding=1, bias=False),
            _group_norm(hidden_channels[1], group_norm_groups),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels[1], output_channels, kernel_size=1),
        )

    @staticmethod
    def masked_mean_fusion(projected_features: torch.Tensor, availability: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Fuse ``[B, M, C, d, h, w]`` features using a ``[B, M]`` availability mask."""
        if projected_features.ndim != 6:
            raise ValueError(f"Expected projected features with 6 dimensions, got {projected_features.shape}.")
        if availability.ndim != 2 or availability.shape[:2] != projected_features.shape[:2]:
            raise ValueError(
                f"Availability must have shape {tuple(projected_features.shape[:2])}, got {tuple(availability.shape)}."
            )
        availability = availability.to(device=projected_features.device, dtype=projected_features.dtype)
        counts = availability.sum(dim=1)
        invalid_cases = torch.nonzero(counts <= 0, as_tuple=False).flatten()
        if invalid_cases.numel() > 0:
            indices = invalid_cases.detach().cpu().tolist()
            raise ValueError(f"Each case needs at least one available modality; all-zero mask at batch indices {indices}.")
        mask = availability.view(availability.shape[0], availability.shape[1], 1, 1, 1, 1)
        numerator = (projected_features * mask).sum(dim=1)
        denominator = counts.view(-1, 1, 1, 1, 1)
        return numerator / (denominator + eps)

    def forward(
        self,
        spatial_feature_maps: Dict[str, torch.Tensor],
        availability: torch.Tensor,
        target_shape: Sequence[int],
    ) -> torch.Tensor:
        missing = [modality for modality in self.modalities if modality not in spatial_feature_maps]
        if missing:
            raise KeyError(f"Missing encoder feature maps for modalities: {missing}")
        projected = torch.stack(
            [self.projections[modality](spatial_feature_maps[modality]) for modality in self.modalities],
            dim=1,
        )
        if availability.ndim == 1:
            availability = availability.unsqueeze(0)
        fused = self.masked_mean_fusion(projected, availability)
        logits_low = self.segmentation_head(fused)
        if tuple(logits_low.shape[-3:]) == tuple(target_shape):
            return logits_low
        return F.interpolate(logits_low, size=tuple(target_shape), mode="trilinear", align_corners=False)


class HybridHypergraphClassifier(nn.Module):
    """Anatomy-only hypergraph classifier.

    The model keeps the existing two-step feature flow used by the project:
    1. pool per-modality features into five explicit anatomy/context nodes;
    2. run a fixed anatomy-only hypergraph over those five nodes.

    Prototype nodes and prototype hyperedges are intentionally removed for this
    baseline. The old ``knn`` branch key is kept only for API compatibility and
    no longer controls any prototype/KNN edge in this anatomy-only model.
    """

    def __init__(self, config: Dict):
        super().__init__()
        model_cfg = config["model"]
        data_cfg = config["data"]
        graph_cfg = config.get("graph", {})
        fusion_cfg = config.get("fusion", {})
        self.modalities = data_cfg["modalities"]
        self.mask_order = model_cfg.get("mask_order", ["t1", "t1ce", "t2", "flair"])
        self.modality_to_index = {name: idx for idx, name in enumerate(self.modalities)}
        self.roi_names = model_cfg["roi_names"]
        self.num_explicit_nodes = len(self.roi_names)
        self.num_prototype_nodes = 0
        self.total_nodes = self.num_explicit_nodes

        branches_cfg = model_cfg.get("branches", {})
        self.use_modal_edges = bool(branches_cfg.get("use_modal_edges", True))
        self.use_prior_edges = bool(branches_cfg.get("use_prior_edges", True))
        self.use_knn_edges = False
        self.use_prototype_nodes = bool(graph_cfg.get("use_prototype_nodes", False))
        self.use_prototype_edges = bool(graph_cfg.get("use_prototype_edges", False))
        self.use_mask_aware_classifier = bool(model_cfg.get("use_mask_aware_classifier", True))
        self.mask_head_type = str(model_cfg.get("mask_head_type", "bias"))
        self.mask_embed_dim = int(model_cfg.get("mask_embed_dim", 8))
        self.use_mask_aware_node_fusion = bool(fusion_cfg.get("use_mask_aware_node_fusion", True))
        self.use_node_type_embed = bool(fusion_cfg.get("use_node_type_embed", True))
        self.fusion_mask_embed_dim = int(fusion_cfg.get("mask_embed_dim", 8))
        self.node_type_embed_dim = int(fusion_cfg.get("node_type_embed_dim", 4))
        self.gating_hidden_dim = int(fusion_cfg.get("gating_hidden_dim", 32))
        self.no_t1ce_t1_penalty = float(fusion_cfg.get("no_t1ce_t1_penalty", 1.0))
        if self.use_prototype_nodes or self.use_prototype_edges:
            raise ValueError("Anatomy-only mode requires graph.use_prototype_nodes=false and graph.use_prototype_edges=false.")

        self.backbones = nn.ModuleDict({m: Light3DBackbone(1, tuple(model_cfg["backbone_channels"])) for m in self.modalities})
        feat_dim = next(iter(self.backbones.values())).out_channels
        auxiliary_cfg = model_cfg.get("auxiliary_segmentation", {})
        self.auxiliary_segmentation_enabled = bool(auxiliary_cfg.get("enabled", False))
        self.auxiliary_segmentation = (
            AuxiliaryWTSegmentationBranch(self.modalities, feat_dim, auxiliary_cfg)
            if self.auxiliary_segmentation_enabled
            else None
        )
        pooled_dim = feat_dim * 2
        self.explicit_projector = nn.Linear(pooled_dim, model_cfg["roi_hidden_dim"])
        self.shared_projector = nn.Linear(model_cfg["roi_hidden_dim"] * 2, model_cfg["roi_hidden_dim"])
        self.stage1_modal_encoder = HGNNStack(
            in_dim=model_cfg["roi_hidden_dim"],
            hidden_dim=model_cfg["roi_hidden_dim"],
            num_layers=int(model_cfg.get("stage1_num_layers", 1)),
            dropout=model_cfg["dropout"],
        )

        self.fusion_mask_encoder = nn.Sequential(
            nn.Linear(4, self.fusion_mask_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.fusion_mask_embed_dim, self.fusion_mask_embed_dim),
            nn.ReLU(inplace=True),
        )
        if self.use_node_type_embed:
            self.node_type_embed = nn.Embedding(self.num_explicit_nodes, self.node_type_embed_dim)
        else:
            self.node_type_embed = None
        fusion_input_dim = model_cfg["roi_hidden_dim"] + self.fusion_mask_embed_dim + (self.node_type_embed_dim if self.use_node_type_embed else 0)
        self.modality_gate = nn.Sequential(
            nn.Linear(fusion_input_dim, self.gating_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.gating_hidden_dim, 1),
        )

        hidden_dim = model_cfg["hgnn_hidden_dim"]
        self.graph_input_projector = nn.Linear(model_cfg["roi_hidden_dim"], hidden_dim)
        self.anatomy_hypergraph = HGNNStack(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=int(model_cfg.get("graph_num_layers", model_cfg.get("stage2_num_layers", 2))),
            dropout=model_cfg["dropout"],
        )
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, model_cfg["classifier_hidden_dim"]),
            nn.ReLU(inplace=True),
            nn.Dropout(model_cfg["dropout"]),
            nn.Linear(model_cfg["classifier_hidden_dim"], 2),
        )
        if self.use_mask_aware_classifier and self.mask_head_type != "bias":
            raise ValueError("Experiment 1 implements the minimal mask-aware bias head only: set mask_head_type: bias.")
        self.mask_encoder = nn.Sequential(
            nn.Linear(4, self.mask_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.mask_embed_dim, self.mask_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.mask_bias_head = nn.Linear(self.mask_embed_dim, 2)


    def get_classifier_info(self) -> Dict[str, object]:
        return {
            "use_mask_aware_classifier": bool(self.use_mask_aware_classifier),
            "mask_head_type": self.mask_head_type if self.use_mask_aware_classifier else "none",
            "mask_order": list(self.mask_order),
            "use_mask_aware_node_fusion": bool(self.use_mask_aware_node_fusion),
            "use_node_type_embed": bool(self.use_node_type_embed),
            "no_t1ce_t1_penalty": float(self.no_t1ce_t1_penalty),
        }

    def _mask_in_classifier_order(self, available_modalities: torch.Tensor) -> torch.Tensor:
        ordered = torch.zeros(len(self.mask_order), device=available_modalities.device, dtype=available_modalities.dtype)
        for order_idx, modality in enumerate(self.mask_order):
            src_idx = self.modality_to_index.get(modality)
            if src_idx is not None and src_idx < available_modalities.shape[0]:
                ordered[order_idx] = available_modalities[src_idx]
        return ordered.float()

    def _features_in_classifier_order(self, modality_features: List[torch.Tensor]) -> List[torch.Tensor]:
        zero_feature = torch.zeros_like(modality_features[0])
        ordered_features: List[torch.Tensor] = []
        for modality in self.mask_order:
            src_idx = self.modality_to_index.get(modality)
            if src_idx is None or src_idx >= len(modality_features):
                ordered_features.append(zero_feature.clone())
            else:
                ordered_features.append(modality_features[src_idx])
        return ordered_features

    def _apply_mask_aware_bias(self, base_logits: torch.Tensor, available_modalities: torch.Tensor):
        if not self.use_mask_aware_classifier:
            return base_logits, torch.zeros_like(base_logits), self._mask_in_classifier_order(available_modalities)
        mask = self._mask_in_classifier_order(available_modalities).to(base_logits.device)
        mask_embed = self.mask_encoder(mask.unsqueeze(0)).squeeze(0)
        mask_bias = self.mask_bias_head(mask_embed)
        return base_logits + mask_bias, mask_bias, mask

    def get_enabled_branches(self, branch_override: Optional[Dict[str, bool]] = None) -> Dict[str, bool]:
        enabled = {"modal": self.use_modal_edges, "prior": self.use_prior_edges, "knn": False}
        if branch_override:
            for key, value in branch_override.items():
                if key in {"modal", "prior"}:
                    enabled[key] = bool(value)
        enabled["knn"] = False
        return enabled

    def _masked_mean_max_pool(self, feat_map: torch.Tensor, roi_masks: torch.Tensor) -> torch.Tensor:
        _, c, d, h, w = feat_map.shape
        roi_resized = F.interpolate(roi_masks.unsqueeze(0), size=(d, h, w), mode="nearest").squeeze(0)
        flat_feats = feat_map.squeeze(0).view(c, -1)
        flat_masks = roi_resized.view(self.num_explicit_nodes, -1)
        pooled = []
        for idx in range(self.num_explicit_nodes):
            mask = flat_masks[idx] > 0.5
            if mask.sum() == 0:
                pooled.append(torch.zeros(c * 2, device=feat_map.device))
                continue
            masked_feats = flat_feats[:, mask]
            mean_feat = masked_feats.mean(dim=-1)
            max_feat = masked_feats.max(dim=-1).values
            pooled.append(torch.cat([mean_feat, max_feat], dim=0))
        return torch.stack(pooled, dim=0)

    def _edge_list_to_incidence(self, edges: List[List[int]], num_nodes: int, device: torch.device) -> torch.Tensor:
        if not edges:
            return torch.empty((num_nodes, 0), device=device)
        h = torch.zeros(num_nodes, len(edges), device=device)
        for edge_idx, edge in enumerate(edges):
            for node_idx in edge:
                h[node_idx, edge_idx] = 1.0
        return h

    def _build_local_modal_incidence(self, num_modality_nodes: int, device: torch.device) -> torch.Tensor:
        num_nodes = 1 + num_modality_nodes
        edges: List[List[int]] = []
        modality_indices = list(range(1, num_nodes))
        if num_modality_nodes >= 2:
            edges.append(modality_indices)
        if num_modality_nodes >= 1:
            edges.append([0] + modality_indices)
        if not edges:
            return torch.eye(num_nodes, device=device)
        return self._edge_list_to_incidence(edges, num_nodes, device)

    def _extract_explicit_modality_features(
        self,
        images: torch.Tensor,
        roi_masks: torch.Tensor,
        roi_valid: torch.Tensor,
        available_modalities: torch.Tensor,
        preserve_spatial: bool = False,
    ):
        modality_features = []
        active_feats = []
        spatial_feature_maps = {} if preserve_spatial else None
        for mod_idx, modality in enumerate(self.modalities):
            img = images[mod_idx : mod_idx + 1].unsqueeze(0)
            fmap = self.backbones[modality](img)
            if spatial_feature_maps is not None:
                spatial_feature_maps[modality] = fmap
            pooled = self._masked_mean_max_pool(fmap, roi_masks)
            pooled = self.explicit_projector(pooled) * roi_valid.unsqueeze(-1)
            if available_modalities[mod_idx] < 0.5:
                pooled = torch.zeros_like(pooled)
            else:
                active_feats.append(pooled)
            modality_features.append(pooled)

        if active_feats:
            stacked = torch.stack(active_feats, dim=0)
            mean_feat = stacked.mean(dim=0)
            max_feat = stacked.max(dim=0).values
            shared_init = self.shared_projector(torch.cat([mean_feat, max_feat], dim=-1)) * roi_valid.unsqueeze(-1)
        else:
            shared_init = torch.zeros(self.num_explicit_nodes, self.explicit_projector.out_features, device=images.device)
        return modality_features, shared_init, spatial_feature_maps

    def _stage1_modal_aggregate(self, modality_features: List[torch.Tensor], shared_init: torch.Tensor, available_modalities: torch.Tensor, roi_valid: torch.Tensor):
        if self.use_mask_aware_node_fusion:
            return self._mask_aware_node_fuse(modality_features, available_modalities, roi_valid)
        if not self.use_modal_edges:
            return shared_init * roi_valid.unsqueeze(-1), torch.zeros(self.num_explicit_nodes, len(self.mask_order), device=shared_init.device)
        device = shared_init.device
        active_mods = [idx for idx, flag in enumerate(available_modalities.tolist()) if flag > 0.5]
        aggregated = []
        legacy_gates = []
        if active_mods:
            gate_value = 1.0 / len(active_mods)
        else:
            gate_value = 0.0
        for roi_idx in range(self.num_explicit_nodes):
            shared_node = shared_init[roi_idx : roi_idx + 1]
            modality_nodes = [modality_features[m_idx][roi_idx : roi_idx + 1] for m_idx in active_mods]
            local_nodes = torch.cat([shared_node] + modality_nodes, dim=0) if modality_nodes else shared_node
            local_h = self._build_local_modal_incidence(len(modality_nodes), device)
            local_out = self.stage1_modal_encoder(local_nodes, local_h)
            aggregated.append(local_out[0] * roi_valid[roi_idx])
            gate_row = torch.zeros(len(self.mask_order), device=device)
            for active_idx in active_mods:
                modality_name = self.modalities[active_idx]
                if modality_name in self.mask_order:
                    gate_row[self.mask_order.index(modality_name)] = gate_value
            legacy_gates.append(gate_row * roi_valid[roi_idx])
        return torch.stack(aggregated, dim=0), torch.stack(legacy_gates, dim=0)


    def _mask_aware_node_fuse(self, modality_features: List[torch.Tensor], available_modalities: torch.Tensor, roi_valid: torch.Tensor):
        ordered_mask = self._mask_in_classifier_order(available_modalities).to(roi_valid.device)
        ordered_features = torch.stack(self._features_in_classifier_order(modality_features), dim=1)
        mask_embed = self.fusion_mask_encoder(ordered_mask.unsqueeze(0)).squeeze(0)
        mask_embed = mask_embed.unsqueeze(0).unsqueeze(0).expand(self.num_explicit_nodes, len(self.mask_order), -1)

        fusion_inputs = [ordered_features, mask_embed]
        if self.use_node_type_embed and self.node_type_embed is not None:
            node_ids = torch.arange(self.num_explicit_nodes, device=roi_valid.device)
            node_embed = self.node_type_embed(node_ids).unsqueeze(1).expand(-1, len(self.mask_order), -1)
            fusion_inputs.append(node_embed)
        gate_input = torch.cat(fusion_inputs, dim=-1)
        scores = self.modality_gate(gate_input).squeeze(-1)

        availability = ordered_mask.unsqueeze(0).expand(self.num_explicit_nodes, -1) > 0.5
        if ordered_mask[0] > 0.5 and ordered_mask[1] < 0.5:
            scores[:, 0] = scores[:, 0] - self.no_t1ce_t1_penalty
        scores = scores.masked_fill(~availability, -1e9)
        gates = torch.softmax(scores, dim=1)
        gates = gates * availability.float()
        gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-8)
        gates = gates * roi_valid.unsqueeze(-1)
        fused = torch.sum(ordered_features * gates.unsqueeze(-1), dim=1) * roi_valid.unsqueeze(-1)
        return fused, gates

    def _build_anatomy_incidence(self, enabled: Dict[str, bool], device: torch.device) -> torch.Tensor:
        if not enabled["prior"]:
            return torch.eye(self.num_explicit_nodes, device=device)
        return self._edge_list_to_incidence(ANATOMY_HYPEREDGES, self.num_explicit_nodes, device)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        roi_drop_mask: torch.Tensor | None = None,
        branch_override: Optional[Dict[str, bool]] = None,
        return_segmentation: bool | None = None,
    ) -> Dict[str, torch.Tensor]:
        images = batch["images"].float()
        roi_masks = batch["roi_masks"].float()
        roi_valid = batch["roi_valid"].float()
        available_modalities = batch["available_modalities"].float()
        batch_size = images.shape[0]
        enabled = self.get_enabled_branches(branch_override)
        compute_segmentation = self.auxiliary_segmentation_enabled and (
            True if return_segmentation is None else bool(return_segmentation)
        )

        logits_list = []
        prob_list = []
        roi_attention_list = []
        roi_feature_list = []
        stage_stats_list = []
        mask_bias_list = []
        classifier_mask_list = []
        modality_gate_list = []
        seg_logits_list = []

        for b_idx in range(batch_size):
            modality_features, shared_init, spatial_feature_maps = self._extract_explicit_modality_features(
                images[b_idx],
                roi_masks[b_idx],
                roi_valid[b_idx],
                available_modalities[b_idx],
                preserve_spatial=compute_segmentation,
            )
            explicit_nodes, modality_gates = self._stage1_modal_aggregate(modality_features, shared_init, available_modalities[b_idx], roi_valid[b_idx])

            if roi_drop_mask is not None:
                current_drop = roi_drop_mask[b_idx].to(explicit_nodes.device)
                explicit_nodes = explicit_nodes * current_drop.unsqueeze(-1)

            incidence = self._build_anatomy_incidence(enabled, explicit_nodes.device)
            projected_nodes = self.graph_input_projector(explicit_nodes)
            updated_nodes = self.anatomy_hypergraph(projected_nodes, incidence)

            attn_logits = self.attention_mlp(updated_nodes).squeeze(-1)
            node_attn = torch.softmax(attn_logits, dim=0)
            graph_repr = torch.sum(updated_nodes * node_attn.unsqueeze(-1), dim=0)
            base_logits = self.classifier(graph_repr)
            logits, mask_bias, classifier_mask = self._apply_mask_aware_bias(base_logits, available_modalities[b_idx])
            prob = torch.softmax(logits, dim=-1)[1]

            logits_list.append(logits)
            prob_list.append(prob)
            roi_attention_list.append(node_attn)
            roi_feature_list.append(explicit_nodes)
            mask_bias_list.append(mask_bias)
            classifier_mask_list.append(classifier_mask)
            modality_gate_list.append(modality_gates)
            if compute_segmentation:
                if self.auxiliary_segmentation is None or spatial_feature_maps is None:
                    raise RuntimeError("Auxiliary segmentation was requested but its modules are not initialized.")
                seg_logits_list.append(
                    self.auxiliary_segmentation(
                        spatial_feature_maps,
                        available_modalities[b_idx],
                        target_shape=images.shape[-3:],
                    ).squeeze(0)
                )
            stage_stats_list.append(torch.tensor([
                float(incidence.shape[1]),
                float(enabled["prior"]),
                0.0,
            ], device=explicit_nodes.device))

        output = {
            "logits": torch.stack(logits_list, dim=0),
            "prob": torch.stack(prob_list, dim=0),
            "roi_attention": torch.stack(roi_attention_list, dim=0),
            "node_attention_all": torch.stack(roi_attention_list, dim=0),
            "roi_shared_features": torch.stack(roi_feature_list, dim=0),
            "mask_bias": torch.stack(mask_bias_list, dim=0),
            "classifier_mask": torch.stack(classifier_mask_list, dim=0),
            "modality_gates": torch.stack(modality_gate_list, dim=0),
            "branch_enabled": enabled,
            "stage_stats": torch.stack(stage_stats_list, dim=0),
        }
        if seg_logits_list:
            output["seg_logits"] = torch.stack(seg_logits_list, dim=0)
        return output

    def load_checkpoint_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load baseline checkpoints strictly, allowing only new auxiliary keys to be absent."""
        if not self.auxiliary_segmentation_enabled:
            return self.load_state_dict(state_dict, strict=True)
        incompatible = self.load_state_dict(state_dict, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        non_auxiliary_missing = [
            key for key in incompatible.missing_keys if not key.startswith("auxiliary_segmentation.")
        ]
        if unexpected or non_auxiliary_missing:
            raise RuntimeError(
                "Checkpoint is incompatible beyond the optional auxiliary branch: "
                f"missing={non_auxiliary_missing}, unexpected={unexpected}."
            )
        return incompatible
