"""The official learned TriPAF-Net v2 architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from models.blocks import ConvNormAct, DownBlock, ResidualBlock, UpBlock


def haze_descriptor(
    rgb: torch.Tensor,
    dark: torch.Tensor,
    bright: torch.Tensor,
    sky: torch.Tensor,
) -> torch.Tensor:
    """Return [mean(D), mean(B), mean(M), mean(Y), std(Y)] per image."""

    luminance = 0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    return torch.cat(
        (
            dark.mean(dim=(-2, -1)),
            bright.mean(dim=(-2, -1)),
            sky.mean(dim=(-2, -1)),
            luminance.mean(dim=(-2, -1)),
            luminance.std(dim=(-2, -1), correction=0),
        ),
        dim=1,
    )


class HazeConditionedTriPAF(nn.Module):
    """Learn channel, spatial, context, and detail prior fusion at one scale."""

    def __init__(
        self, image_channels: int, prior_channels: int, adaptive: bool
    ) -> None:
        super().__init__()
        self.adaptive = bool(adaptive)
        self.image_proj = nn.Conv2d(image_channels, image_channels, 1, bias=False)
        self.prior_proj = nn.Conv2d(prior_channels, image_channels, 1, bias=False)
        hidden = max(8, image_channels // 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(image_channels * 2, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, image_channels, 1),
        )
        self.spatial_gate = nn.Conv2d(image_channels * 2 + 1, 1, 5, padding=2)
        self.context_gate = nn.Sequential(
            nn.Linear(6, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, image_channels),
        )
        self.detail_gate = nn.Sequential(
            nn.Conv2d(image_channels * 3 + 2, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )
        self.out = nn.Sequential(
            ConvNormAct(image_channels * 3, image_channels),
            nn.Conv2d(image_channels, image_channels, 1, bias=False),
        )
        self.scale = nn.Parameter(torch.tensor(0.10))
        self.detail_scale = nn.Parameter(torch.tensor(0.05))
        # Nonzero initialization makes all adaptive terms trainable from step 1.
        self.adaptivity_scale = nn.Parameter(torch.tensor(0.20))

    def forward(
        self,
        image: torch.Tensor,
        prior: torch.Tensor,
        sky: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_feature = self.image_proj(image)
        prior_feature = self.prior_proj(prior)
        sky = functional.interpolate(
            sky,
            image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        detail = image_feature - functional.avg_pool2d(image_feature, 3, 1, 1)
        detail_energy = detail.abs().mean(1, keepdim=True)
        joined = torch.cat((image_feature, prior_feature), dim=1)

        if self.adaptive:
            context_map = self.context_gate(context).unsqueeze(-1).unsqueeze(-1)
            context_map = context_map.expand_as(image_feature)
            amplitude = torch.tanh(self.adaptivity_scale)
            gate_terms = (
                self.channel_gate(joined)
                + self.spatial_gate(torch.cat((joined, sky), dim=1))
                + context_map
            )
            prior_gate = 0.5 + 0.5 * amplitude * torch.tanh(gate_terms)
            detail_input = torch.cat((joined, sky, detail_energy, context_map), dim=1)
            detail_gate = 0.5 + 0.5 * amplitude * torch.tanh(
                self.detail_gate(detail_input)
            )
        else:
            prior_gate = torch.full_like(image_feature, 0.5)
            detail_gate = torch.full_like(detail_energy, 0.5)

        mixed = image_feature + prior_gate * (prior_feature - image_feature)
        correction = self.out(torch.cat((image_feature, prior_feature, mixed), dim=1))
        fused = image + torch.tanh(self.scale) * correction
        fused = fused + torch.tanh(self.detail_scale) * detail_gate * detail
        return fused, prior_gate, detail_gate


class TriPAFNetV2(nn.Module):
    """Adaptive TriPAF-Net v2 with a matched fixed-gate ablation."""

    def __init__(
        self,
        base_channels: int = 24,
        adaptive_fusion: bool = True,
        residual_scale: float = 0.5,
        t_min: float = 0.08,
    ) -> None:
        super().__init__()
        self.base_channels = int(base_channels)
        self.adaptive_fusion = bool(adaptive_fusion)
        self.residual_scale = float(residual_scale)
        self.t_min = float(t_min)

        c0, c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 12,
        )
        p0, p1, p2, p3, p4 = (
            max(8, c0 // 2),
            max(8, c1 // 2),
            c2 // 2,
            c3 // 2,
            c4 // 2,
        )
        self.image_stem = nn.Sequential(ConvNormAct(3, c0), ResidualBlock(c0))
        self.image_down = nn.ModuleList(
            (DownBlock(c0, c1), DownBlock(c1, c2), DownBlock(c2, c3), DownBlock(c3, c4))
        )
        self.prior_stem = nn.Sequential(ConvNormAct(3, p0), ResidualBlock(p0))
        self.prior_down = nn.ModuleList(
            (DownBlock(p0, p1), DownBlock(p1, p2), DownBlock(p2, p3), DownBlock(p3, p4))
        )
        channels = ((c0, p0), (c1, p1), (c2, p2), (c3, p3), (c4, p4))
        self.fusions = nn.ModuleList(
            HazeConditionedTriPAF(image_channels, prior_channels, self.adaptive_fusion)
            for image_channels, prior_channels in channels
        )
        self.bottleneck = nn.Sequential(ResidualBlock(c4), ResidualBlock(c4))
        self.decoder = nn.ModuleList(
            (
                UpBlock(c4, c3, c3),
                UpBlock(c3, c2, c2),
                UpBlock(c2, c1, c1),
                UpBlock(c1, c0, c0),
            )
        )
        self.output_refine = nn.Sequential(ConvNormAct(c0 + 3, c0), ResidualBlock(c0))
        self.residual_head = nn.Conv2d(c0, 3, 3, padding=1)
        self.transmission_head = nn.Conv2d(c0, 1, 3, padding=1)
        self.blend_head = nn.Conv2d(c0, 1, 3, padding=1)
        self.edge_blend_scale = nn.Parameter(torch.tensor(0.0))
        atmospheric_hidden = max(16, c4 // 4)
        self.atmospheric_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c4, atmospheric_hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(atmospheric_hidden, 3, 1),
        )
        descriptor_hidden = max(8, c0 // 2)
        self.severity_head = nn.Sequential(
            nn.Linear(5, descriptor_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(descriptor_hidden, 1),
        )
        self.final_fusion_head = nn.Sequential(
            nn.Linear(c0 + 6, max(16, c0)),
            nn.SiLU(inplace=True),
            nn.Linear(max(16, c0), 3),
        )

    def _prior_reconstruction(
        self,
        rgb: torch.Tensor,
        dark: torch.Tensor,
        bright: torch.Tensor,
        sky: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return auxiliary DCP/BCP/sky physical evidence."""

        height, width = rgb.shape[-2:]
        count = max(1, int(height * width * 0.001))
        flat_rgb = rgb.flatten(2)
        dark_indices = dark.flatten(2).topk(count, dim=2).indices.expand(-1, 3, -1)
        bright_indices = bright.flatten(2).topk(count, dim=2).indices.expand(-1, 3, -1)
        atmospheric_dcp = flat_rgb.gather(2, dark_indices).mean(2).clamp(0.05, 0.98)
        atmospheric_bcp = flat_rgb.gather(2, bright_indices).mean(2).clamp(0.05, 0.98)
        dcp_floor = (
            atmospheric_dcp.min(1, keepdim=True).values[..., None, None].clamp_min(0.05)
        )
        bcp_ceiling = atmospheric_bcp.max(1, keepdim=True).values[..., None, None]
        transmission_dcp = 1.0 - 0.88 * dark / dcp_floor
        transmission_bcp = (bright - bcp_ceiling).abs() / (1.0 - bcp_ceiling).clamp_min(
            0.05
        )
        sky_smooth = functional.avg_pool2d(sky, 15, 1, 7).clamp(0.0, 1.0)
        transmission = (
            (1.0 - sky_smooth) * transmission_dcp + sky_smooth * transmission_bcp
        ).clamp(self.t_min, 1.0)
        atmospheric = (1.0 - sky_smooth) * atmospheric_dcp[..., None, None]
        atmospheric = atmospheric + sky_smooth * atmospheric_bcp[..., None, None]
        reconstruction = ((rgb - atmospheric) / transmission + atmospheric).clamp(
            0.0, 1.0
        )
        return reconstruction, transmission, atmospheric

    @staticmethod
    def _stable_inference_output(
        rgb: torch.Tensor,
        restoration: torch.Tensor,
        output_weights: torch.Tensor,
        sky: torch.Tensor,
    ) -> torch.Tensor:
        """Bound the trained correction without replacing the learned restoration."""

        learned_weights = output_weights[:, :2]
        learned_weights = learned_weights / learned_weights.sum(
            1, keepdim=True
        ).clamp_min(1e-6)
        learned_strength = 0.65 + 0.25 * learned_weights[:, 1, None, None, None]
        network_image = rgb + learned_strength * (restoration - rgb)

        residual_limit = 0.22
        residual = residual_limit * torch.tanh((network_image - rgb) / residual_limit)
        luminance_weights = rgb.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
        luminance_residual = (residual * luminance_weights).sum(1, keepdim=True)
        chroma_residual = residual - luminance_residual

        sky_smooth = functional.avg_pool2d(sky, 31, stride=1, padding=15).clamp(
            0.0, 1.0
        )
        luminance_strength = 1.0 - 0.10 * sky_smooth
        chroma_strength = 0.55 * (1.0 - 0.60 * sky_smooth)
        input_detail = rgb - functional.avg_pool2d(rgb, 5, stride=1, padding=2)
        edge_energy = input_detail.abs().mean(1, keepdim=True)
        edge_confidence = 1.0 / (
            1.0 + 24.0 * functional.avg_pool2d(edge_energy, 5, 1, 2)
        )
        stable_residual = luminance_strength * luminance_residual
        stable_residual = stable_residual + chroma_strength * chroma_residual
        stable = rgb + edge_confidence * stable_residual

        output_detail = stable - functional.avg_pool2d(stable, 5, stride=1, padding=2)
        stable = stable + 0.40 * (input_detail - output_detail)
        return stable.clamp(0.0, 1.0)

    @staticmethod
    def _reflect_box_blur(tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
        height, width = tensor.shape[-2:]
        kernel_size = min(kernel_size, 2 * min(height, width) - 1)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        if kernel_size <= 1:
            return tensor
        radius = kernel_size // 2
        padded = functional.pad(
            tensor,
            (radius, radius, radius, radius),
            mode="reflect",
        )
        return functional.avg_pool2d(padded, kernel_size, stride=1)

    @classmethod
    def _model_conditioned_color_contrast_refinement(
        cls,
        image: torch.Tensor,
        sky: torch.Tensor,
        severity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply learned-model-conditioned chroma and contrast refinement."""

        luminance_weights = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
        luminance = (image * luminance_weights).sum(1, keepdim=True)
        chroma = image - luminance
        chroma_magnitude = chroma.square().mean(1, keepdim=True).add(1e-8).sqrt()

        local_mean = cls._reflect_box_blur(luminance, 31)
        detail = image - cls._reflect_box_blur(image, 5)
        edge_energy = detail.abs().mean(1, keepdim=True)
        edge_gate = edge_energy / (edge_energy + 0.025)
        severity_map = severity[:, :, None, None]

        sky_smooth = cls._reflect_box_blur(sky, 31).clamp(0.0, 1.0)
        vertical = torch.linspace(
            1.0,
            0.0,
            image.shape[-2],
            device=image.device,
            dtype=image.dtype,
        ).view(1, 1, -1, 1)
        upper_region = ((vertical - 0.30) / 0.70).clamp(0.0, 1.0)
        smooth_region = (1.0 - edge_gate).square()
        bright_region = torch.sigmoid(12.0 * (local_mean - 0.30))
        sky_fallback = upper_region * smooth_region * bright_region
        sky_protection = torch.maximum(sky_smooth, sky_fallback).clamp(0.0, 1.0)
        non_sky = 1.0 - sky_protection

        # Strengthen structure primarily where the image contains reliable edges.
        # Sky/fog protection still gates both adjustments to avoid clipped horizons.
        contrast_strength = non_sky * (0.18 + 0.12 * edge_gate)
        contrast_strength = contrast_strength * (0.85 + 0.15 * severity_map)
        brightness = 0.045 * severity_map * non_sky * (1.0 - luminance)
        refined_luminance = luminance + contrast_strength * (luminance - local_mean)
        refined_luminance = (refined_luminance + brightness).clamp(0.0, 1.0)

        # Fog suppresses the weak chroma of distant traffic targets long before
        # it removes their luminance edges.  Compare chroma with its local
        # neighbourhood so the additional gain is limited to colour outliers,
        # while grey buildings, asphalt and the sky remain neutral.
        local_chroma = cls._reflect_box_blur(chroma_magnitude, 21)
        chroma_excess = (chroma_magnitude - local_chroma).clamp_min(0.0)
        chroma_presence = (chroma_magnitude - 0.005).clamp_min(0.0)
        target_color_gate = (chroma_presence / (chroma_presence + 0.010)) * (
            chroma_excess / (chroma_excess + 0.006)
        )

        # A distant vehicle can retain a weak colour direction without enough
        # absolute saturation to pass the local-outlier test above.  The three
        # opponent axes detect that residual cue without favouring red over
        # green or blue; edge support rejects broad colour casts.
        opponent = image - (image.sum(1, keepdim=True) - image) * 0.5
        dominant_opponent = opponent.amax(dim=1, keepdim=True).clamp_min(0.0)
        opponent_gate = (dominant_opponent - 0.004).clamp_min(0.0)
        opponent_gate = opponent_gate / (opponent_gate + 0.008)
        target_color_gate = torch.maximum(
            target_color_gate,
            opponent_gate * (0.35 + 0.65 * edge_gate),
        )

        # Warm residuals are especially important for brake lights, vehicle
        # paint and warning signs.  This is a hue-preserving chroma gain: a
        # neutral pixel has a zero gate and cannot acquire a red cast.
        red_opponent = (
            image[:, 0:1] - 0.5 * (image[:, 1:2] + image[:, 2:3])
        ).clamp_min(0.0)
        warm_gate = (red_opponent - 0.003).clamp_min(0.0)
        warm_gate = warm_gate / (warm_gate + 0.006)
        warm_gate = warm_gate * (0.30 + 0.70 * edge_gate)

        requested_alpha = 1.0 + non_sky * (
            0.08
            + 0.08 * severity_map
            + 0.04 * edge_gate
            + 0.55 * target_color_gate
            + 0.30 * warm_gate
        )
        requested_extra = (requested_alpha - 1.0) * chroma_magnitude
        chroma_headroom = (0.32 - chroma_magnitude).clamp_min(0.0)
        accepted_extra = torch.minimum(requested_extra, chroma_headroom)
        safe_alpha = 1.0 + accepted_extra / chroma_magnitude.clamp_min(1e-4)

        refined = refined_luminance + safe_alpha * chroma
        return (
            refined.clamp(0.0, 1.0),
            safe_alpha,
            contrast_strength,
            sky_protection,
            target_color_gate,
        )

    @staticmethod
    def _color_statistics(
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
        luminance = (image * weights).sum(1, keepdim=True)
        chroma = image - luminance
        mean_rgb = image.mean(dim=(-2, -1))
        mean_chroma = chroma.square().mean(1, keepdim=True).add(1e-8).sqrt()
        channel_min = image.amin(dim=(-2, -1))
        channel_max = image.amax(dim=(-2, -1))
        return mean_rgb, channel_min, channel_max, mean_chroma.mean(dim=(-2, -1))

    def forward(
        self,
        rgb: torch.Tensor,
        dark: torch.Tensor,
        bright: torch.Tensor,
        sky: torch.Tensor,
        return_aux: bool = True,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        descriptor = haze_descriptor(rgb, dark, bright, sky)
        severity = torch.sigmoid(self.severity_head(descriptor))
        context = torch.cat((descriptor, severity), dim=1)
        image = self.image_stem(rgb)
        prior = self.prior_stem(torch.cat((dark, bright, sky), dim=1))
        skips: list[torch.Tensor] = []
        prior_gates: list[torch.Tensor] = []
        detail_gates: list[torch.Tensor] = []

        image, prior_gate, detail_gate = self.fusions[0](image, prior, sky, context)
        skips.append(image)
        prior_gates.append(prior_gate)
        detail_gates.append(detail_gate)
        for index, (image_down, prior_down) in enumerate(
            zip(self.image_down, self.prior_down), 1
        ):
            image = image_down(image)
            prior = prior_down(prior)
            image, prior_gate, detail_gate = self.fusions[index](
                image, prior, sky, context
            )
            skips.append(image)
            prior_gates.append(prior_gate)
            detail_gates.append(detail_gate)

        bottleneck = self.bottleneck(skips[-1])
        decoded = bottleneck
        for decoder, skip in zip(self.decoder, reversed(skips[:-1])):
            decoded = decoder(decoded, skip)
        decoded = self.output_refine(torch.cat((decoded, rgb), dim=1))
        residual = torch.tanh(self.residual_head(decoded))
        direct = (rgb + self.residual_scale * residual).clamp(0.0, 1.0)
        transmission = self.t_min + (1.0 - self.t_min) * torch.sigmoid(
            self.transmission_head(decoded)
        )
        atmospheric = torch.sigmoid(self.atmospheric_head(bottleneck))
        physical = ((rgb - atmospheric) / transmission + atmospheric).clamp(0.0, 1.0)
        edge_energy = (
            (rgb - functional.avg_pool2d(rgb, 3, 1, 1)).abs().mean(1, keepdim=True)
        )
        blend = torch.sigmoid(
            self.blend_head(decoded) + self.edge_blend_scale * edge_energy
        )
        restoration = (blend * direct + (1.0 - blend) * physical).clamp(0.0, 1.0)
        prior_image, prior_transmission, prior_atmospheric = self._prior_reconstruction(
            rgb, dark, bright, sky
        )
        pooled_decoder = functional.adaptive_avg_pool2d(decoded, 1).flatten(1)
        output_weights = torch.softmax(
            self.final_fusion_head(torch.cat((pooled_decoder, context), 1)), 1
        )
        restored = output_weights[:, 0].view(-1, 1, 1, 1) * rgb
        restored = restored + output_weights[:, 1].view(-1, 1, 1, 1) * restoration
        restored = restored + output_weights[:, 2].view(-1, 1, 1, 1) * prior_image
        restored = restored.clamp(0.0, 1.0)
        stable_before_color = self._stable_inference_output(
            rgb,
            restoration,
            output_weights,
            sky,
        )
        (
            stable_image,
            color_alpha,
            color_contrast,
            color_sky_protection,
            color_target_gate,
        ) = self._model_conditioned_color_contrast_refinement(
            stable_before_color, sky, severity
        )
        if not return_aux:
            return stable_image
        (
            color_pre_mean_rgb,
            color_pre_channel_min,
            color_pre_channel_max,
            color_pre_mean_chroma,
        ) = self._color_statistics(stable_before_color)
        (
            color_post_mean_rgb,
            color_post_channel_min,
            color_post_channel_max,
            color_post_mean_chroma,
        ) = self._color_statistics(stable_image)
        return {
            "image": restored,
            "stable_image": stable_image,
            "stable_before_color": stable_before_color,
            "restoration": restoration,
            "direct": direct,
            "physical": physical,
            "transmission": transmission,
            "atmospheric_light": atmospheric,
            "blend": blend,
            "residual": residual,
            "severity": severity,
            "haze_descriptor": descriptor,
            "output_weights": output_weights,
            "color_alpha_mean": color_alpha.mean(dim=(-2, -1)),
            "color_contrast_mean": color_contrast.mean(dim=(-2, -1)),
            "color_sky_protection_mean": color_sky_protection.mean(dim=(-2, -1)),
            "color_sky_protection": color_sky_protection,
            "color_target_gate_mean": color_target_gate.mean(dim=(-2, -1)),
            "color_pre_mean_rgb": color_pre_mean_rgb,
            "color_post_mean_rgb": color_post_mean_rgb,
            "color_pre_channel_min": color_pre_channel_min,
            "color_pre_channel_max": color_pre_channel_max,
            "color_post_channel_min": color_post_channel_min,
            "color_post_channel_max": color_post_channel_max,
            "color_pre_mean_chroma": color_pre_mean_chroma,
            "color_post_mean_chroma": color_post_mean_chroma,
            "prior_reconstruction": prior_image,
            "prior_transmission": prior_transmission,
            "prior_atmospheric_light": prior_atmospheric,
            "mean_prior_gate": torch.stack(
                [value.mean((1, 2, 3)) for value in prior_gates], 1
            ),
            "mean_detail_gate": torch.stack(
                [value.mean((1, 2, 3)) for value in detail_gates], 1
            ),
        }
