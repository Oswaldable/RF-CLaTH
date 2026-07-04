from __future__ import annotations

from typing import Dict, Optional

import torch


def _nested(cfg: Dict, path: tuple[str, ...], default=None):
    cur = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _float_value(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().float().mean().cpu().item())
    return float(value)


def _metric(metrics: Optional[Dict], key: str, default: float = 0.0) -> float:
    if not metrics:
        return float(default)
    return _float_value(metrics.get(key, default), default=default)


def _tensor_stats(values: torch.Tensor) -> tuple[float, float]:
    if values.numel() == 0:
        return 0.0, 0.0
    values = values.detach().float()
    mean = float(values.mean().cpu().item())
    std = float(values.std(unbiased=False).cpu().item()) if values.numel() > 1 else 0.0
    return mean, std


class AgenticTrainingController:
    """Outer-loop controller for agentic retrieval training.

    The controller is intentionally lightweight: it does not replace the inner
    optimizer and it does not introduce policy-gradient training. It turns the
    current planner/memory/hash state into explicit per-sample actions:
    slow/fast routing alpha, sample contribution omega, and retrieval budget.
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg.get("agentic", {})
        self.enabled = bool(self.cfg.get("enabled", False))
        self.log_style = str(self.cfg.get("log_style", "agent")).lower()

        policy_cfg = self.cfg.get("policy", {})
        source_cfg = self.cfg.get(
            "source_weights",
            cfg.get("agentic_contrastive", cfg.get("loss", {}).get("agentic_contrastive", {})).get(
                "source_weights",
                {},
            ),
        )
        self.route_enabled = bool(policy_cfg.get("route_enabled", True))
        self.sample_weight_enabled = bool(policy_cfg.get("sample_weight_enabled", True))
        self.use_routed_similarity = bool(policy_cfg.get("use_routed_similarity", True))
        self.update_feedback_graph = bool(policy_cfg.get("update_feedback_graph", True))
        self.alpha_default = float(policy_cfg.get("alpha_default", 0.5))
        self.alpha_kappa = float(policy_cfg.get("alpha_kappa", 6.0))
        self.alpha_bias = float(policy_cfg.get("alpha_bias", 0.0))
        self.trust_kappa = float(policy_cfg.get("trust_kappa", 8.0))
        self.trust_center = float(policy_cfg.get("trust_center", 0.02))
        self.omega_min = float(policy_cfg.get("omega_min", 0.5))
        self.omega_max = float(policy_cfg.get("omega_max", 1.5))
        self.omega_scale = float(policy_cfg.get("omega_scale", 0.5))
        self.cold_start_updates = max(1.0, float(policy_cfg.get("cold_start_updates", 2.0)))
        self.edge_slots = int(policy_cfg.get("edge_slots", 40))
        self.edge_posterior_momentum = float(policy_cfg.get("edge_posterior_momentum", 0.80))
        self.edge_reliability_momentum = float(policy_cfg.get("edge_reliability_momentum", 0.80))
        self.edge_decay_gamma = float(policy_cfg.get("edge_decay_gamma", 0.98))

        self.source_weight_view = float(source_cfg.get("view", 1.0))
        self.source_weight_batch = float(source_cfg.get("batch_neighbor", 0.75))
        self.source_weight_memory = float(source_cfg.get("memory_neighbor", 0.25))
        self.source_weight_arf = float(source_cfg.get("arf_planned", 0.25))
        self.source_weight_missed_bonus = float(source_cfg.get("arf_missed_bonus", 0.25))
        agentic_cfg = cfg.get("agentic_contrastive", cfg.get("loss", {}).get("agentic_contrastive", {}))
        self.hard_negative_weight = float(agentic_cfg.get("hard_negative_weight", policy_cfg.get("hard_negative_weight", 1.25)))
        self.top_r = int(_nested(cfg, ("retrieval_environment", "top_r"), _nested(cfg, ("planner", "top_m"), 20)))
        self.budget_min = max(1, int(policy_cfg.get("budget_min", max(1, self.top_r // 2))))
        self.budget_max = max(self.budget_min, int(policy_cfg.get("budget_max", self.top_r)))

        stop_cfg = self.cfg.get("stop_policy", {})
        self.stop_enabled = bool(stop_cfg.get("enabled", False))
        self.stop_min_epoch = int(stop_cfg.get("min_epoch", 40))
        self.stop_patience_windows = max(1, int(stop_cfg.get("patience_windows", 3)))
        self.stop_gain_beta = float(stop_cfg.get("gain_beta", 0.7))
        self.stop_min_gain = float(stop_cfg.get("min_gain", 1e-4))
        self.stop_min_memory_stability = float(stop_cfg.get("min_memory_stability", 0.95))
        self.stop_entropy_low = float(stop_cfg.get("gate_entropy_low", 0.05))
        self.stop_entropy_high = float(stop_cfg.get("gate_entropy_high", 0.98))
        self.stop_cost_weight = float(stop_cfg.get("cost_weight", 0.0))
        self.stop_min_utility = float(stop_cfg.get("min_utility", 0.0))
        self.stop_horizon_windows = max(1, int(stop_cfg.get("horizon_windows", self.stop_patience_windows)))

        self.last_map: Optional[float] = None
        self.gain_ema = 0.0
        self.stop_window_count = 0

    @property
    def logs_agent_style(self) -> bool:
        return self.enabled and self.log_style == "agent"

    @torch.no_grad()
    def act(
        self,
        outputs: Dict[str, torch.Tensor],
        planner_memory,
        graph_planner,
        sample_indices: torch.Tensor,
        epoch: int,
    ) -> Dict:
        device = outputs["u_a"].device
        batch_size = int(outputs["u_a"].shape[0])
        alpha = torch.full((batch_size,), self.alpha_default, device=device, dtype=torch.float32)
        omega = torch.ones(batch_size, device=device, dtype=torch.float32)
        budget = torch.full((batch_size,), self.budget_max, device=device, dtype=torch.float32)
        trust = torch.zeros(batch_size, device=device, dtype=torch.float32)
        ctx = {}
        memory_state = {}

        if self.enabled and planner_memory is not None and graph_planner is not None:
            ctx = graph_planner.action_context(planner_memory, sample_indices)
            p_s = ctx["planner_p_s_topm"].to(device=device, dtype=torch.float32)
            p_t = ctx["planner_p_t_topm"].to(device=device, dtype=torch.float32)
            p_final = ctx["planner_p_final_topm"].to(device=device, dtype=torch.float32)
            p_random = ctx["planner_p_random"].to(device=device, dtype=torch.float32)
            valid = ctx["planner_valid"].to(device=device, dtype=torch.float32)

            memory_device_indices = sample_indices.detach().long().to(planner_memory.device)
            update_count = planner_memory.update_count.index_select(0, memory_device_indices).to(device=device).float()
            cold_gate = (update_count / self.cold_start_updates).clamp(0.0, 1.0)
            margin = torch.nan_to_num(p_final - p_random, nan=0.0, posinf=0.0, neginf=0.0)
            trust = torch.sigmoid(self.trust_kappa * (margin - self.trust_center)) * valid * cold_gate

            if self.route_enabled:
                route_logit = self.alpha_kappa * torch.nan_to_num(p_s - p_t, nan=0.0) + self.alpha_bias
                routed_alpha = torch.sigmoid(route_logit).clamp(0.02, 0.98)
                alpha = torch.where(valid.bool(), routed_alpha, alpha)
            if self.sample_weight_enabled:
                omega = (1.0 + self.omega_scale * (trust - 0.5)).clamp(self.omega_min, self.omega_max)
            budget = (self.budget_min + trust * float(self.budget_max - self.budget_min)).round()
            budget = budget.clamp(self.budget_min, self.budget_max)
            if hasattr(planner_memory, "edge_summary"):
                memory_state = planner_memory.edge_summary(sample_indices)

            planner_memory.update_agent_actions(sample_indices, alpha, omega, epoch=epoch)

        alpha_mean, alpha_std = _tensor_stats(alpha)
        omega_mean, omega_std = _tensor_stats(omega)
        trust_mean, _ = _tensor_stats(trust)
        budget_mean, budget_std = _tensor_stats(budget)
        action_top_r = max(1, int(round(budget_mean)))
        entropy = self._binary_entropy(alpha)
        source_scale = float(max(0.0, min(1.0, trust_mean if self.enabled else 1.0)))
        source_weights = {
            "view": self.source_weight_view,
            "batch_neighbor": self.source_weight_batch,
            "memory_neighbor": self.source_weight_memory * max(omega_mean, 0.0),
            "arf_planned": self.source_weight_arf * source_scale,
            "arf_missed_bonus": self.source_weight_missed_bonus * source_scale,
            "hard_negative_weight": 1.0 + (self.hard_negative_weight - 1.0) * source_scale,
        }

        action = {
            "enabled": self.enabled,
            "use_routed_similarity": self.enabled and self.use_routed_similarity,
            "sample_weight_enabled": self.enabled and self.sample_weight_enabled,
            "update_feedback_graph": self.enabled and self.update_feedback_graph,
            "edge_slots": self.edge_slots,
            "edge_posterior_momentum": self.edge_posterior_momentum,
            "edge_reliability_momentum": self.edge_reliability_momentum,
            "edge_decay_gamma": self.edge_decay_gamma,
            "alpha": alpha,
            "omega": omega,
            "budget": budget,
            "top_r": action_top_r,
            "trust": trust,
            "state": {
                "h_s": outputs.get("h_s_a", outputs.get("h_s")).detach() if torch.is_tensor(outputs.get("h_s_a", outputs.get("h_s"))) else None,
                "h_f": outputs.get("h_f_a").detach() if torch.is_tensor(outputs.get("h_f_a")) else None,
                "u_s": outputs.get("u_s_a").detach() if torch.is_tensor(outputs.get("u_s_a")) else None,
                "u_f": outputs.get("u_f_a").detach() if torch.is_tensor(outputs.get("u_f_a")) else None,
                "m": memory_state,
                "q": {key: value.detach() if torch.is_tensor(value) else value for key, value in ctx.items()},
                "e": {
                    "epoch": int(epoch),
                    "top_r_min": self.budget_min,
                    "top_r_max": self.budget_max,
                },
            },
            "source_weights": source_weights,
            "metrics": {
                "agent_route_alpha": alpha_mean,
                "agent_route_alpha_std": alpha_std,
                "agent_route_entropy": entropy,
                "agent_sample_omega": omega_mean,
                "agent_sample_omega_std": omega_std,
                "agent_trust": trust_mean,
                "agent_budget": budget_mean,
                "agent_budget_std": budget_std,
                "agent_top_r": float(action_top_r),
                "agent_obs_sem": _float_value(ctx.get("planner_p_s_topm", 0.0)),
                "agent_obs_dyn": _float_value(ctx.get("planner_p_t_topm", 0.0)),
                "agent_obs_final": _float_value(ctx.get("planner_p_final_topm", 0.0)),
                "agent_obs_random": _float_value(ctx.get("planner_p_random", 0.0)),
                "agent_obs_valid": _float_value(ctx.get("planner_valid", 0.0)),
            },
        }
        return action

    def _binary_entropy(self, alpha: torch.Tensor) -> float:
        if alpha.numel() == 0:
            return 0.0
        p = alpha.detach().float().clamp(1e-6, 1.0 - 1e-6)
        entropy = -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p))
        return float(entropy.mean().cpu().item())

    def memory_summary(self, planner_memory, sample_indices: torch.Tensor) -> Dict[str, float]:
        if planner_memory is None or not hasattr(planner_memory, "edge_summary"):
            return {}
        return planner_memory.edge_summary(sample_indices)

    def format_step(
        self,
        epoch: int,
        step: int,
        total_steps: int,
        losses: Dict,
        hash_metrics: Dict[str, float],
        action: Dict,
        planner_metrics: Optional[Dict[str, float]],
        memory_metrics: Optional[Dict[str, float]],
    ) -> str:
        action_metrics = action.get("metrics", {}) if action else {}
        source = action.get("source_weights", {}) if action else {}
        return (
            f"agent_loop epoch={epoch} step={step}/{total_steps} "
            f"observe(valid={_metric(action_metrics, 'agent_obs_valid'):.3f} "
            f"sem={_metric(action_metrics, 'agent_obs_sem'):.4f} "
            f"dyn={_metric(action_metrics, 'agent_obs_dyn'):.4f} "
            f"final={_metric(action_metrics, 'agent_obs_final'):.4f} "
            f"hash_entropy={_metric(hash_metrics, 'metric_bit_entropy'):.3f} "
            f"bit_use={_metric(hash_metrics, 'metric_bit_usage'):.3f}) "
            f"action(alpha={_metric(action_metrics, 'agent_route_alpha'):.3f}"
            f"+/-{_metric(action_metrics, 'agent_route_alpha_std'):.3f} "
            f"H={_metric(action_metrics, 'agent_route_entropy'):.3f} "
            f"omega={_metric(action_metrics, 'agent_sample_omega'):.3f} "
            f"budget={_metric(action_metrics, 'agent_budget'):.0f} "
            f"top_r={_metric(losses, 'metric_agentic_trace_top_r', _metric(action_metrics, 'agent_top_r')):.0f}) "
            f"feedback(planned={_metric(losses, 'metric_agentic_pos_arf'):.1f} "
            f"actual_overlap={_metric(losses, 'metric_arf_actual_overlap'):.3f} "
            f"missed={_metric(losses, 'metric_arf_missed_ratio'):.3f} "
            f"false={_metric(losses, 'metric_arf_false_ratio'):.3f} "
            f"hpos={_metric(losses, 'metric_agentic_hard_positive_count'):.1f} "
            f"hneg={_metric(losses, 'metric_agentic_hard_negative_count'):.1f} "
            f"edge={_metric(losses, 'metric_agentic_edge_factor'):.3f}) "
            f"memory(valid={_metric(planner_metrics, 'planner_valid_final', _metric(action_metrics, 'agent_obs_valid')):.3f} "
            f"edge_p={_metric(memory_metrics, 'memory_edge_posterior'):.3f} "
            f"edge_r={_metric(memory_metrics, 'memory_edge_reliability'):.3f} "
            f"stable={_metric(memory_metrics, 'memory_edge_stability'):.3f}) "
            f"adapt(loss={_metric(losses, 'loss'):.4f} "
            f"aucl={_metric(losses, 'component_agentic_contrastive', _metric(losses, 'loss_semantic')):.4f} "
            f"hash={_metric(losses, 'loss_hash'):.4f} "
            f"q={_metric(losses, 'component_quant'):.4f} "
            f"bal={_metric(losses, 'component_bit_balance'):.4f} "
            f"K={_metric(losses, 'metric_agentic_semantic_bits'):.0f}+{_metric(losses, 'metric_agentic_temporal_bits'):.0f} "
            f"src={source.get('view', 0.0):.2f}/{source.get('batch_neighbor', 0.0):.2f}/"
            f"{source.get('memory_neighbor', 0.0):.2f}/{source.get('arf_planned', 0.0):.2f}/"
            f"{source.get('arf_missed_bonus', 0.0):.2f}/h{source.get('hard_negative_weight', 1.0):.2f})"
        )

    def format_epoch(self, epoch: int, stats: Dict[str, float]) -> str:
        return (
            f"agent_epoch epoch={epoch} time={stats.get('train_time_sec', 0.0):.1f}s "
            f"observe(valid={stats.get('agent_obs_valid', 0.0):.3f} "
            f"sem={stats.get('agent_obs_sem', 0.0):.4f} dyn={stats.get('agent_obs_dyn', 0.0):.4f} "
            f"hash_entropy={stats.get('metric_bit_entropy', 0.0):.3f} bit_use={stats.get('metric_bit_usage', 0.0):.3f}) "
            f"action(alpha={stats.get('agent_route_alpha', 0.0):.3f}+/-{stats.get('agent_route_alpha_std', 0.0):.3f} "
            f"H={stats.get('agent_route_entropy', 0.0):.3f} omega={stats.get('agent_sample_omega', 0.0):.3f} "
            f"budget={stats.get('agent_budget', 0.0):.0f} top_r={stats.get('metric_agentic_trace_top_r', stats.get('agent_top_r', 0.0)):.0f}) "
            f"feedback(planned={stats.get('metric_agentic_pos_arf', 0.0):.1f} "
            f"missed={stats.get('metric_arf_missed_ratio', 0.0):.3f} "
            f"false={stats.get('metric_arf_false_ratio', 0.0):.3f} "
            f"overlap={stats.get('metric_arf_actual_overlap', 0.0):.3f} "
            f"edge={stats.get('metric_agentic_edge_factor', 0.0):.3f}) "
            f"memory(edge_p={stats.get('memory_edge_posterior', 0.0):.3f} "
            f"edge_r={stats.get('memory_edge_reliability', 0.0):.3f} "
            f"stable={stats.get('memory_edge_stability', 0.0):.3f}) "
            f"adapt(loss={stats.get('loss', 0.0):.4f} aucl={stats.get('component_agentic_contrastive', stats.get('loss_semantic', 0.0)):.4f} "
            f"hash={stats.get('loss_hash', 0.0):.4f})"
        )

    def observe_eval(
        self,
        epoch: int,
        metrics: Dict[str, float],
        train_stats: Dict[str, float],
        planner_memory=None,
    ) -> Dict[str, float | bool | str]:
        current_map = float(metrics.get("mAP", 0.0))
        if self.last_map is None:
            gain = 0.0
            self.gain_ema = 0.0
        else:
            gain = current_map - self.last_map
            self.gain_ema = self.stop_gain_beta * self.gain_ema + (1.0 - self.stop_gain_beta) * gain
        self.last_map = current_map

        if planner_memory is not None and hasattr(planner_memory, "global_edge_stability"):
            memory_stability = float(planner_memory.global_edge_stability())
        else:
            memory_stability = float(train_stats.get("memory_edge_stability", 0.0))
        gate_entropy = float(train_stats.get("agent_route_entropy", 0.0))
        cost = float(train_stats.get("train_time_sec", 0.0))
        utility = self.gain_ema * self.stop_horizon_windows - self.stop_cost_weight * cost * self.stop_horizon_windows
        stop_ready = (
            self.stop_enabled
            and epoch >= self.stop_min_epoch
            and self.gain_ema < self.stop_min_gain
            and memory_stability >= self.stop_min_memory_stability
            and self.stop_entropy_low <= gate_entropy <= self.stop_entropy_high
            and utility < self.stop_min_utility
        )
        if stop_ready:
            self.stop_window_count += 1
        else:
            self.stop_window_count = 0
        should_stop = bool(self.stop_enabled and self.stop_window_count >= self.stop_patience_windows)
        reason = "continue"
        if should_stop:
            reason = "low_utility_stable_memory"
        elif stop_ready:
            reason = f"confirming_{self.stop_window_count}/{self.stop_patience_windows}"
        return {
            "agent_eval_gain": gain,
            "agent_eval_gain_ema": self.gain_ema,
            "agent_eval_utility": utility,
            "agent_eval_memory_stability": memory_stability,
            "agent_eval_gate_entropy": gate_entropy,
            "agent_eval_stop": should_stop,
            "agent_eval_reason": reason,
        }

    def format_eval(self, epoch: int, metrics: Dict[str, float], eval_state: Dict[str, float | bool | str], best_map: float) -> str:
        return (
            f"agent_eval epoch={epoch} mAP={metrics.get('mAP', 0.0):.4f} "
            f"mAP@100={metrics.get('mAP@100', metrics.get('mAP@All', 0.0)):.4f} "
            f"P@5={metrics.get('Precision@5', 0.0):.4f} R@100={metrics.get('Recall@100', 0.0):.4f} "
            f"best={best_map:.4f} "
            f"gain={float(eval_state.get('agent_eval_gain', 0.0)):.5f} "
            f"gain_ema={float(eval_state.get('agent_eval_gain_ema', 0.0)):.5f} "
            f"utility={float(eval_state.get('agent_eval_utility', 0.0)):.5f} "
            f"memory_stable={float(eval_state.get('agent_eval_memory_stability', 0.0)):.3f} "
            f"gate_H={float(eval_state.get('agent_eval_gate_entropy', 0.0)):.3f} "
            f"stop={bool(eval_state.get('agent_eval_stop', False))} "
            f"reason={eval_state.get('agent_eval_reason', 'continue')}"
        )
