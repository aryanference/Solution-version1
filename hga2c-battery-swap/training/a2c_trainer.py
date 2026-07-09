"""A2C + GAE training loop for HGA²C (§6.1)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter

from env.battery_swap_env import BatterySwapEnv, make_env
from env.simulator import Plan, VehicleRoute, simulate_plan
from models.hga2c_policy import HGA2CPolicy, build_policy_from_config

logger = logging.getLogger(__name__)


class A2CTrainer:
    def __init__(
        self,
        policy: HGA2CPolicy,
        hyperparams: dict[str, Any],
        economics: dict[str, Any],
        device: str = "cpu",
        log_dir: str = "runs/hga2c",
    ) -> None:
        self.policy = policy.to(device)
        self.device = device
        self.hp = hyperparams
        self.economics = economics

        # A2C hyperparameters
        self.gamma = hyperparams.get("gamma", 0.99)
        self.lambda_gae = hyperparams.get("lambda_gae", 0.95)
        self.entropy_coeff = hyperparams.get("entropy_coeff", 0.01)
        self.value_loss_coeff = hyperparams.get("value_loss_coeff", 0.5)
        self.grad_clip = hyperparams.get("grad_clip", 1.0)
        self.grad_accum_steps = hyperparams.get("gradient_accumulation_steps", 1)
        self.penalty_schedule_end = hyperparams.get("penalty_schedule_end_episode", 0)

        # Base penalty for curriculum
        self.base_lambda_unmet = economics.get("lambda_unmet", 100.0)
        self.max_lambda_unmet = 1000.0

        # Optimizer with separate LR groups
        lr_encoder = hyperparams.get("lr_encoder", 1e-4)
        lr_critic = hyperparams.get("lr_critic", 3e-4)

        encoder_params = list(policy.encoder.parameters()) + \
                         list(policy.allocation_actor.parameters()) + \
                         list(policy.routing_actor.parameters())
        critic_params = list(policy.critic.parameters())

        self.optimizer = torch.optim.Adam([
            {"params": encoder_params, "lr": lr_encoder},
            {"params": critic_params, "lr": lr_critic},
        ])

        # TensorBoard
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

    def train_episode(
        self,
        env: BatterySwapEnv,
        instance: dict[str, Any],
        x_labels: list[int] | None = None,
        p_labels: list[list[int]] | None = None,
        accumulate: bool = False,
    ) -> dict[str, float]:
        self.policy.train()

        obs, info = env.reset(options={"instance": instance})

        # Forward pass: get full allocation + routing plan
        result = self.policy.forward(
            obs, instance, self.economics, greedy=False,
            x_labels=x_labels, p_labels=p_labels
        )

        # Build plan for simulation
        n_regions = instance["region_count"]
        vehicle_routes_obj = []
        for v_idx, route in enumerate(result["vehicle_routes"]):
            vehicle_routes_obj.append(VehicleRoute(vehicle_id=v_idx, route=route))

        plan = Plan(
            x=result["x"],
            p=result["p"],
            vehicle_routes=vehicle_routes_obj,
            vehicle_assignments={
                r: v for v, route in enumerate(result["vehicle_routes"]) for r in route
            },
        )

        sim_result = simulate_plan(plan, instance, self.economics)
        reward = -sim_result.objective_z

        # Compute loss
        log_probs = result["log_probs"]
        entropies = result["entropies"]
        value = result["value"].squeeze()

        if log_probs:
            total_log_prob = torch.stack(log_probs).sum()
            total_entropy = torch.stack(entropies).sum()
        else:
            total_log_prob = torch.tensor(0.0, device=self.device)
            total_entropy = torch.tensor(0.0, device=self.device)

        # Advantage = reward - value (single-step episode)
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
        advantage = (reward_tensor - value).detach()

        # A2C loss components
        policy_loss = -(total_log_prob * advantage)
        value_loss = (value - reward_tensor) ** 2
        entropy_bonus = total_entropy

        loss = (
            policy_loss
            + self.value_loss_coeff * value_loss
            - self.entropy_coeff * entropy_bonus
        ) / self.grad_accum_steps

        # Backward pass
        if not accumulate and self.grad_accum_steps == 1:
            self.optimizer.zero_grad()
            
        loss.backward()

        # Gradient clipping and stepping
        if not accumulate:
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.grad_clip
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            # We can't accurately get grad_norm here without stepping, so return 0
            grad_norm = 0.0

        # Metrics
        metrics = {
            "reward": reward,
            "objective_z": sim_result.objective_z,
            "travel_cost": sim_result.travel_cost,
            "unmet_demand": sim_result.total_unmet_demand,
            "delay_penalty": sim_result.delay_penalty,
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_bonus.item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "violations": len(sim_result.feasibility_violations),
        }

        return metrics

    def train(
        self,
        instances: Any,  # list[dict] or IterableDataset
        n_episodes: int,
        eval_interval: int = 100,
        eval_instance: dict[str, Any] | None = None,
        checkpoint_dir: str = "checkpoints",
        stage_name: str = "stage",
        start_episode: int = 0,
        early_stopping_patience: int = 5,
        early_stopping_threshold: float = 0.5,
    ) -> list[dict[str, float]]:
        rng = np.random.RandomState(self.hp.get("seed", 42))
        all_metrics: list[dict[str, float]] = []

        env = make_env(economics=self.economics)

        is_list = isinstance(instances, list)
        if not is_list:
            instance_iter = iter(instances)

        logger.info("Starting training: %d episodes (resuming from %d)", n_episodes, start_episode)
        
        self.optimizer.zero_grad()
        self.global_step = start_episode
        
        best_eval_z = float("inf")
        patience_counter = 0

        for episode in range(start_episode, n_episodes):
            # Penalty Curriculum
            if self.penalty_schedule_end > 0:
                progress = min(1.0, episode / self.penalty_schedule_end)
                curr_lambda = self.base_lambda_unmet + progress * (self.max_lambda_unmet - self.base_lambda_unmet)
                env.economics["lambda_unmet"] = curr_lambda

            # Sample random instance
            if is_list:
                idx = rng.randint(0, len(instances))
                sample = instances[idx]
            else:
                sample = next(instance_iter)
            
            # Check if this is a labeled instance (Stage 1)
            if "labels" in sample:
                instance = sample["instance"]
                x_labels = sample["labels"]["x"]
                p_labels = sample["labels"]["p"]
            else:
                instance = sample
                x_labels, p_labels = None, None

            accumulate = (episode + 1) % self.grad_accum_steps != 0 and (episode + 1) != n_episodes
            metrics = self.train_episode(
                env, instance, x_labels=x_labels, p_labels=p_labels, accumulate=accumulate
            )
            all_metrics.append(metrics)

            # Log to TensorBoard
            self.global_step += 1
            for key, val in metrics.items():
                self.writer.add_scalar(f"train/{key}", val, self.global_step)

            # Progress logging
            if (episode + 1) % 10 == 0:
                recent = all_metrics[-10:]
                avg_reward = np.mean([m["reward"] for m in recent])
                avg_unmet = np.mean([m["unmet_demand"] for m in recent])
                avg_violations = np.mean([m["violations"] for m in recent])
                logger.info(
                    "Episode %d/%d: avg_reward=%.2f, avg_unmet=%.1f, violations=%.0f",
                    episode + 1, n_episodes, avg_reward, avg_unmet, avg_violations,
                )

            # Evaluation
            if eval_instance and (episode + 1) % eval_interval == 0:
                eval_metrics = self.evaluate(env, eval_instance)
                for key, val in eval_metrics.items():
                    self.writer.add_scalar(f"eval/{key}", val, self.global_step)
                logger.info(
                    "EVAL episode %d: Z=%.2f, unmet=%.1f",
                    episode + 1, eval_metrics["objective_z"],
                    eval_metrics["unmet_demand"],
                )
                
                # Early Stopping Logic
                if eval_metrics["unmet_demand"] <= early_stopping_threshold:
                    patience_counter += 1
                    logger.info("Early stopping condition met (unmet <= %.1f). Patience: %d/%d", 
                                early_stopping_threshold, patience_counter, early_stopping_patience)
                    
                    if patience_counter >= early_stopping_patience:
                        logger.info("Early stopping triggered! Model has converged.")
                        break
                else:
                    patience_counter = 0

            # Checkpoint
            if (episode + 1) % (eval_interval * 5) == 0:
                ckpt_path = Path(checkpoint_dir) / f"{stage_name}_ep{episode+1}.pt"
                self.policy.save_checkpoint(ckpt_path)
                logger.info("Saved checkpoint: %s", ckpt_path)

        # Final checkpoint
        final_path = Path(checkpoint_dir) / f"{stage_name}_final.pt"
        self.policy.save_checkpoint(final_path)
        logger.info("Training complete. Final checkpoint: %s", final_path)

        self.writer.close()
        return all_metrics

    def evaluate(
        self,
        env: BatterySwapEnv,
        instance: dict[str, Any],
        n_rollouts: int = 1,
    ) -> dict[str, float]:
        self.policy.eval()
        metrics_list: list[dict[str, float]] = []

        with torch.no_grad():
            for _ in range(n_rollouts):
                obs, _ = env.reset(options={"instance": instance})
                result = self.policy.forward(obs, instance, self.economics, greedy=True)

                vehicle_routes_obj = []
                for v_idx, route in enumerate(result["vehicle_routes"]):
                    vehicle_routes_obj.append(VehicleRoute(vehicle_id=v_idx, route=route))

                plan = Plan(
                    x=result["x"],
                    p=result["p"],
                    vehicle_routes=vehicle_routes_obj,
                    vehicle_assignments={
                        r: v for v, route in enumerate(result["vehicle_routes"]) for r in route
                    },
                )

                sim_result = simulate_plan(plan, instance, self.economics)
                metrics_list.append({
                    "reward": -sim_result.objective_z,
                    "objective_z": sim_result.objective_z,
                    "unmet_demand": sim_result.total_unmet_demand,
                    "violations": len(sim_result.feasibility_violations),
                })

        # Average
        avg: dict[str, float] = {}
        for key in metrics_list[0]:
            avg[key] = np.mean([m[key] for m in metrics_list])
        return avg
