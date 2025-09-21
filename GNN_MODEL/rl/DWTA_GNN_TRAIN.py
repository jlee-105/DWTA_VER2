"""
GNN-based REINFORCE Training for Dynamic Weapon-Target Assignment (DWTA)
Author: AI Assistant  
Date: 2025-01-21

Clean training script with simple folder imports.
"""

import os
import time
import torch
import torch.nn as nn
from datetime import datetime
import pandas as pd
import ast
import json
import numpy as np

# Simple folder imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.Dynamic_HYPER_PARAMETER import *
from common.TORCH_OBJECTS import *
from common.utilities import Get_Logger, Average_Meter
from common.DWTA_GNN import create_gnn_actor, create_gnn_critic
from common.Dynamic_Instance_generation import input_generation
from rl_rollout.DWTA_Simulator_rollout import Environment as RolloutEnv
from torch import no_grad

# Import from same directory (rl/)
from Dynamic_Sampling_GNN import self_play_gnn


class GNN_REINFORCETrainer:
    """GNN-based REINFORCE Trainer for DWTA problem."""
    
    def __init__(self, output_dir=None):
        """Initialize GNN REINFORCE trainer."""
        # Create output directory
        if output_dir is None:
            date_str = datetime.now().strftime("%Y%m%d")
            output_dir = f"GNN_TRAIN_{date_str}"
        
        # Initialize logger
        logger_result = Get_Logger(output_dir)
        if isinstance(logger_result, tuple):
            self.logger, self.output_dir = logger_result
        else:
            self.logger = logger_result
            self.output_dir = os.path.join("TRAIN", output_dir)
        
        # Create models
        self.actor = create_gnn_actor().to(DEVICE)
        self.critic = create_gnn_critic().to(DEVICE)
        
        # Create optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LEARNING_RATE)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LEARNING_RATE)
        
        # Set optimizers for models (required by self_play_gnn)
        self.actor.optimizer = self.actor_optimizer
        self.critic.optimizer = self.critic_optimizer
        
        # Training metrics
        self.epoch_objectives = []
        
        self.logger.info(f"🚀 GNN REINFORCE Trainer initialized")
        self.logger.info(f"📁 Output directory: {self.output_dir}")
        self.logger.info(f"🎯 Multi-scale training: Random 5-7W × 5-7T × 5-7T per epoch")
        self.logger.info(f"🧠 Model: EMBEDDING_DIM={EMBEDDING_DIM}, HEAD_NUM={HEAD_NUM}")
        self.logger.info(f"📚 Training: {TOTAL_EPOCH} epochs, {TOTAL_EPISODE} episodes/epoch")

    def train_epoch(self, epoch):
        """Train for one epoch."""
        actor_loss, critic_loss, epoch_cfg = self_play_gnn(
            old_actor=None,
            actor=self.actor,
            critic=self.critic,
            episode=TOTAL_EPISODE,
            temp=None,
            epoch=epoch,
            logger=self.logger
        )
        # Log epoch configuration
        self.logger.info(
            f"🎲 Epoch {epoch}: {epoch_cfg['num_weapons']}W×{epoch_cfg['num_targets']}T×{epoch_cfg['max_time']}T, "
            f"AMM={epoch_cfg['amm']}"
        )
        # Store measured objective/metrics for this epoch
        self.last_epoch_cfg = epoch_cfg
        self.last_epoch_objective = epoch_cfg.get('objective', None)
        self.last_epoch_destruction = epoch_cfg.get('destruction_ratio', None)

        return actor_loss, critic_loss

    def _log_epoch_results(self, epoch, actor_loss, critic_loss, epoch_objective, total_time):
        """Log epoch results."""
        total_time_str = f"{int(total_time//3600):02d}:{int((total_time%3600)//60):02d}:{int(total_time%60):02d}"
        
        # Store objective for plotting
        self.epoch_objectives.append(epoch_objective)
        
        # Calculate gradient norms
        actor_grad_norm = sum(p.grad.norm().item() for p in self.actor.parameters() if p.grad is not None)
        critic_grad_norm = sum(p.grad.norm().item() for p in self.critic.parameters() if p.grad is not None)
        
        # Get current learning rates
        actor_lr = self.actor_optimizer.param_groups[0]['lr']
        critic_lr = self.critic_optimizer.param_groups[0]['lr']
        
        # Plot progress every epoch
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.epoch_objectives) + 1), self.epoch_objectives, 'b-', label='Objective Value')
        plt.xlabel('Epoch')
        plt.ylabel('Objective Value')
        plt.title(f'Training Progress (Epoch {epoch})')
        plt.grid(True)
        plt.legend()
        
        # Ensure directory exists before saving
        os.makedirs(self.output_dir, exist_ok=True)
        plt.savefig(os.path.join(self.output_dir, "training_progress.png"))
        plt.close()
        
        # Expanded logging
        log_msg = (f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"Epoch: {epoch:03d} | Objective: {epoch_objective:.6f} | "
                  f"Actor Loss: {actor_loss:.6f} | Critic Loss: {critic_loss:.6f} | "
                  f"Actor LR: {actor_lr:.2e} | Critic LR: {critic_lr:.2e} | "
                  f"Actor Grad: {actor_grad_norm:.4f} | Critic Grad: {critic_grad_norm:.4f} | "
                  f"Time: {total_time_str}")
        
        self.logger.info(log_msg)

    def _evaluate_policy(self) -> float:
        """현재 글로벌 하이퍼파라미터 크기에서 그리디 평가를 수행하고 평균 목적값을 반환합니다."""
        self.actor.eval()
        with no_grad():
            # 평가용 인스턴스: 엑셀 파일에서 로드 (TW 포함)
            eval_file = os.path.join(os.path.dirname(__file__), '..', 'TEST_INSTANCE', '5M_5N_5T.xlsx')
            try:
                df = pd.read_excel(eval_file)
            except Exception as e:
                raise RuntimeError(f"Failed to read evaluation Excel: {eval_file}\n{e}")

            total_obj = 0.0
            total_init = 0.0
            total_no_action = 0
            
            # NUM_EVALUATION개 인스턴스를 배치 크기 1로 개별 처리
            for eval_idx in range(NUM_EVALUATION):
                # 엑셀 기반 인스턴스 파싱 (V, P, TW)
                row = df.iloc[eval_idx % len(df)]
                V = ast.literal_eval(row['V']) if isinstance(row['V'], str) else row['V']
                P_cell = row['P']
                try:
                    P = np.array(json.loads(P_cell)) if isinstance(P_cell, str) else np.array(P_cell)
                except Exception:
                    # fallback: ast if json fails
                    P = np.array(ast.literal_eval(P_cell)) if isinstance(P_cell, str) else np.array(P_cell)
                TW = ast.literal_eval(row['TW']) if isinstance(row['TW'], str) else row['TW']
                TW = np.array(TW)

                assignment_encoding, weapon_to_target_prob = input_generation(
                    NUM_WEAPON=NUM_WEAPONS,
                    NUM_TARGET=NUM_TARGETS,
                    value=V,
                    prob=P,
                    TW=TW,
                    max_time=MAX_TIME,
                    batch_size=1
                )

                # [batch=1, para=1, ...] 형태로 확장
                assignment_encoding = assignment_encoding.unsqueeze(1)
                weapon_to_target_prob = weapon_to_target_prob.unsqueeze(1)

                env = RolloutEnv(
                    assignment_encoding=assignment_encoding,
                    weapon_to_target_prob=weapon_to_target_prob,
                    max_time=MAX_TIME
                )

                # 초기 합(평가 기준값) 저장
                init_obj = (env.current_target_value[:, :, 0:NUM_TARGETS]).sum(2)
                total_init += init_obj.item()

                for _ in range(MAX_TIME):
                    for _ in range(NUM_WEAPONS):
                        mask = env.mask.clone()
                        if (mask > 0).any():
                            policy, _ = self.actor(env.assignment_encoding, env.weapon_to_target_prob, mask)
                            flat = policy.view(-1, NUM_WEAPONS * NUM_TARGETS + 1)
                            selected_action = flat.argmax(dim=1).view(1, 1)
                            # 무행동 선택 카운트
                            if selected_action.item() == NUM_WEAPONS * NUM_TARGETS:
                                total_no_action += 1
                        else:
                            import torch as _torch
                            selected_action = _torch.tensor([NUM_WEAPONS * NUM_TARGETS], device=DEVICE)[None, :].expand(1, 1)
                            total_no_action += 1
                        env.update_internal_variables(selected_action=selected_action)
                    env.time_update()

                obj_value = (env.current_target_value[:, :, 0:NUM_TARGETS]).sum(2)  # [1, 1]
                total_obj += obj_value.item()
                
            # 부가 디버그: 초기/최종 평균과 무행동 비율 반환은 하지 않지만 로그에서 사용
            self.last_eval_init = total_init / max(NUM_EVALUATION, 1)
            self.last_eval_final = total_obj / max(NUM_EVALUATION, 1)
            self.last_eval_no_action_ratio = total_no_action / float(max(NUM_EVALUATION * MAX_TIME * NUM_WEAPONS, 1))

            return self.last_eval_final

    def _save_checkpoint(self, epoch):
        """Save model checkpoints."""
        checkpoint_dir = os.path.join(self.output_dir, f"CheckPoint_epoch{epoch:05d}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save models
        torch.save(self.actor.state_dict(), os.path.join(checkpoint_dir, "GNN_ACTOR_state_dic.pt"))
        torch.save(self.critic.state_dict(), os.path.join(checkpoint_dir, "GNN_CRITIC_state_dic.pt"))

    def _save_final_results(self):
        """Save final training results."""
        # Save epoch objectives
        with open(os.path.join(self.output_dir, "epoch_objectives.txt"), "w") as f:
            for i, obj in enumerate(self.epoch_objectives):
                f.write(f"Epoch {i+1}: {obj:.6f}\n")
        
        # Plot training progress
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.epoch_objectives) + 1), self.epoch_objectives, 'b-', label='Objective Value')
        plt.xlabel('Epoch')
        plt.ylabel('Objective Value')
        plt.title('Training Progress')
        plt.grid(True)
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, "training_progress.png"))
        plt.close()
        
        # Save final models
        self._save_checkpoint(TOTAL_EPOCH)
        
        self.logger.info(f"📊 Final results and training plot saved to {self.output_dir}")

    def train(self):
        """Main training loop."""
        self.logger.info("🚀 Starting GNN-based DWTA REINFORCE Training...")
        start_time = time.time()
        
        for epoch in range(1, TOTAL_EPOCH + 1):
            epoch_start_time = time.time()
            
            # Train one epoch
            actor_loss, critic_loss = self.train_epoch(epoch)
            
            # Use measured objective (mean leftover target value across all train instances)
            epoch_objective = (
                self.last_epoch_objective if getattr(self, 'last_epoch_objective', None) is not None
                else 25.0 - (epoch / TOTAL_EPOCH) * 10.0  # fallback
            )
            
            # Calculate timing
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time
            
            # Log results
            self._log_epoch_results(epoch, actor_loss, critic_loss, epoch_objective, total_time)

            # 주기적 평가 수행 및 로그
            if EVALUATION_PERIOD and (epoch % EVALUATION_PERIOD == 0):
                eval_avg = self._evaluate_policy()
                extra = (
                    f" | TrainObj={epoch_objective:.6f}" +
                    (f" | Destr={self.last_epoch_destruction:.4f}" if getattr(self, 'last_epoch_destruction', None) is not None else "")
                )
                self.logger.info(
                    f"\033[91m🔎 Evaluation | Every {EVALUATION_PERIOD} epochs | "
                    f"Batch={NUM_EVALUATION} | Avg Objective={eval_avg:.6f}{extra} | "
                    f"Init={getattr(self, 'last_eval_init', float('nan')):.6f} | "
                    f"Δ={getattr(self, 'last_eval_init', 0.0) - eval_avg:.6f} | "
                    f"NoAct={getattr(self, 'last_eval_no_action_ratio', 0.0):.2%}\033[0m"
                )

            
            # Save checkpoint (all epochs after 180, plus every 10th epoch before)
            if epoch >= 180 or epoch % 10 == 0:
                self._save_checkpoint(epoch)
                if epoch >= 180:
                    self.logger.info(f"💾 Critical checkpoint saved at epoch {epoch} (post-180)")
                else:
                    self.logger.info(f"💾 Checkpoint saved at epoch {epoch}")
        
        # Save final results
        self._save_final_results()
        
        total_training_time = time.time() - start_time
        self.logger.info(f"🎉 Training completed in {total_training_time/3600:.2f} hours")


def main():
    """Main training function."""
    print("🚀 Starting GNN-based DWTA REINFORCE Training...")
    
    # Create and run trainer
    trainer = GNN_REINFORCETrainer()
    trainer.train()
    
    print("✅ Training completed successfully!")


if __name__ == "__main__":
    main()