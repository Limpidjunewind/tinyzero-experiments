# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO trainer v3-random: Outcome Reward (v2) + RANDOM (untrained) PRM.
Ablation control: uses the same PRM architecture but with random weights
to test whether PRM's contribution comes from learned signal or just
from the act of injecting any step-level reward.
"""

import os
import sys
from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math, multiply, countdown_v2
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


PRM_CHECKPOINT = os.environ.get(
    'PRM_CHECKPOINT',
    '/data/fangda/tinyzero/prm/checkpoints_v1_improved/best_model.pt'
)
PRM_WEIGHT = float(os.environ.get('PRM_WEIGHT', '0.1'))


def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score
    elif "countdown" in data_source:
        return countdown_v2.compute_score
    else:
        raise NotImplementedError


class RewardManagerWithPRM():
    """Reward manager combining outcome reward (v2) + process reward (PRM)."""

    def __init__(self, tokenizer, num_examine, prm_scorer=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.prm_scorer = prm_scorer

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        prm_rewards_sum = 0.0
        prm_steps_total = 0
        prm_count = 0

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # 1. Outcome reward (v2 multi-granularity)
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            outcome_score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)

            # 2. Process reward (PRM)
            process_reward = 0.0
            if self.prm_scorer is not None:
                process_reward, num_steps, avg_score = self.prm_scorer.compute_process_reward(sequences_str)
                prm_rewards_sum += process_reward
                prm_steps_total += num_steps
                prm_count += 1

            # Combine: outcome + process
            total_reward = outcome_score + process_reward
            reward_tensor[i, valid_response_length - 1] = total_reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        # Log PRM stats periodically
        if prm_count > 0:
            import random
            if random.randint(1, 10) == 1:
                avg_prm = prm_rewards_sum / prm_count
                avg_steps = prm_steps_total / prm_count
                print(f"[PRM] avg_reward={avg_prm:.4f}, avg_steps={avg_steps:.1f}, samples={prm_count}")

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # Initialize RANDOM PRM scorer (runs on CPU, no checkpoint loaded)
    sys.path.insert(0, '/data/fangda/tinyzero/prm')
    from integrate_random import PRMScorer

    prm_scorer = PRMScorer(
        checkpoint_path=None,
        device='cpu',
        prm_weight=PRM_WEIGHT
    )
    print(f"[PRM-RANDOM] Using random weights, weight={PRM_WEIGHT}")

    reward_fn = RewardManagerWithPRM(tokenizer=tokenizer, num_examine=0, prm_scorer=prm_scorer)
    val_reward_fn = RewardManagerWithPRM(tokenizer=tokenizer, num_examine=1, prm_scorer=prm_scorer)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
