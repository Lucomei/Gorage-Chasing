#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。
"""

import numpy as np

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max distance bucket / 距离桶最大值
MAX_DIST_BUCKET = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5
        self.last_total_score = 0.0
        self.last_treasure_count = 0.0
        self.last_buff_remain_norm = 0.0
        self.last_min_treasure_dist_norm = 1.0

    def _to_float(self, value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _extract_env_metrics(self, observation):
        """Best-effort extraction of score/treasure metrics from observation.

        尽可能从观测中提取分数/宝箱指标（兼容不同字段命名）。
        """
        env_info = observation.get("env_info", {}) or {}
        frame_state = observation.get("frame_state", {}) or {}

        # Score candidates / 分数字段候选
        total_score = self._to_float(
            env_info.get("total_score", env_info.get("score", frame_state.get("total_score", 0.0))),
            0.0,
        )

        # Treasure candidates / 宝箱字段候选
        treasure_count = 0.0
        treasure_candidates = [
            env_info.get("treasure_count"),
            env_info.get("collected_treasure"),
            env_info.get("treasure_collected"),
            frame_state.get("treasure_count"),
            frame_state.get("collected_treasure"),
            frame_state.get("treasure_collected"),
        ]

        for item in treasure_candidates:
            if item is not None:
                treasure_count = self._to_float(item, 0.0)
                break

        # If treasure list is provided, use its length as fallback / 若给出宝箱列表，退化用长度
        if treasure_count <= 0.0:
            for key in ("treasures", "treasure_list", "chests", "chest_list"):
                value = frame_state.get(key)
                if isinstance(value, list):
                    treasure_count = float(len(value))
                    break

        return total_score, treasure_count

    def _extract_min_treasure_dist_norm(self, frame_state, hero_pos):
        """Estimate normalized min distance to visible treasures/chests.

        估计角色到可见宝箱的最小归一化距离。
        """
        min_dist_norm = 1.0
        for key in ("treasures", "treasure_list", "chests", "chest_list"):
            items = frame_state.get(key)
            if not isinstance(items, list):
                continue

            for item in items:
                if not isinstance(item, dict):
                    continue
                pos = item.get("pos")
                if not isinstance(pos, dict):
                    continue
                tx = pos.get("x")
                tz = pos.get("z")
                if tx is None or tz is None:
                    continue
                raw_dist = np.sqrt((hero_pos["x"] - tx) ** 2 + (hero_pos["z"] - tz) ** 2)
                min_dist_norm = min(min_dist_norm, _norm(raw_dist, MAP_SIZE * 1.41))
        return min_dist_norm

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, and reward.

        将 env_obs 转换为特征向量、合法动作掩码和即时奖励。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)

        # Hero self features (4D) / 英雄自身特征
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        flash_cd_norm = _norm(hero["flash_cooldown"], MAX_FLASH_CD)
        buff_remain_norm = _norm(hero["buff_remaining_time"], MAX_BUFF_DURATION)

        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm], dtype=np.float32)

        # Monster features (5D x 2) / 怪物特征
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                m_pos = m["pos"]
                if is_in_view:
                    m_x_norm = _norm(m_pos["x"], MAP_SIZE)
                    m_z_norm = _norm(m_pos["z"], MAP_SIZE)
                    m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)

                    # Euclidean distance / 欧式距离
                    raw_dist = np.sqrt((hero_pos["x"] - m_pos["x"]) ** 2 + (hero_pos["z"] - m_pos["z"]) ** 2)
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                else:
                    m_x_norm = 0.0
                    m_z_norm = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0
                monster_feats.append(
                    np.array([is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm], dtype=np.float32)
                )
            else:
                monster_feats.append(np.zeros(5, dtype=np.float32))

        # Local map features (16D) / 局部地图特征
        map_feat = np.zeros(16, dtype=np.float32)
        if map_info is not None and len(map_info) >= 13:
            center = len(map_info) // 2
            flat_idx = 0
            for row in range(center - 2, center + 2):
                for col in range(center - 2, center + 2):
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        map_feat[flat_idx] = float(map_info[row][col] != 0)
                    flat_idx += 1

        # Legal action mask (8D) / 合法动作掩码
        legal_action = [1] * 8
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(8, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < 8}
                legal_action = [1 if j in valid_set else 0 for j in range(8)]

        if sum(legal_action) == 0:
            legal_action = [1] * 8

        # Progress features (2D) / 进度特征
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        # Concatenate features / 拼接特征
        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                map_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
            ]
        )

        # Step reward / 即时奖励
        cur_min_dist_norm = 1.0
        for m_feat in monster_feats:
            if m_feat[0] > 0:
                cur_min_dist_norm = min(cur_min_dist_norm, m_feat[4])

        survive_reward = 0.01
        dist_shaping = 0.08 * (cur_min_dist_norm - self.last_min_monster_dist_norm)

        self.last_min_monster_dist_norm = cur_min_dist_norm

        # Treasure/score shaping / 宝箱与得分塑形
        total_score, treasure_count = self._extract_env_metrics(observation)
        score_delta = max(0.0, total_score - self.last_total_score)
        treasure_delta = max(0.0, treasure_count - self.last_treasure_count)

        # Treasure proximity shaping / 接近宝箱塑形（距离变小给正奖励）
        cur_min_treasure_dist_norm = self._extract_min_treasure_dist_norm(frame_state, hero_pos)
        treasure_approach_reward = 0.08 * (self.last_min_treasure_dist_norm - cur_min_treasure_dist_norm)

        # Buff pickup reward / 拾取加速增益奖励（buff剩余时间从0上升）
        buff_pick_reward = 0.0
        if self.last_buff_remain_norm <= 1e-6 and buff_remain_norm > 1e-6:
            buff_pick_reward = 0.35

        # Encourage task objective stronger than before.
        # 提高任务导向强度：宝箱与得分增量权重上调。
        score_reward = 0.03 * score_delta
        treasure_reward = 0.9 * treasure_delta

        self.last_total_score = total_score
        self.last_treasure_count = treasure_count
        self.last_buff_remain_norm = buff_remain_norm
        self.last_min_treasure_dist_norm = cur_min_treasure_dist_norm

        reward = [
            survive_reward
            + dist_shaping
            + score_reward
            + treasure_reward
            + treasure_approach_reward
            + buff_pick_reward
        ]

        return feature, legal_action, reward
