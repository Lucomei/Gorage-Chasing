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
from agent_ppo.conf.conf import Config

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

# Reward coefficients (easy to tune) / 奖励系数（便于微调）
SURVIVE_REWARD = 0.015
MONSTER_DIST_COEF = 0.12  # 提高：让怪物威胁更明显，惩罚向怪物靠近的行为
SCORE_DELTA_COEF = 0.05
TREASURE_PICK_COEF = 2.8
TREASURE_APPROACH_COEF = 0.45
BUFF_PICK_REWARD = 0.95
FLASH_EFFECTIVE_REWARD = 0.35  # 鼓励更积极地闪现拉开位移
WALL_COLLISION_PENALTY_MOVE = -0.25  # 再次重罚：蹭墙成本极大，强迫 agent 只有离开墙体才能获得正收益
WALL_COLLISION_PENALTY_FLASH = -0.40  # 严禁对着墙闪现
EXPLORE_ANCHOR_INTERVAL = 10  # 稍微拉长窗口，给予加速跑的空间
EXPLORE_MOVE_AWAY_COEF = 0.08  # 大幅增加移动奖励：位移本身就是奖励，只要不撞墙，跑得越快奖励越高
EXPLORE_TARGET_VISIBLE_SCALE = 0.05  # 降低：即使没看到宝箱，也要为了“位移”而疯狂跑动
EXPLORE_STAY_DIST_THRESHOLD = 5.0  # 激进阈值：10步内位移如果不远，立刻判定为打转
EXPLORE_WINDOW_STAY_PENALTY = -0.15  # 极重发呆惩罚：原地打转 10 步的惩罚相当于死一次的几分之一收益

# Dynamic Risk Management / 动态风险控制
SAFE_DIST_THRESHOLD = 0.12  # 极限距离限制：只在最后一刻才考虑逃避，平时全身心投入奔跑
DANGER_ZONE_PENALTY_SCALE = 3.0  # 进阶危险评估：进入危险区后的负收益呈指数级跳跃
EXPLORE_DESIRE_BASE = 1.0

# Opportunity-cost penalty (gated) / 机会成本惩罚（条件触发）
NO_TREASURE_STEP_THRESHOLD = 100  # 缩短焦虑周期
NO_TREASURE_PENALTY = -0.06
NO_TREASURE_STAGE2_STEP = 200  # 既然能活 500 步，200 步没成绩就开始严厉扣分
NO_TREASURE_STAGE2_PENALTY = -0.15
SAFE_MONSTER_DIST_NORM = 0.30
NO_TREASURE_PENALTY_START_PROGRESS = 0.15  # 让惩罚更早生效

# Annealing schedule: early risky -> later stable
# 退火调度：前期偏冒险，后期偏稳定
ANNEAL_START_PROGRESS = 0.10
ANNEAL_END_PROGRESS = 0.80
EARLY_TREASURE_PICK_SCALE = 2.0
EARLY_TREASURE_APPROACH_SCALE = 1.9
EARLY_BUFF_PICK_SCALE = 1.6
EARLY_MONSTER_DIST_SCALE = 0.3
EARLY_NO_TREASURE_PENALTY_SCALE = 1.8


def _phase_mix(step_norm):
    """Return exploration intensity in [0, 1].

    1 表示前期探索强，0 表示后期稳定。
    """
    if step_norm <= ANNEAL_START_PROGRESS:
        return 1.0
    if step_norm >= ANNEAL_END_PROGRESS:
        return 0.0
    span = ANNEAL_END_PROGRESS - ANNEAL_START_PROGRESS
    return 1.0 - (step_norm - ANNEAL_START_PROGRESS) / span


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
        self.last_min_monster_dist_norm = 1.0  # 修改：初始距离设为最大，避免第1步误判距离缩短
        self.last_total_score = 0.0
        self.last_treasure_collected = 0.0
        self.last_treasure_remaining = None
        self.last_buff_remain_norm = 0.0
        self.last_min_treasure_dist_norm = 1.0
        self.no_treasure_steps = 0
        self.total_exploration_dist = 0.0  # 全局地图探索累计（用于衡量疲劳/渴望）
        self.last_hero_pos_raw = None
        self.last_hero_pos = None
        self.last_action = 0  # 增加：显式重置
        self.explore_anchor_pos = None
        self.explore_window_steps = 0
        self.explore_last_anchor_dist = 0.0
        self.explore_window_max_dist = 0.0

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

        # Collected treasure candidates / 已收集宝箱字段候选
        treasure_collected = 0.0
        collected_candidates = [
            env_info.get("collected_treasure"),
            env_info.get("treasure_collected"),
            frame_state.get("collected_treasure"),
            frame_state.get("treasure_collected"),
        ]
        for item in collected_candidates:
            if item is not None:
                treasure_collected = self._to_float(item, 0.0)
                break

        # Remaining treasure candidates / 剩余宝箱字段候选
        treasure_remaining = None
        remaining_candidates = [
            env_info.get("remain_treasure"),
            env_info.get("remaining_treasure"),
            env_info.get("left_treasure"),
            env_info.get("treasure_count"),
            frame_state.get("remain_treasure"),
            frame_state.get("remaining_treasure"),
            frame_state.get("left_treasure"),
            frame_state.get("treasure_count"),
        ]
        for item in remaining_candidates:
            if item is not None:
                treasure_remaining = self._to_float(item, 0.0)
                break

        # If treasure list is provided, use its length as remaining fallback.
        # 若给出宝箱列表，退化为“剩余宝箱数”。
        if treasure_remaining is None:
            for key in ("treasures", "treasure_list", "chests", "chest_list"):
                value = frame_state.get(key)
                if isinstance(value, list):
                    treasure_remaining = float(len(value))
                    break

        return total_score, treasure_collected, treasure_remaining

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

    def _extract_target_feature(self, frame_state, hero_pos, keys, max_count):
        """Build target feature: [has_visible, count_norm, dx, dz, dist].

        目标特征: [是否可见, 可见数量归一化, 最近目标相对x, 相对z, 最近距离归一化]。
        """
        visible_count = 0
        min_dist = 1e9
        best_dx = 0.0
        best_dz = 0.0

        for key in keys:
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

                is_in_view = float(item.get("is_in_view", 1))
                if is_in_view <= 0:
                    continue

                visible_count += 1
                dx = (tx - hero_pos["x"]) / MAP_SIZE
                dz = (tz - hero_pos["z"]) / MAP_SIZE
                raw_dist = np.sqrt((hero_pos["x"] - tx) ** 2 + (hero_pos["z"] - tz) ** 2)
                if raw_dist < min_dist:
                    min_dist = raw_dist
                    best_dx = float(np.clip(dx, -1.0, 1.0))
                    best_dz = float(np.clip(dz, -1.0, 1.0))

        has_visible = 1.0 if visible_count > 0 else 0.0
        count_norm = _norm(visible_count, max_count)
        dist_norm = _norm(min_dist, MAP_SIZE * 1.41) if visible_count > 0 else 1.0

        return np.array([has_visible, count_norm, best_dx, best_dz, dist_norm], dtype=np.float32)

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, and reward.

        将 env_obs 转换为特征向量、合法动作掩码和即时奖励。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation.get("legal_act", observation.get("legal_action", []))

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

        # Target features (10D): nearest treasure + nearest buff
        # 目标特征（10维）：最近宝箱 + 最近buff
        treasure_target_feat = self._extract_target_feature(
            frame_state,
            hero_pos,
            keys=("treasures", "treasure_list", "chests", "chest_list"),
            max_count=10.0,
        )
        buff_target_feat = self._extract_target_feature(
            frame_state,
            hero_pos,
            keys=("buffs", "buff_list", "speed_buffs", "speed_buff_list"),
            max_count=2.0,
        )

        # Wall collision penalty / 撞墙（或被障碍阻挡）惩罚
        wall_collision_penalty = 0.0
        if self.last_hero_pos is not None and isinstance(last_action, (int, np.integer)):
            action_id = int(last_action)
            if 0 <= action_id < 16:
                dx = hero_pos["x"] - self.last_hero_pos["x"]
                dz = hero_pos["z"] - self.last_hero_pos["z"]
                actual_disp = float(np.sqrt(dx * dx + dz * dz))

                # Context-aware penalty mitigation: 
                # If a treasure is nearby and visible, we mitigate the wall penalty 
                # to allow picking up treasures near boundaries.
                # 环境感知豁免：如果附近有宝箱，降低撞墙惩罚，允许在墙边捡宝箱。
                is_scavenging = bool(treasure_target_feat[0] > 0.5 and treasure_target_feat[4] < 0.15)
                current_wall_penalty = WALL_COLLISION_PENALTY_MOVE * (0.2 if is_scavenging else 1.0)

                # Approx expected displacement by action type.
                if action_id < 8:
                    expected_disp = 1.0
                    if actual_disp < 0.35 * expected_disp:
                        wall_collision_penalty = current_wall_penalty
                else:
                    expected_disp = 10.0 if action_id in (8, 10, 12, 14) else 8.0
                    if actual_disp < 0.30 * expected_disp:
                        wall_collision_penalty = WALL_COLLISION_PENALTY_FLASH  # 闪现撞墙仍保持高惩罚

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

        # Legal action mask (16D) / 合法动作掩码
        action_dim = Config.ACTION_NUM
        legal_action = [1] * action_dim
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(action_dim, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if 0 <= int(a) < action_dim}
                legal_action = [1 if j in valid_set else 0 for j in range(action_dim)]

        if sum(legal_action) == 0:
            legal_action = [1] * action_dim

        # Progress features (2D) / 进度特征
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        # Exploration reward with a 10-step anchor window.
        # 探索奖励：每10步记录锚点，随后10步鼓励远离锚点探索。
        if self.explore_anchor_pos is None or self.explore_window_steps >= EXPLORE_ANCHOR_INTERVAL:
            self.explore_anchor_pos = {"x": hero_pos["x"], "z": hero_pos["z"]}
            self.explore_window_steps = 0
            self.explore_last_anchor_dist = 0.0
            self.explore_window_max_dist = 0.0

        anchor_dx = hero_pos["x"] - self.explore_anchor_pos["x"]
        anchor_dz = hero_pos["z"] - self.explore_anchor_pos["z"]
        current_anchor_dist = float(np.sqrt(anchor_dx * anchor_dx + anchor_dz * anchor_dz))
        target_visible = bool(treasure_target_feat[0] > 0.5 or buff_target_feat[0] > 0.5)
        explore_scale = EXPLORE_TARGET_VISIBLE_SCALE if target_visible else 1.0
        explore_reward = explore_scale * EXPLORE_MOVE_AWAY_COEF * max(0.0, current_anchor_dist - self.explore_last_anchor_dist)

        self.explore_window_max_dist = max(self.explore_window_max_dist, current_anchor_dist)
        self.explore_last_anchor_dist = current_anchor_dist
        self.explore_window_steps += 1

        if self.explore_window_steps >= EXPLORE_ANCHOR_INTERVAL:
            if (not target_visible) and step_norm > 0.2 and self.explore_window_max_dist < EXPLORE_STAY_DIST_THRESHOLD:
                explore_reward += EXPLORE_WINDOW_STAY_PENALTY
            self.explore_anchor_pos = None
            self.explore_window_steps = 0
            self.explore_last_anchor_dist = 0.0
            self.explore_window_max_dist = 0.0

        # Concatenate features / 拼接特征
        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                map_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
                treasure_target_feat,
                buff_target_feat,
            ]
        )

        # Step reward / 即时奖励
        cur_min_dist_norm = 1.0
        monster_detected = False
        for m_feat in monster_feats:
            if m_feat[0] > 0:
                cur_min_dist_norm = min(cur_min_dist_norm, m_feat[4])
                monster_detected = True
        
        # [动态调整] 安全系数：当距离小于阈值时，生存避怪权重提升
        safety_status = 1.0
        if cur_min_dist_norm < SAFE_DIST_THRESHOLD:
            # 距离越近，权重放大倍数越大
            safety_status = DANGER_ZONE_PENALTY_SCALE * (1.0 + (SAFE_DIST_THRESHOLD - cur_min_dist_norm) / SAFE_DIST_THRESHOLD)

        # [动态调整] 探索渴望：长时间没拿宝箱则渴望上升；探索距离过长则渴望自然衰减
        if self.last_hero_pos is not None:
             move_dist = float(np.sqrt((hero_pos["x"]-self.last_hero_pos["x"])**2 + (hero_pos["z"]-self.last_hero_pos["z"])**2))
             self.total_exploration_dist += move_dist

        explore_desire = (1.0 + self.no_treasure_steps/150.0) / (1.0 + self.total_exploration_dist/800.0)
        explore_desire = float(np.clip(explore_desire, 0.5, 2.5))

        # 初始步修正：如果第1步没看到怪且距离从1.0变小，判定为逻辑误差，修正为不产生惩罚
        if self.step_no <= 1 and not monster_detected:
            cur_min_dist_norm = 1.0

        phase = _phase_mix(step_norm)
        monster_dist_coef = MONSTER_DIST_COEF * (1.0 + (EARLY_MONSTER_DIST_SCALE - 1.0) * phase) * safety_status

        # 动态宝箱冒险倾向：如果在危险区域，尝试减小宝箱诱惑
        risk_adjustment = 1.0 if safety_status < 1.1 else 0.65
        treasure_pick_coef = TREASURE_PICK_COEF * (1.0 + (EARLY_TREASURE_PICK_SCALE - 1.0) * phase) * risk_adjustment
        treasure_approach_coef = TREASURE_APPROACH_COEF * (1.0 + (EARLY_TREASURE_APPROACH_SCALE - 1.0) * phase) * risk_adjustment
        buff_pick_reward_cfg = BUFF_PICK_REWARD * (1.0 + (EARLY_BUFF_PICK_SCALE - 1.0) * phase) * risk_adjustment
        no_treasure_penalty_cfg = NO_TREASURE_PENALTY * (1.0 + (EARLY_NO_TREASURE_PENALTY_SCALE - 1.0) * phase)

        prev_min_monster_dist_norm = self.last_min_monster_dist_norm
        prev_min_treasure_dist_norm = self.last_min_treasure_dist_norm

        survive_reward = SURVIVE_REWARD
        dist_shaping = monster_dist_coef * (cur_min_dist_norm - prev_min_monster_dist_norm)
        
        # 探索分应用 explore_desire 动态权重
        actual_explore_reward = explore_reward * explore_desire

        # Treasure/score shaping / 宝箱与得分塑形
        total_score, treasure_collected, treasure_remaining = self._extract_env_metrics(observation)
        score_delta = max(0.0, total_score - self.last_total_score)

        # Support both treasure semantics:
        # 1) collected counter increases when picking treasure
        # 2) remaining counter decreases when picking treasure
        treasure_delta_from_collected = max(0.0, treasure_collected - self.last_treasure_collected)
        treasure_delta_from_remaining = 0.0
        if treasure_remaining is not None and self.last_treasure_remaining is not None:
            treasure_delta_from_remaining = max(0.0, self.last_treasure_remaining - treasure_remaining)
        treasure_delta = max(treasure_delta_from_collected, treasure_delta_from_remaining)

        # Treasure proximity shaping / 接近宝箱塑形（距离变小给正奖励）
        cur_min_treasure_dist_norm = self._extract_min_treasure_dist_norm(frame_state, hero_pos)
        treasure_approach_reward = treasure_approach_coef * (prev_min_treasure_dist_norm - cur_min_treasure_dist_norm)

        # Buff pickup reward / 拾取加速增益奖励（buff剩余时间从0上升）
        buff_pick_reward = 0.0
        if self.last_buff_remain_norm <= 1e-6 and buff_remain_norm > 1e-6:
            buff_pick_reward = buff_pick_reward_cfg

        # Encourage task objective stronger than before.
        # 提高任务导向强度：宝箱与得分增量权重上调。
        score_reward = SCORE_DELTA_COEF * score_delta
        treasure_reward = treasure_pick_coef * treasure_delta

        # [建议2] 阶梯式机会成本惩罚
        if treasure_delta > 0:
            self.no_treasure_steps = 0
        else:
            self.no_treasure_steps += 1

        no_treasure_penalty = 0.0
        if (
            self.no_treasure_steps >= NO_TREASURE_STEP_THRESHOLD
            and step_norm >= NO_TREASURE_PENALTY_START_PROGRESS
            and cur_min_dist_norm >= SAFE_MONSTER_DIST_NORM
        ):
            # 采用阶梯式逻辑
            if self.no_treasure_steps >= NO_TREASURE_STAGE2_STEP:
                 no_treasure_penalty = NO_TREASURE_STAGE2_PENALTY
            else:
                 no_treasure_penalty = no_treasure_penalty_cfg
        
        self.last_total_score = total_score
        self.last_treasure_collected = treasure_collected
        self.last_treasure_remaining = treasure_remaining
        self.last_buff_remain_norm = buff_remain_norm

        reward = [
            survive_reward
            + dist_shaping
            + score_reward
            + treasure_reward
            + treasure_approach_reward
            + buff_pick_reward
            + actual_explore_reward
            + no_treasure_penalty
            + wall_collision_penalty
        ]

        # Reward effective flash usage (actions 8~15):
        # If flash helps approach treasure or increases safety, add a small bonus.
        # 有效闪现奖励（动作8~15）：若闪现后更接近宝箱或更安全，给小额奖励。
        if isinstance(last_action, (int, np.integer)) and 8 <= int(last_action) <= 15:
            flash_effective = 0.0
            if prev_min_treasure_dist_norm - cur_min_treasure_dist_norm > 0.03:
                flash_effective += FLASH_EFFECTIVE_REWARD
            if cur_min_dist_norm - prev_min_monster_dist_norm > 0.02:
                flash_effective += 0.5 * FLASH_EFFECTIVE_REWARD
            reward[0] += flash_effective

        self.last_min_monster_dist_norm = cur_min_dist_norm
        self.last_min_treasure_dist_norm = cur_min_treasure_dist_norm
        self.last_hero_pos = {"x": hero_pos["x"], "z": hero_pos["z"]}

        return feature, legal_action, reward
