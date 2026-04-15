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
SURVIVE_REWARD = 0.025  # [四号强化] 生存底薪提高，极大增强活下去的欲望 (0.015 -> 0.025)
MONSTER_DIST_COEF = 0.35  # [四号强化] 脱逃奖励巨幅上调，鼓励极限拉扯拿跑酷分 (0.22 -> 0.35)
SCORE_DELTA_COEF = 0.05
TREASURE_PICK_COEF =  8.5 # [四号强化] 拾取意愿核弹级上调 (5.2 -> 8.5)
TREASURE_APPROACH_COEF = 1.0 # [四号强化] 更强的磁吸效应 (0.75 -> 1.0)
BUFF_PICK_REWARD = 3.5  # [四号强化] 增益雷达强化，Buff优先级媲美宝箱 (1.25 -> 3.5)
BUFF_APPROACH_COEF = 0.8  # [新增] 引入 Buff 吸力
FLASH_EFFECTIVE_REWARD = 0.45
WALL_COLLISION_PENALTY_MOVE = -0.28  # 【防溜边】加重物理墙体的撞击痛感，逼迫模型远离空气墙
WALL_COLLISION_PENALTY_FLASH = -1.0  # 【严禁乱闪】闪现撞墙惩罚极大化，彻底避免浪费
EXPLORE_ANCHOR_INTERVAL = 5  # 再次缩短：从 6 降低到 5，更频繁地检测“发呆”
EXPLORE_MOVE_AWAY_COEF = 0.12  # 全面增强：从 0.08 提升至 0.12，高额位移奖励
EXPLORE_TARGET_VISIBLE_SCALE = 0.20
EXPLORE_STAY_DIST_THRESHOLD = 3.5  # 环境感知：更严苛的发呆距离判定
EXPLORE_WINDOW_STAY_PENALTY = -0.25  # 【重罚发呆】原地打转的代价提高

# Dynamic Risk Management / 动态风险控制
SAFE_DIST_THRESHOLD = 0.26  # [四号强化] 警戒线外扩，提早开溜保证高活存率 (0.20 -> 0.26)
DANGER_ZONE_PENALTY_SCALE = 3.5  # 【关键修正】大幅下调权重，防止原地等死 (从 6.0 降至 3.5)
EXPLORE_DESIRE_BASE = 1.0

# Visit Map (Anti-Backtrack) / 路径热度图（防止回头路）
VISIT_MAP_RESOLUTION = 8.0  # 地图网格分辨率（128/8 = 16x16个格子）
VISIT_COUNT_PENALTY_COEF = -0.005  # 恢复强度：加大对老路的厌恶感
VISIT_COUNT_MAX_PENALTY = -0.15  # 提高上限：大幅增加区域周转压力

# 区域厌倦感 (Regional Satiety) / 宏观热力图
REGION_SIZE = 32.0 # 4x4 的大区域（128/32）
REGION_SATIETY_THRESHOLD = 250 # 提高阈值：从 150 增加到 250，给更多时间在这个区域探索
REGION_SATIETY_PENALTY = -0.15 # 略微加重惩罚：跨区意愿更强

# 全局指南针灵敏度 (Compass Sensitivity)
COMPASS_REWARD_COEF = 0.16  # 【全图雷达】进一步强化方向引导
COMPASS_ANGLE_THRESHOLD = 0.3  # 约 17 度以内视为“对准”
COMPASS_DISTANCE_CUTOFF = 0.9  # 【几乎全图】除了极远角，所有宝箱都有引力

# 终极版新增：闪现门控与死胡同感知
FLASH_CONSERVE_REWARD = 0.008  # 憋着闪现不用的持续小奖
EMERGENCY_FLASH_DIST = 0.15    # 判定为“保命闪现”的距离阈值
DEAD_END_PENALTY = -0.45       # 死胡同（三面环墙）重罚

# 避怪机制增强 (Monster Avoidance)

# 避怪机制增强 (Monster Avoidance)
MONSTER_DANGER_ALPHA = 1.8   # 【平滑】下调指数项，防止惩罚在边界处突变 (从 2.5 降至 1.8)
STAGNATION_PENALTY = -0.18 # 专门针对速度极慢（卡顿、罚站）的额外惩罚
MIN_VELOCITY_THRESHOLD = 0.05   # 判定为卡顿的速度阈值

# Opportunity-cost penalty (gated) / 机会成本惩罚（条件触发）
NO_TREASURE_STEP_THRESHOLD = 100  # 收紧：回到 100 步，让它快点去找宝箱
NO_TREASURE_PENALTY = -0.08  # 加重：从 -0.06 到 -0.08
NO_TREASURE_STAGE2_STEP = 200  # 收紧：回到 200 步
NO_TREASURE_STAGE2_PENALTY = -0.15  # 加重：回到 -0.15
SAFE_MONSTER_DIST_NORM = 0.28  # 微调：从 0.3 降到 0.28，更贴脸才判定为绝对危险
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
        self.max_step = 1000
        self.last_min_monster_dist_norm = 1.0
        self.last_total_score = 0.0
        self.last_treasure_collected = 0.0
        self.last_treasure_remaining = None
        self.last_buff_remain_norm = 0.0
        self.last_min_treasure_dist_norm = 1.0
        self.last_min_buff_dist_norm = 1.0 # 记录 Buff 的最近距离
        self.no_treasure_steps = 0
        self.total_exploration_dist = 0.0
        self.last_hero_pos_raw = None
        self.last_hero_pos = None
        self.last_action = 0
        self.explore_anchor_pos = None
        self.explore_window_steps = 0
        self.explore_last_anchor_dist = 0.0
        self.explore_window_max_dist = 0.0
        self.visit_map = np.zeros((16, 16), dtype=np.float32)
        # 怪物的残影记忆
        self.monster_memory = [None, None]
        # 跨区探索记忆
        self.current_region_id = -1
        self.region_stay_steps = 0
        self.region_visit_history = np.zeros((4, 4), dtype=np.float32) # 全局宏观足迹 (4x4)

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
        
        # 提取地图 ID 用于动态均衡化处理
        map_id = self._to_float(env_info.get("map_id", 0.0), 0.0)
        is_hard_map = map_id in [1.0, 2.0, 3.0] # 确定 1, 2, 3 为难图

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

        # --- [地图均衡化 3.0：二号灵魂 + 三号防线] 动态奖励缩放与难度归一化 ---
        # 恢复二号技师的强力均衡逻辑：赋予难图更高的奖励回报
        map_scaling = 1.0
        wall_penalty_scale = 1.0
        stay_threshold_scale = 1.0
        
        # 针对 Map 2 进行精细化“止血”：作为唯一的薄弱环节
        if map_id == 2.0: 
            map_scaling = 3.2 # 极高补偿：诱惑智能体在复杂窄道中移动
            wall_penalty_scale = 0.4 # 降低撞墙痛感：防止因为怕撞墙而不敢进窄道捡宝箱
            stay_threshold_scale = 0.5 # 严苛的发呆判定：逼它在死角里动起来
        # 恢复二号版本的经典“难图组”增强 (1/3/4/6/7)
        elif map_id in [1.0, 3.0, 4.0, 6.0, 7.0]:
            map_scaling = 1.8 
            wall_penalty_scale = 0.7
            stay_threshold_scale = 0.8
        # 优势地图 (8/9/10)：保持适度压制，防止整体模型被这类刷分图带偏
        elif map_id in [8.0, 9.0, 10.0]:
            map_scaling = 0.85
            wall_penalty_scale = 1.2
            stay_threshold_scale = 1.1
            
        explore_move_away_coef = EXPLORE_MOVE_AWAY_COEF * map_scaling
        wall_collision_penalty_move = WALL_COLLISION_PENALTY_MOVE * wall_penalty_scale
        explore_stay_dist_threshold = EXPLORE_STAY_DIST_THRESHOLD * stay_threshold_scale
        
        # [四号加强] 狂暴寻宝模式：拿到 Buff 时，胆量和指路引力双倍提升
        is_frenzy_mode = bool(buff_remain_norm > 1e-6)
        dynamic_compass_coef = COMPASS_REWARD_COEF * (2.0 if is_frenzy_mode else 1.0)
        
        # [四号加强] 威胁感知指南针 (Safe-Path Compass)
        compass_reward = 0.0
        if treasure_target_feat[0] > 0.5: # 如果可见宝箱
            dx, dz = treasure_target_feat[2], treasure_target_feat[3]
            dist_norm = treasure_target_feat[4]
            # 计算当前朝向与宝箱方向的夹角
            if isinstance(last_action, (int, np.integer)) and 0 <= int(last_action) < 8 and dist_norm < COMPASS_DISTANCE_CUTOFF:
                angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
                target_angle = np.arctan2(dz, dx)
                move_angle = angles[int(last_action)]
                angle_diff = np.abs(np.arctan2(np.sin(target_angle - move_angle), np.cos(target_angle - move_angle)))
                
                # 寻找视野内最近怪物的相对方向
                closest_m_dist = 1.0
                closest_m_angle_diff = np.pi
                for m in frame_state.get("monsters", []):
                    if m.get("is_in_view", 0) > 0:
                        mx, mz = m["pos"]["x"] - hero_pos["x"], m["pos"]["z"] - hero_pos["z"]
                        m_dist = np.sqrt(mx**2 + mz**2) / (MAP_SIZE * 1.41)
                        if m_dist < closest_m_dist:
                            closest_m_dist = m_dist
                            m_angle = np.arctan2(mz, mx)
                            closest_m_angle_diff = np.abs(np.arctan2(np.sin(m_angle - move_angle), np.cos(m_angle - move_angle)))

                # 安全判断：如果不处于无敌狂暴，且怪物距离很近（<0.26），且我正面对着怪走（<45度）
                if not is_frenzy_mode and closest_m_dist < 0.26 and closest_m_angle_diff < (np.pi / 4):
                    # 否决直线寻宝，因为怪物挡路
                    compass_reward = -0.05 # 负反馈：此路不通，快绕路
                elif angle_diff < COMPASS_ANGLE_THRESHOLD:
                    compass_reward = dynamic_compass_coef * (1.0 - dist_norm) # 正常发奖

        # [新增] 专门针对“卡顿/罚站”的惩罚
        stagnation_penalty = 0.0
        # 获取当前是否在捡宝箱或Buff（scavenging 状态全局计算）
        is_scavenging_treasure = bool(treasure_target_feat[0] > 0.5 and treasure_target_feat[4] < 0.15)
        is_scavenging_buff = bool(buff_target_feat[0] > 0.5 and buff_target_feat[4] < 0.15)
        is_scavenging = is_scavenging_treasure or is_scavenging_buff
        
        # [终极版新增] 死胡同感知：检测周围墙壁情况
        dead_end_penalty = 0.0
        wall_count = 0
        _early_monster_detected = any(m.get("is_in_view", 0) for m in frame_state.get("monsters", []))
        if map_info is not None:
            center = len(map_info) // 2
            # 检测前后左右四邻域
            offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dr, dc in offsets:
                if 0 <= center+dr < len(map_info) and 0 <= center+dc < len(map_info[0]):
                    if map_info[center+dr][center+dc] != 0:
                        wall_count += 1
            if wall_count >= 3: # 被三面环墙，高度怀疑是死胡同
                dead_end_penalty = DEAD_END_PENALTY * (1.0 if not _early_monster_detected else 0.5) # [优化] 有怪时降低死胡同惩罚，生命高于风水

        if self.last_hero_pos is not None:
            dx_stag = hero_pos["x"] - self.last_hero_pos["x"]
            dz_stag = hero_pos["z"] - self.last_hero_pos["z"]
            velocity = np.sqrt(dx_stag*dx_stag + dz_stag*dz_stag)
            
            # 提前计算是否属于绝境逃生（被怪贴脸卡角落），此时如果停滞是因为被墙卡住，不要追加卡顿惩罚
            in_danger_escape_stag = (self.last_min_monster_dist_norm < 0.12 and wall_count >= 2)
            if velocity < MIN_VELOCITY_THRESHOLD and not (is_scavenging or in_danger_escape_stag):
                stagnation_penalty = STAGNATION_PENALTY

        # 记录上一轮动作相关（用于检测撞墙）
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
                
                # 使用地图感知后的 penalty
                # [四号修复] 溜边症结：非死路/非狭窄地形直接逃跑，禁止溜边！
                # 只有在极度危险贴脸（< 0.12） 且 身处狭窄带/死角（墙数>=2）被迫走位时，才豁免撞墙惩罚
                in_danger_escape = (self.last_min_monster_dist_norm < 0.12 and wall_count >= 2)
                mitigation = 0.2 if (is_scavenging or in_danger_escape) else 1.0
                current_wall_penalty = wall_collision_penalty_move * mitigation

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
                
                # [四号技师] 残影记忆更新
                if is_in_view:
                    self.monster_memory[i] = {"x": m_pos["x"], "z": m_pos["z"], "speed": m.get("speed", 1), "step": self.step_no}
                
                if is_in_view:
                    m_x_norm = _norm(m_pos["x"], MAP_SIZE)
                    m_z_norm = _norm(m_pos["z"], MAP_SIZE)
                    m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)

                    # Euclidean distance / 欧式距离
                    raw_dist = np.sqrt((hero_pos["x"] - m_pos["x"]) ** 2 + (hero_pos["z"] - m_pos["z"]) ** 2)
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                else:
                    # 检查残影记忆（短时：最近30步以内）
                    mem = self.monster_memory[i]
                    if mem is not None and self.step_no - mem["step"] <= 30:
                        m_x_norm = _norm(mem["x"], MAP_SIZE)
                        m_z_norm = _norm(mem["z"], MAP_SIZE)
                        m_speed_norm = _norm(mem["speed"], MAX_MONSTER_SPEED)
                        raw_dist = np.sqrt((hero_pos["x"] - mem["x"]) ** 2 + (hero_pos["z"] - mem["z"]) ** 2)
                        
                        # 随时间距离衰减（时间越久，威胁感越小）
                        decay_factor = 1.0 + (self.step_no - mem["step"]) / 30.0 
                        dist_norm = min(1.0, _norm(raw_dist * decay_factor, MAP_SIZE * 1.41))
                        # 用 0.5 或者一个介于 0~1 的值告知网络：这是残影不是真身
                        is_in_view = 0.5 * max(0.0, 1.0 - (self.step_no - mem["step"])/30.0)
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
        # 使用地图感知后的 explore_move_away_coef
        explore_reward = explore_scale * explore_move_away_coef * max(0.0, current_anchor_dist - self.explore_last_anchor_dist)

        self.explore_window_max_dist = max(self.explore_window_max_dist, current_anchor_dist)
        self.explore_last_anchor_dist = current_anchor_dist
        self.explore_window_steps += 1

        if self.explore_window_steps >= EXPLORE_ANCHOR_INTERVAL:
            # 使用地图感知后的 explore_stay_dist_threshold
            if (not target_visible) and step_norm > 0.2 and self.explore_window_max_dist < explore_stay_dist_threshold:
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
            # [新增：夹缝求生机制] 当被压迫在狭窄地形（2面及以上有墙）且怪物已经贴脸（距离<0.12），
            # 如果继续放大恐惧，AI将只能原地等死。此时必须“关闭恐惧限幅”，允许它通过走位或交闪现与怪物擦肩而过！
            if wall_count >= 2 and cur_min_dist_norm < 0.12:
                safety_status = DANGER_ZONE_PENALTY_SCALE * 0.3 # 大幅削弱恐惧，激发突破求生本能
            else:
                # 正常危险区：距离越近，权重放大倍数越大
                safety_status = DANGER_ZONE_PENALTY_SCALE * (1.0 + (SAFE_DIST_THRESHOLD - cur_min_dist_norm) / SAFE_DIST_THRESHOLD)
            
        # [终极版修复] 开局强力避让逻辑
        if self.step_no < 15:
            if wall_count >= 2 and cur_min_dist_norm < 0.15:
                # [特判] 出生即处于狭窄地形且附近有怪：保持勇敢，立刻寻找出路
                safety_status *= 0.2
            else:
                # 空旷开局：显著放大避怪权重，防止因为初始方向随机导致撞怪
                safety_status *= 1.5 

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

        prev_min_monster_dist_norm = self.last_min_monster_dist_norm
        prev_min_treasure_dist_norm = self.last_min_treasure_dist_norm

        survive_reward = SURVIVE_REWARD
        # 核心逻辑增强：在“看不见怪”的时候，额外强化主动跑动的欲望
        dist_diff = cur_min_dist_norm - prev_min_monster_dist_norm
        
        # 避怪机制增强：引入指数级惩罚，距离越近惩罚增长越快
        if cur_min_dist_norm < SAFE_DIST_THRESHOLD:
            # 引入指数项：(1.0 - norm)^alpha，当 norm 越小，惩罚越大
            dist_shaping = monster_dist_coef * dist_diff * ( (1.0 - cur_min_dist_norm) ** MONSTER_DANGER_ALPHA )
        else:
            dist_shaping = monster_dist_coef * dist_diff
        
        # 增加巡航激励：如果看不见怪，且产生了位移，给一个基于探索激励的额外“前进奖”
        # [四号修复] 防止溜边：如果看不见怪，除了要走，最好还要走向开阔地带
        if not monster_detected and explore_reward > 0.005:
            dist_shaping += (EXPLORE_MOVE_AWAY_COEF * 0.4)
            # 如果附近没有墙（非常空旷），额外表扬
            if wall_count == 0:
                dist_shaping += 0.02

        # Treasure proximity shaping / 接近宝箱塑形（距离变小给正奖励）
        cur_min_treasure_dist_norm = self._extract_min_treasure_dist_norm(frame_state, hero_pos)
        
        # [四号加强] Buff 接近磁吸：像宝箱一样对周围的Buff产生渴望
        cur_min_buff_dist_norm = buff_target_feat[4] if buff_target_feat[0] > 0.5 else 1.0
        prev_min_buff_dist_norm = self.last_min_buff_dist_norm
        
        # [动态调整] 宝箱冒险倾向：如果在严重危险区域，尝试减小宝箱诱惑
        # 针对反馈修复：只要怪物不是处于“抓捕即死”的超近距离，就允许为了宝箱进行博弈
        # 将 2.0 的严苛门槛放宽，允许在普通危险区捡宝箱
        risk_adjustment = 1.0 if safety_status < 1.6 else 0.85 
        treasure_pick_coef = TREASURE_PICK_COEF * (1.0 + (EARLY_TREASURE_PICK_SCALE - 1.0) * phase) * risk_adjustment
        
        # 贪婪豁免与路过顺手：如果就在旁边，风险直接无视（拿到再说）；如果是顺路经过的距离，增强顺手奖励
        if cur_min_treasure_dist_norm < 0.03:
             risk_adjustment = 1.0
             treasure_pick_coef *= 1.5 
        elif cur_min_treasure_dist_norm < 0.08 and cur_min_dist_norm > 0.15:
             # 如果不是极度危险（绝对不贴脸），而且宝箱都在视距内比较近的位置，顺手拿它的权重提升
             risk_adjustment = 1.0
             treasure_pick_coef *= 1.2
        
        # 修正：即使有风险，趋近宝箱的奖励也要保持竞争力，防止“路过不拿”
        treasure_approach_coef = TREASURE_APPROACH_COEF * (1.0 + (EARLY_TREASURE_APPROACH_SCALE - 1.0) * phase) * max(risk_adjustment, 0.85)
        buff_pick_reward_cfg = BUFF_PICK_REWARD * (1.0 + (EARLY_BUFF_PICK_SCALE - 1.0) * phase) * risk_adjustment
        no_treasure_penalty_cfg = NO_TREASURE_PENALTY * (1.0 + (EARLY_NO_TREASURE_PENALTY_SCALE - 1.0) * phase) # 修正变量名错误

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

        # 核心修正：加大“趋近”灵敏度，如果非常近了，额外加强吸引力，防止临门一脚不进。
        dist_diff_treasure = prev_min_treasure_dist_norm - cur_min_treasure_dist_norm
        if cur_min_treasure_dist_norm < 0.05 and dist_diff_treasure > 0:
            # 距离宝箱极近时（约 8.9 像素范围内），吸引力加倍
            treasure_approach_reward = treasure_approach_coef * dist_diff_treasure * 4.0
        else:
            treasure_approach_reward = treasure_approach_coef * dist_diff_treasure

        # [四号加强] Buff 趋近奖励
        dist_diff_buff = prev_min_buff_dist_norm - cur_min_buff_dist_norm
        buff_approach_reward = 0.0
        if dist_diff_buff > 0 and buff_target_feat[0] > 0.5:
            buff_approach_reward = BUFF_APPROACH_COEF * dist_diff_buff * 2.0

        # Buff pickup reward / 拾取加速增益奖励（buff剩余时间从0上升）
        buff_pick_reward = 0.0
        if self.last_buff_remain_norm <= 1e-6 and buff_remain_norm > 1e-6:
            buff_pick_reward = buff_pick_reward_cfg

        # Encourage task objective stronger than before.
        # 提高任务导向强度：宝箱与得分增量权重上调。
        score_reward = SCORE_DELTA_COEF * score_delta
        treasure_reward = treasure_pick_coef * treasure_delta

        # [建议2] 阶梯式机会成本惩罚
        if treasure_delta > 0 or buff_pick_reward > 0:
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
        self.last_min_buff_dist_norm = cur_min_buff_dist_norm

        # [终极版新增] 闪现保留奖励：如果憋着不用且环境安全，给一点小奖励
        flash_conserve_reward = 0.0
        if hero.get("flash_cooldown", 0) == 0 and cur_min_dist_norm > SAFE_DIST_THRESHOLD:
            flash_conserve_reward = FLASH_CONSERVE_REWARD

        reward = [
            survive_reward
            + dist_shaping
            + score_reward
            + treasure_reward
            + treasure_approach_reward
            + buff_pick_reward
            + buff_approach_reward
            + actual_explore_reward
            + no_treasure_penalty
            + wall_collision_penalty
            + compass_reward
            + stagnation_penalty
            + dead_end_penalty
            + flash_conserve_reward
        ]

        # Reward effective flash usage (actions 8~15):
        # 终极版闪现策略：只有紧急且有效的情况下才给奖励，平时开闪现给负反馈
        if isinstance(last_action, (int, np.integer)) and 8 <= int(last_action) <= 15:
            flash_reward = -0.10 # [优化] 降低门控惩罚 (-0.15 -> -0.10)，增加容错
            # 只有当怪物距离小于阈值探测为“保命闪现”时
            if prev_min_monster_dist_norm < EMERGENCY_FLASH_DIST:
                flash_reward = 0.05 # [优化] 给保命闪现一个基础正奖，鼓励生存本能
                
                # [新增：环境因素评级] 根据闪现前/后的环境险恶程度，给予巨额加分
                if wall_count >= 2:
                    flash_reward += 0.20 # 死路逃生：能在被两面墙+1个怪包夹时交出闪现，加倍表扬
                
                if treasure_delta > 0 or buff_pick_reward > 0:
                    flash_reward += 0.25 # 火中取栗：刀口舔血捡了宝箱/Buff还能用闪现逃脱，顶尖操作
                    
                if prev_min_treasure_dist_norm - cur_min_treasure_dist_norm > 0.03:
                    flash_reward += FLASH_EFFECTIVE_REWARD
                    
                escape_diff = cur_min_dist_norm - prev_min_monster_dist_norm
                if escape_diff > 0.02:
                    escape_bonus = 0.5 * FLASH_EFFECTIVE_REWARD
                    # 如果这一下闪现大幅拉开了差距，甚至脱离了紧急危险区，大幅增加奖励
                    if cur_min_dist_norm >= EMERGENCY_FLASH_DIST:
                        escape_bonus *= 1.8
                    flash_reward += escape_bonus
                    
            reward[0] += float(flash_reward)

        self.last_min_monster_dist_norm = cur_min_dist_norm
        self.last_min_treasure_dist_norm = cur_min_treasure_dist_norm
        self.last_hero_pos = {"x": hero_pos["x"], "z": hero_pos["z"]}

        # Anti-Backtrack: Update Visit Map and apply path penalty
        # 防止回头路：更新访问热度图并施加路径惩罚。
        grid_x = int(np.clip(hero_pos["x"] / VISIT_MAP_RESOLUTION, 0, 15))
        grid_z = int(np.clip(hero_pos["z"] / VISIT_MAP_RESOLUTION, 0, 15))
        
        # [动态巡逻半径设置] 怪物距离越远，防兜圈子的力度越大；越危险，则越包容原地小范围拉扯。
        visit_count = self.visit_map[grid_x, grid_z]
        path_penalty = 0.0
        
        # 将怪物的距离直接等价于“自由探索空间”。距离越远，安全系数越高，原地留恋扣分越重。
        dynamic_visit_coef = VISIT_COUNT_PENALTY_COEF * (1.0 + cur_min_dist_norm * 2.5) 
        dynamic_max_penalty = VISIT_COUNT_MAX_PENALTY * (1.0 + cur_min_dist_norm * 1.5)
        
        # < 0.15 属于极其危险的贴脸距离，此时完全停止路径惩罚，赋予极限拉扯走位空间
        new_area_bonus = 0.0
        if cur_min_dist_norm >= 0.15:
            # [四号修复] 前期探图刚需：如果前期（安全状态），对走老路的厌恶感翻 1.5 倍（原为4倍，导致它疯狂跑向没人走过的边缘迷宫）
            early_overdrive = 1.0
            if step_norm < 0.3 and cur_min_dist_norm > SAFE_DIST_THRESHOLD:
                early_overdrive = 1.5 
            
            path_penalty = max(dynamic_max_penalty * early_overdrive, visit_count * dynamic_visit_coef * early_overdrive)
            
            # [四号加强] 处女地开拓奖：安全时，如果踩进完全没有去过的网格，给予正反馈
            if visit_count == 0:
                new_area_bonus = 0.02 * (1.0 - step_norm) # 下调奖励，防止为了踩格子乱冲边缘
        
        # Update visit count (使用增量记录足迹)
        self.visit_map[grid_x, grid_z] += 1.0
        
        # --- [跨区探索激励 3.0] 宏观热力图动态化 ---
        reg_x = int(np.clip(hero_pos["x"] / 32.0, 0, 3))
        reg_z = int(np.clip(hero_pos["z"] / 32.0, 0, 3))
        reg_id = reg_x * 4 + reg_z
        
        if reg_id == self.current_region_id:
            self.region_stay_steps += 1
        else:
            self.current_region_id = reg_id
            self.region_stay_steps = 0 # 进入新区域，重置厌倦感
            
        region_penalty = 0.0
        
        # 动态区域厌倦感：如果怪物非常远（安全距离，例如大于0.5），强迫快速开荒（厌倦阈值直接减半）
        dynamic_satiety_threshold = REGION_SATIETY_THRESHOLD if cur_min_dist_norm < 0.5 else int(REGION_SATIETY_THRESHOLD * 0.5) 
        
        # 如果在同一个 32x32 的大区域待太久，且周围没有急需采集的宝箱
        if self.region_stay_steps > dynamic_satiety_threshold and not is_scavenging:
            region_penalty = REGION_SATIETY_PENALTY * (1.0 + cur_min_dist_norm) # 安全时，大区的厌世感加倍
            
        reward[0] += float(path_penalty + region_penalty + new_area_bonus)

        return feature, legal_action, reward
