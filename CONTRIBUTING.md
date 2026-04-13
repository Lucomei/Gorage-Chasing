# 版本迭代与实验追踪规范

## 分支策略
- 主分支：main（保持可运行）
- 功能/实验分支命名：
  - feat/<short-topic>
  - fix/<short-topic>
  - exp/<short-topic>

## 提交信息规范
提交标题格式：
<type>: <summary>

可选 type：
- feat: 新功能或新策略
- fix: 缺陷修复
- refactor: 重构（不改变行为）
- docs: 文档更新
- chore: 工程维护
- exp: 训练实验与调参迭代

示例：
- exp: tune treasure reward weight to 1.2
- fix: handle empty legal_action mask

## 实验记录最小字段
每次实验至少记录以下内容（写在 commit message 正文或 PR 描述中）：
- exp_id: 唯一实验编号（如 exp-20260413-01）
- 目标: 本次想验证什么
- 变更: 改了哪些文件、哪些参数
- 配置: 地图、max_step、buff/treasure 相关配置
- 指标: 平均宝箱数、胜率、平均步数、总分
- 结论: 是否保留该改动

## 推荐提交节奏
- 一次提交只做一个主题改动（奖励、特征、超参数分开）
- 关键实验一条提交，便于回滚和对比
- 每个阶段打 tag

## Tag 规则
- 阶段基线：baseline-vX
- 关键实验：exp-vX
- 可复现最好模型：best-vX

示例：
- git tag baseline-v1
- git tag exp-v3
- git tag best-v1

## 常用命令
- 查看日志：git log --oneline --decorate --graph -20
- 对比实验：git diff <commitA> <commitB>
- 回退到指定提交（仅本地验证建议）：git checkout <commit>
- 打标签并推送：
  - git tag exp-v4
  - git push origin exp-v4
