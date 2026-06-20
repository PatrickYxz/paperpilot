# 2026-06-20 GitHub Push For Mac Transition Plan

## 背景与目标

用户新增一台 Mac，计划后续在 Mac 上继续开发 PaperPilot，因此需要把当前 Windows 工作区中的项目状态推送到 GitHub 远端 `origin`，便于在新设备上拉取。

## 约束条件

- 当前远端为 `https://github.com/PatrickYxz/paperpilot.git`。
- 当前分支为 `codex/qasper-eval-upgrade`。
- `.env`、`.venv`、IDE 缓存、个人目录、日志和大部分本地数据目录不应被提交。
- 用户要求“全部 push”，本次按当前 Git 工作区中未忽略的修改和新增文件作为提交范围处理。
- GitHub CLI `gh` 当前不可用，因此只执行 Git 提交和推送；不依赖 `gh` 创建 PR。

## 分步骤执行计划

1. 核对仓库状态、当前分支、远端地址和未提交文件列表。
2. 检查 `.gitignore` 是否覆盖本地密钥、虚拟环境、IDE 文件、日志和本地数据产物。
3. 对将要提交的文件做敏感词和大文件风险检查，重点排除 token、key、secret、password 等明显泄露风险。
4. 运行相关测试，优先覆盖当前变更涉及的 eval baseline、query planner、evidence selection 逻辑。
5. 使用 `git add -A` 暂存未忽略的当前工作区内容。
6. 创建一次说明性提交。
7. 将当前分支推送到 GitHub，并设置 upstream。
8. 汇报分支、提交、推送结果、验证情况和 Mac 端后续拉取方式。

## 验证方式

- `git status --short --branch`
- `git diff --stat`
- `git check-ignore`
- 相关 `pytest`
- `git push -u origin codex/qasper-eval-upgrade`

## 风险与待确认项

- 如果 `git push` 需要 GitHub 凭据且本机未登录，可能需要用户完成认证。
- 如果测试依赖本地 venv，普通沙箱可能误报找不到 Python；按项目说明需要提权运行。
- 由于用户明确要求全部推送，本次不会逐个文件二次确认，但会在提交前做敏感文件检查。
