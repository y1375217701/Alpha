# 配置本地 Hermes Agent

## 背景 / 目标

在本机部署独立 Hermes Agent，复用 CLIProxyAPI 提供的 GPT-6.1 SOL，并向 Hermes 提供 Alpha 的 MCP 工具。

## 任务分解（checklist）

- [x] 安装隔离的 Hermes Python 运行环境。
- [x] 配置本地 API、模型代理和 Alpha MCP。
- [x] 验证 Alpha 进入 Agent 模式，并执行一次带 MCP 工具的连通性测试。

## 验收标准

Hermes API 在本机可达。认证后可调用 GPT-6.1 SOL，Hermes 能读取 Alpha MCP 数据，Alpha 状态返回 `mode=agent`。
