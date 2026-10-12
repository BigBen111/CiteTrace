# CiteTrace

论文证据问答系统：上传 PDF，针对论文提问，查看原文页码与逐条结论核查结果。Qwen 负责回答，GLM 负责核查，DeepSeek 在调用失败时备用。引用是否存在类问题优先使用全文扫描。

适合个人使用、小组内部阅读和课堂展示。当前采用**单用户、单任务执行**设计，不是多租户服务；不要启动多个 Uvicorn worker 或多副本。模型核查仍可能出错，重要结论需要人工复核。

## 1. 准备服务器

推荐 Linux 服务器，建议至少 2 核、2 GB 内存（实际需求取决于 PDF 大小）。需要 Git、Python 3.10+（初始化脚本使用）、Docker Engine 和 Docker Compose v2。安装 Docker 请按 [Docker 官方说明](https://docs.docker.com/engine/install/) 操作。

先确认命令可用：

```bash
git --version
python3 --version
docker version
docker compose version
```

服务器需要能访问镜像仓库、Python 包索引及三家模型 API。请准备 Qwen、GLM、DeepSeek 的 API Key；网页能启动不代表模型账户有额度。项目不会提供或共享作者的密钥。

## 2. 下载并初始化

以下命令都在你自己的服务器执行：

```bash
git clone https://github.com/BigBen111/CiteTrace.git
cd CiteTrace
python3 deploy/init_config.py
```

脚本会创建 `config/.env`、`data/` 和 `artifacts/`，自动生成两个不同的随机登录口令。重复执行不会覆盖已有配置。

用你熟悉的编辑器打开配置，例如：

```bash
nano config/.env
```

必须填写以下三项：

| 配置 | 用途 |
| --- | --- |
| `QWEN_API_KEY` | 回答及必要时的检索规划 |
| `GLM_API_KEY` | 对回答逐条核查 |
| `DEEPSEEK_API_KEY` | 主模型出错时备用 |

同时检查 `QWEN_MODEL`、`GLM_MODEL`、`DEEPSEEK_MODEL` 是自己账户可用的模型 ID。示例默认值不是权限或余额保证；如模型不可用，请修改型号及对应服务地址。当前问答流程初始化会读取三家配置，请全部填写。

`CITETRACE_ADMIN_TOKEN` 是管理员登录口令，`CITETRACE_USER_TOKEN` 是普通用户口令，**不是模型 API Key**。请从自己的配置文件中读取并妥善保管，不要把配置文件或口令截图提交到 GitHub。普通用户共享每日提问额度，默认 20 次；管理员不受该次数限制，但仍一次只处理一个任务。

## 3. 启动并检查

```bash
docker compose up -d --build
docker compose ps
curl --fail http://127.0.0.1:8000/health
```

健康接口应返回 `{"status":"ok"}`，容器稍后显示 `healthy`。它只证明服务存活，不会调用模型检查余额。

默认端口仅绑定服务器的 `127.0.0.1:8000`，不能直接通过公网 IP 访问，这是为了避免把登录口令通过明文 HTTP 暴露出去。配置、论文和运行记录挂载到宿主机目录，重建容器不会清除它们。

### 没有域名：通过 SSH 安全访问

在**自己的电脑**终端执行，把 `YOUR_USER` 和 `YOUR_SERVER_IP` 换成实际值：

```bash
ssh -N -L 18000:127.0.0.1:8000 YOUR_USER@YOUR_SERVER_IP
```

保持终端连接，浏览器打开 **http://127.0.0.1:18000**。只需要服务器 SSH 端口可达，不用在安全组开放 8000。SSH 若使用其他端口，请加 `-p 端口号`。

### 有域名：公网 HTTPS 访问

把域名 A 记录指向服务器公网 IPv4；如果设置了 AAAA 记录，也必须指向可用的 IPv6。确保 80、443 未被其他服务占用，在云安全组和系统防火墙开放**入方向 TCP 80、443**。8000 不需要对公网开放。

在服务器的项目目录执行，替换示例域名：

```bash
export CITETRACE_DOMAIN=paper.example.com
docker compose -f docker-compose.yml -f docker-compose.https.yml up -d --build
```

附加服务 Caddy 会申请和续期证书，将 HTTPS 请求转给应用。浏览器打开 `https://paper.example.com`。若已有 Nginx/Caddy，可直接让现有 HTTPS 代理转发到 `127.0.0.1:8000`，不用重复启动代理。

HTTPS 部署后，管理、升级或停止服务时继续使用上述两个 `-f` 参数；新终端先重新设置 `CITETRACE_DOMAIN`。域名只是你自己的部署配置，无需修改源代码。

## 4. 第一次使用

1. 点击页面上的“登录”，输入管理员口令。
2. 在设置中确认三家服务地址、模型和密钥配置；密钥输入框留空表示保留已有值。
3. 上传自己有权使用的、可提取文字的 PDF（最大 50 MB），等待解析完成。扫描版 PDF 需要先做 OCR。
4. 针对这篇论文提问，例如“论文训练使用了什么硬件？请提供原文证据。”
5. 点击答案中的证据编号查看摘录和页码，在核查区域查看每条陈述的结论。

启动时没有预装论文索引，页面示例论文未就绪时请先上传 PDF。论文摘录及问题会发送给已配置的模型服务，敏感文档上传前请确认使用权限。实际提问会产生模型调用费用。

## 5. 日常维护与排错

```bash
# 状态与最近日志
docker compose ps
docker compose logs --tail=100 citetrace

# 修改 config/.env 后重启
docker compose restart citetrace

# 更新代码（先备份，确保当前没有任务运行）
git pull --ff-only
docker compose up -d --build

# 停止服务，保留持久化目录
docker compose down
```

使用 HTTPS 附加配置时，以上命令中的 `docker compose` 要替换为 `docker compose -f docker-compose.yml -f docker-compose.https.yml`。

备份 `config/`、`data/`、`artifacts/`；备份包含密钥和论文内容，必须私密保存。建议停止服务后备份，恢复时保持这三个目录对应，随后重新启动。不要删除挂载目录或使用 `down -v` 清除 HTTPS 证书卷。

| 现象 | 检查方法 |
| --- | --- |
| 公网 IP:8000 打不开 | 默认只监听本机；使用 SSH 隧道或 HTTPS 方案 |
| 登录失败 | 使用本机 `config/.env` 的登录口令，不是模型密钥；修改后重启 |
| 首页正常，但问答失败 | 检查三家密钥、模型 ID、账户额度及出网连接；看任务错误和服务日志 |
| 上传失败或 413 | 文件不得超过 50 MB；自建代理也需允许相应请求大小 |
| PDF 没有可用文字 | 先 OCR；当前不内置扫描件 OCR |
| 提示已有任务 | 等当前任务完成，勿重复提交；默认总超时 300 秒 |
| 重启后任务中断 | 系统不会自动重放可能收费的请求，需手动重新提问 |
| HTTPS 证书失败 | 检查 DNS、80/443 入站、端口占用及 Caddy 日志 |
| 配置无法保存 | 检查挂载目录写权限；不要用 `chmod 777` 暴露密钥 |

## 本地开发与测试（不使用 Docker）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,web]"
python3 deploy/init_config.py
export CITETRACE_CONFIG_DIR="$PWD/config"
uvicorn citetrace.server:app --host 127.0.0.1 --port 8000
```

后台运行请自行配置进程管理。可选 systemd 模板在 `deploy/citetrace.service`：默认安装目录 `/opt/citetrace`、服务用户 `citetrace`，需先创建该用户、准备虚拟环境并赋予配置和数据目录的写权限，再按实际路径调整模板。Docker 用户不需要安装该服务。

```bash
python -m pytest -q
```

回归测试使用本地假模型服务，不消耗付费模型额度。GitHub Actions 同时检查单元测试、Compose 配置、Docker 构建和容器健康接口；不会向真实供应商发送问题。

CLI 使用项目根目录 `.env`，与上述 Web 的 `config/.env` 路径不同；若需要 CLI，请单独参考 `.env.example` 配置，不要把真实配置提交到仓库。

```bash
citetrace ingest ./paper.pdf --paper-id demo-paper
citetrace ask --paper-id demo-paper "What hardware was used for training?"
```

## 系统原理与边界

一般问题依次经过：本地检索 → 必要时模型规划检索 → 原文摘录校验 → Qwen 回答 → 结论与证据绑定 → GLM 核查 → 保守汇总。主模型 API 或结构化输出异常时可使用 DeepSeek 备用；“证据不足”不是需要换模型强行作答的错误。

- 一条结论必须关联原文证据；缺少证据、证据冲突或核查失败时可能拒绝作出确定回答。
- 同一条进行中的提问会复用任务；不同提问在当前任务完成前被拒绝。
- 默认任务超时 300 秒；可通过 `CITETRACE_JOB_TIMEOUT_SECONDS` 调整（10–1800 秒）。
- 上传或重启不会悄悄重放付费请求。PDF、问题、原文摘录与调用轨迹留在本机，不应公开。
- 前端极短阶段的显示时间包含平滑展示值；性能分析请使用任务 API 的实测 `stage_history`，不要将动画时间当作性能测量。

源码位于 `citetrace/`，前端位于 `web/`，提示词位于 `prompts/`；详细设计见 [架构说明](docs/ARCHITECTURE.md)，评测入口见 [evaluation](evaluation/README.md)。课程材料、原始运行记录和私密配置不包含在默认发布内容中。
