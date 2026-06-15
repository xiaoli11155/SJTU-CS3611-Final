# 基于深度学习的加密流量实时识别与监控系统

## 一、项目目标

本项目面向 HTTPS/TLS 等加密网络流量场景，设计并实现一个“代理转发 + 特征提取 + 深度学习分类 + 实时监控”的端到端实验系统。系统不解密应用层负载，而是利用 TCP 流量在传输层可观察到的统计特征，对网络连接所属的应用类型进行识别。

通过本项目，重点完成以下目标：

1. 理解 TCP 连接机制以及代理服务器的基本工作原理。
2. 掌握加密流量在无法读取明文内容时的特征提取方法。
3. 使用包长度序列构建固定维度输入，并训练 1D-CNN 分类模型。
4. 将离线训练得到的模型集成到在线代理中，实现实时推理。
5. 通过 Web 看板展示近期流量的应用类型分布和预测记录。

## 二、实验要求

### 1. TCP/HTTP 代理服务器

使用 Python 原生 `socket` 实现一个 TCP/HTTP 代理服务器，能够接收本机客户端，例如浏览器，发起的 HTTP 或 HTTPS CONNECT 请求，并将流量转发至目标公网服务器。

代理服务器需要支持：

- 普通 HTTP 请求转发。
- HTTPS CONNECT 隧道建立。
- 客户端到服务器、服务器到客户端的双向转发。
- 多连接并发处理。
- 保持基本网络连通性，不破坏正常访问流程。

### 2. 流量特征提取与预处理

代理服务器在转发流量时，需要实时截取每条 TCP 连接的前 `N` 个数据包长度，构造固定长度包长序列。

要求如下：

- 默认序列长度为 `N = 100`。
- 不足 `N` 个包时使用 0 补齐。
- 超过 `N` 个包时截断。
- 将包长度归一化到 `[0, 1]` 区间。
- 归一化后的序列作为神经网络输入特征。

### 3. 离线模型训练

下载并使用公开加密流量数据集，例如 ISCX VPN-nonVPN 或 CIC-IDS2017 的子集，选取 3 到 4 类常见应用流量进行分类实验，例如：

- Video
- Chat
- FileTransfer
- Web

离线数据处理需要完成：

- 按五元组，即源 IP、目的 IP、源端口、目的端口、传输层协议，对 PCAP 进行流切分。
- 提取每条流的前 `N` 个包长特征。
- 对特征进行清洗、补齐、截断和归一化。
- 生成包含 `f0 ... f(N-1)` 与 `label` 字段的训练 CSV。

模型训练要求：

- 使用 PyTorch 构建 1D-CNN 分类模型。
- 输出训练过程中的 Loss、Accuracy、Macro-F1 等指标。
- 保存训练好的模型权重与标签映射。
- 生成训练曲线和混淆矩阵，便于分析分类效果。

### 4. 在线实时推理

将训练好的 1D-CNN 模型集成到 TCP 代理服务器中。当某条连接积累到足够的包长序列后，代理服务器应立即调用模型进行预测，并在控制台输出当前连接的应用类型，例如：

```text
Stream Type: Video
```

同时，预测结果需要写入本地日志，供监控看板读取。

### 5. 实时流量监控看板

使用 Streamlit 实现 Web 监控界面，动态展示近期代理流量的识别结果。

看板需要包含：

- 最近一段时间内的应用类型分布。
- 流量类别占比图。
- 最新预测记录列表。
- 可调节的统计时间窗口。

## 三、项目已实现内容

当前仓库已经实现了一套可运行的简化端到端系统，主要模块如下：

| 模块 | 文件 | 说明 |
| --- | --- | --- |
| 代理服务 | `src/proxy/tcp_proxy.py` | 支持 HTTP 与 HTTPS CONNECT 转发，使用多线程处理连接 |
| 包长特征 | `src/features/packet_sequence.py` | 维护前 `N` 个包长并归一化为模型输入 |
| PCAP 预处理 | `src/data/pcap_preprocess.py` | 按 TCP 五元组切分流并生成训练 CSV |
| 多类数据合并 | `scripts/build_training_csv.py` | 支持从多类 PCAP 文件或目录构建统一训练集 |
| CNN 模型 | `src/model/cnn1d.py` | 实现 1D-CNN 流量分类网络 |
| LSTM 对比模型 | `src/model/lstm.py` | 提供 LSTM 分类器用于扩展对比实验 |
| 离线训练 | `scripts/train.py` | 支持 CNN/LSTM 训练、类别权重、训练曲线和混淆矩阵输出 |
| 在线推理 | `src/model/inference.py` | 加载模型权重与标签映射，提供实时预测接口 |
| 预测日志 | `src/utils/monitor_store.py` | 将在线预测结果写入 JSONL 日志 |
| 实时看板 | `dashboard/app.py` | 使用 Streamlit 展示类别分布与近期预测记录 |

## 四、完成度评估

| 要求 | 完成情况 | 说明 |
| --- | --- | --- |
| TCP/HTTP 代理服务器 | 已完成 | 支持普通 HTTP 与 HTTPS CONNECT 隧道，使用线程实现并发双向转发 |
| 实时包长序列提取 | 已完成 | 在线记录前 `N=100` 个包长，并归一化到 `[0, 1]` |
| PCAP 流切分与训练集生成 | 已完成 | 支持按五元组切分 TCP 流，并生成 `f0...fN` + `label` 格式 CSV |
| 1D-CNN 离线训练 | 已完成 | 使用 PyTorch 训练 CNN，保存模型与标签，并输出评估指标 |
| 在线实时推理 | 已完成 | 代理积累到足够特征后调用模型推理，并打印预测结果 |
| Streamlit 实时看板 | 已完成 | 支持近期应用分布饼图与最新预测记录 |
| 训练曲线与混淆矩阵 | 已完成 | 训练脚本会生成 `training_curves.png` 与 `confusion_matrix.png` |
| LSTM 对比实验 | 部分完成 | 已实现 LSTM 模型和训练入口，但仍需补充正式实验结果对比 |
| 抗抖动或 Padding 测试 | 未完成 | 主代理尚未加入随机延迟或包大小扰动实验 |
| 自定义开放问题 | 未完成 | 尚未形成独立的开放性扩展问题与完整实验分析 |

总体来看，项目已经完成基础功能和主要扩展功能，能够覆盖实验要求中的核心部分。若用于最终展示，建议补充数据集来源说明、训练结果截图、LSTM 对比结果，以及抗抖动实验分析。

## 五、运行流程

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 构建训练数据

从多类 PCAP 文件构建训练 CSV：

```bash
python scripts/build_training_csv.py \
  --class-pcap video=data/extracted/video.pcap \
  --class-pcap chat=data/extracted/chat.pcap \
  --class-pcap file=data/extracted/file.pcap \
  --class-pcap web=data/extracted/web.pcap \
  --out data/flows.csv
```

也可以对单个 PCAP 生成单类 CSV：

```bash
python -m src.data.pcap_preprocess \
  --pcap data/video.pcap \
  --out data/video.csv \
  --label Video
```

### 3. 训练模型

训练 1D-CNN：

```bash
python scripts/train.py --csv data/flows.csv --model cnn1d --epochs 15
```

训练完成后会生成：

- `artifacts/cnn1d.pth`
- `artifacts/labels.json`
- `artifacts/training_curves.png`
- `artifacts/confusion_matrix.png`

### 4. 启动监控看板

```bash
python -m streamlit run dashboard/app.py
```

### 5. 启动代理服务

```bash
python -m src.proxy.tcp_proxy
```

随后在系统或浏览器中设置 HTTP/HTTPS 代理：

```text
127.0.0.1:8080
```

浏览网页或产生网络流量后，可在代理控制台查看实时预测结果，并在 Streamlit 看板中查看应用类型分布。

## 六、后续改进方向

1. 补充公开数据集下载、解压和标签整理的自动化脚本。
2. 对 CNN、LSTM、MLP 等模型进行系统性对比，记录准确率、Macro-F1 和推理延迟。
3. 在代理转发链路中加入随机延迟、包大小 Padding 或流量扰动，评估模型鲁棒性。
4. 优化看板交互，例如增加实时刷新、类别筛选和历史趋势图。
5. 完善实验报告，加入数据集说明、训练参数、结果截图和误分类分析。
