# SJTU-CS3611-Final

基于深度学习的加密流量实时识别与监控系统（简化可运行骨架）。

## 当前已实现模块

1. TCP/HTTP 代理服务（支持 CONNECT 隧道转发）。
2. 在线包长度序列提取与归一化（前 N 个长度特征）。
3. PyTorch 1D-CNN 离线训练脚本。
4. 在线推理与预测日志输出。
5. Streamlit 实时流量分布看板。

## 项目结构说明

详细结构见 `PROJECT_STRUCTURE.md`。

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 下载公开数据集（满足实验要求）

- 官方入口：
	- ISCX VPN-nonVPN: https://www.unb.ca/cic/datasets/vpn.html
	- CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- 若拿到可直链下载 URL，可直接执行：

```bash
python scripts/download_dataset.py --url "DIRECT_FILE_URL_1" --url "DIRECT_FILE_URL_2"
```

3. 从 3~4 类 PCAP 构建训练集（Video/Chat/FileTransfer 等）

```bash
python scripts/build_training_csv.py --class-pcap video=data/extracted/video.pcap --class-pcap chat=data/extracted/chat.pcap --class-pcap file=data/extracted/file.pcap --out data/flows.csv
```

4. 或手动准备训练数据（CSV）

- CSV 需包含 `f0..f99` 和 `label` 列。
- 或使用 PCAP 转换脚本（单标签示例）：

```bash
python -m src.data.pcap_preprocess --pcap data/video.pcap --out data/video.csv --label Video
```

5. 训练模型

```bash
python scripts/train.py --csv data/flows.csv --epochs 15
```

6. 启动监控看板

```bash
streamlit run dashboard/app.py
```

7. 启动代理服务

```bash
python -m src.proxy.tcp_proxy
```

8. 在系统或浏览器中设置代理为 `127.0.0.1:8080`，产生网络流量后查看控制台推理结果与看板。

## 后续建议

1. 增加多类数据聚合脚本，将多份 CSV 自动合并为训练集。
2. 补充 LSTM/MLP 对比实验与推理延迟统计。
3. 在代理中加入随机延迟与 padding，完成抗抖动测试。
