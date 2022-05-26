# 语谱图

## 目录

- [目录](#目录)
- [1 程序说明](#1-程序说明)
  - [1.1 安装](#11-安装)
  - [1.2 使用](#12-使用)
  - [1.3 测试](#13-测试)
- [2 程序原理](#2-程序原理)
- [贡献者](#贡献者)
- [许可协议](#许可协议)

## 1 程序说明

### 1.1 安装

在使用前，你需要先安装程序所需的依赖：

- [Anaconda](https://www.anaconda.com/products/individual) 4.12 及以上（含 Python 3.9）

然后执行以下命令配置 Python 虚拟环境：

```bash
conda env update --name dsp --file environment.yml
conda activate dsp
```

### 1.2 使用

将音频文件放置于 `./data/dev_set` 目录下，执行以下命令启动程序：

```bash
python3 main.py
```

生成的 MFCC 系数以及过程中产生的其他图像将保存在 `./assets/mfcc` 目录下。

### 1.3 测试

本实验中，我们使用了预录制的音频文件 `shop.dat`，其内容是单词 shop 的一段语音，按 8000 Hz 采样。

## 2 程序原理

## 贡献者

- [**Hakula Chen**](https://github.com/hakula139)<[i@hakula.xyz](mailto:i@hakula.xyz)> - 复旦大学

## 许可协议

本项目遵循 MIT 许可协议，详情参见 [LICENSE](../LICENSE) 文件。
