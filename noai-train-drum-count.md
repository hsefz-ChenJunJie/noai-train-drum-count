# NOAI 模拟赛：鼓点计数器 (Drum Beat Counter)

## 一、 题目描述

本题目是一个基于 PyTorch 框架的音频与视觉跨领域深度学习项目。选手需要构建一个神经网络，从复杂的音乐混音片段中，精准检测并计算出“底鼓（Kick Drum, KD）”和“军鼓（Snare Drum, SD）”的数量。

在现代音乐制作与音乐信息检索（MIR）中，底鼓与军鼓构成了节奏的骨架。底鼓拥有显著的低频能量与极快的瞬态（Transient），而军鼓则横跨中高频，并常带有白噪声特征。但在复杂的混音（Mix）中，它们极易与其他乐器（如贝斯、踩镲）发生频率混叠。

## 二、 数据集

数据集分为 training_set（训练集）、validation_set（验证集 / A榜）和 testing_set（测试集 / B榜）。

audio_mix (主任务数据)：包含所有乐器的完整混音片段（*_MIX.wav）。测试集仅会提供此类数据。

annotation_xml (真实标签)：记录了对应混音片段中，各类鼓点精确敲击时间的 XML 文件。

audio_isolated (独立音轨)：仅在训练集中提供。包含纯净的底鼓（_KD.wav）、军鼓（_SD.wav）和踩镲（_HH.wav）分离音轨。

audio_training_hits (单音色样本)：仅在训练集中提供。极其短促的单次敲击干声。

**注意数据集较小， audio_mix 中只有 38 个样本。**

三、 任务

本项目的任务是建立一个能够处理单通道梅尔频谱图，并同时输出片段内底鼓数量和军鼓数量的神经网络。

四、 提交

选手需要提交一个名为 submission.zip 的压缩包文件。该压缩包内必须严格包含以下两个文件：

submissionA.csv：包含模型在验证集上的预测结果。

submissionB.csv：包含模型在测试集上的预测结果。

CSV 文件格式要求：

- 每一行代表一个音频片段的预测结果，共两列。第一列为底鼓（KD）数量，第二列为军鼓（SD）数量。

- 请注意：不允许包含任何表头，且预测结果必须为 整数

正确的文件内容示例：

```
2,1
0,0
4,2
1,3
```

五、 评分

系统会将选手提交的 CSV 文件与隐藏的 ground_truth_labels.csv 进行对比。

为了精准衡量模型对“数量”概念的理解，本题采用 平均绝对误差 (Mean Absolute Error, MAE) 作为核心评估指标，并将其映射为百分制分数。

对于数据集中的 $N$ 个样本，整体误差 $MAE_{total}$ 的计算公式如下：


$$MAE_{total} = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{|Pred_{KD,i} - True_{KD,i}| + |Pred_{SD,i} - True_{SD,i}|}{2} \right)$$

最终得分换算：


$$Score = \max(0,\ 1 - 0.1 \times MAE_{total})$$
