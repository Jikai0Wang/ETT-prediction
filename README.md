# 现代电网变压器负载预测
本项目实现了用[ETTh1](https://github.com/zhouhaoyi/ETDataset)数据集训练LSTM和Transformer，并提出了在Transformer的Backbone上基于Fused Attention的改进方法。

## 数据预处理
raw_data文件夹下包含对原始数据集按照6：2：2切分成训练集、验证集和测试集的数据。
<pre>
python data_preprocess.py
</pre>

## 训练LSTM
<pre>
python train_lstm.py
</pre>

## 训练Transformer
<pre>
python train_transformer.py
</pre>

## Transformer+Fused Attention
<pre>
python train_fused_attn.py
</pre>
