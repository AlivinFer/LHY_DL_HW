## Homework 1

#### Purpose： Predict PM2.5

Data 来源于丰原站的观测记录，分成 train set 和 test set

- train.csv：每个月前 20 天的完整资料，每天测量 18 项，每隔 1 小时测一次，总共有 4320 条记录
- test.csv ：剩下的资料取样出连续的九个小时作为feature

1. 数据处理

   以 18 行作为一个整体输入，label 为 PM2.5 连续九个小时后的下一个值

   并将 RAINFALL 那行的 NR 改为 0

2. 梯度下降

3. adagrad  优化学习率

