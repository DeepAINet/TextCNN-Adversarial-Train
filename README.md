# TextCNN-Adversarial-Train
PGD, Free, FGSM used in TextCNN


# 中文数据集
从[THUCNews](http://thuctc.thunlp.org/)中抽取了20万条新闻标题,抽样了部分数据集。

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

# 数据集划分：
   数据集|          数据量|
   -----|          -----|
   训练集           54000
   验证集           3000
   测试集           3000


# 评估结果：
        precision recall    F1-score    acc
PGD     0.8635    0.8630    0.8628      0.8630
FGSM    0.8609    0.8577    0.8584      0.8577
FREE    0.8707    0.8687    0.8688      0.8687
PLAIN   0.8565    0.8560    0.8556      0.8560
PLAIN为base line
