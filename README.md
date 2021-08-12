# Unique-Risk
通过机器学习和特征工程对非上市公司特有风险进行研究<br><br>
**说明**
1. 《unique_risk-副本》为模型的基础数据集，《unique_risk》为经过多次测试后手动删除了部分异常值的数据集。
2.在《特有风险_raw》中使用的是副本数据，在《特有风险_short》中使用的是异常值剔除后的数据。由于数据集的问题，以及在《特有风险_raw》中没有做严格的测训练、测试集的划分，因此在结果上有一定的差异。最终结果以《特有风险_short》为准。
3. 《特有风险_raw》为草稿性质的文件，包含了对这个问题以及机器学习一些问题的研究和一些模型的说明。先阅读《特有风险_short》对《特有风险_short》中存疑的部分可以去raw中寻找更详细的解释。

**一些总结**
1. 调参时需要选择一些跨度比较大的区间，避免陷入局部最优或者由于AdaBoost弱分类器和estimator的联动影响。（比如神经网络需要的比较少，决策树需要的比较多）
2. K-Fold会生成多个模型，最终生成用于后续预测的模型由目前的所有训练样本生成。该模型的好坏由测试集来评估。k-fold依旧不能替代测试集划分的作用。
3.研究为什么k-fold上的表现和测试集表现不一。在交叉验证比较好的模型在测试集上表现不好。下文只是一个猜测和部分解释。
3.1 k-fold可以作为模型特征和超参数组合选择的依据，不能作为模型间比较的依据。建议还是通过在大量数据集的基础上通过划分训练集、测试集、验证集对模型进行评估。10%-30%的测试集是必要的。
4. 非线性模型，特别是Boosting类模型更容易受到异常值的影响。
5. Iforest，One-SVM等用于分类问题的异常值检验算法在回归问题上目前来看并没有比较好的效果。后续对回归问题的异常值检测，以及iforest等算法的不适用性进行检研究。
5.1 进一步研究iforest和原始数据集表现上的差异。 iforest能否用在回归问题。
5.2.不同的模型，leave-one-out差值最大的点不同。leave-one-out不适合做异常值检验的方法。
6.对于RFECV算法，一般来说，先原始数据集进行调参，获得一个基础比较好的解比先进行特征选择再调参能获得更好的结果。
7.AdaBoost类算法弱分类器对模型的表现具有比较大的影响。一般来说，决策树和神经网络是比较好的弱分类器。
8.随机梯度下降，测试集、样本集取样等问题可以通过设置random_state来处理，保证研究结果的可复现性。
