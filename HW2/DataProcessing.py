# coding:utf-8
import csv
import os
import numpy as np
import pandas as pd


def makeDataProcessing(dfData):
    dfDataX = dfData.drop(["education_num", "sex"], axis=1)

    listObjectColumnName = [col for col in dfDataX.columns if dfDataX[col].dtypes == "object"]
    listNonObjectColumnName = [col for col in dfDataX.columns if dfDataX[col].dtype != "object"]

    dfNonObjectData = dfDataX[listNonObjectColumnName]
    dfNonObjectData.insert(2, "sex", (dfData["sex"] == "Male").astype(np.int))  # Male 1, Female 0

    dfObjectData = dfDataX[listObjectColumnName]
    # get_dummies 对数据进行 one_hot 编码
    dfObjectData = pd.get_dummies(dfObjectData)

    dfDataX = dfNonObjectData.join(dfObjectData)
    dfDataX = dfDataX.astype("int64")
    return dfDataX


if __name__ == '__main__':

    # read raw data
    dfDataTrain = pd.read_csv(os.path.join(os.path.dirname(__file__), "./01-Data/train.csv"))
    dfDataTest = pd.read_csv(os.path.join(os.path.dirname(__file__), "./01-Data/test.csv"))

    # show training size and testing size
    intTrainSize = len(dfDataTrain)
    intTestSize = len(dfDataTest)
    print(intTrainSize)
    print(intTestSize)

    # processing training label (Y)
    dfDataTrainY = dfDataTrain["income"]
    dfTrainY = pd.DataFrame((dfDataTrainY == ">50k").astype("int64"), columns=["income"])  # >50k 1,<=50k 0

    # processing training and testing data(X)
    dfDataTrain = dfDataTrain.drop(["income"], axis=1)
    dfAllData = pd.concat([dfDataTrain, dfDataTest], axis=0, ignore_index=True)
    dfAllData = makeDataProcessing(dfData=dfAllData)

    # separate all data to training and testing
    dfTrainX = dfAllData[0:intTrainSize]
    dfTestX = dfAllData[intTrainSize:(intTrainSize + intTestSize)]

    # save training data, testing data and training label
    dfTrainX.to_csv(os.path.join(os.path.dirname(__file__), "./01-Data/X_train_my.csv"), index=False)
    dfTestX.to_csv(os.path.join(os.path.dirname(__file__), "./01-Data/X_test_my.csv"), index=False)
    dfTrainY.to_csv(os.path.join(os.path.dirname(__file__), "./01-Data/Y_train_my.csv"), index=False)