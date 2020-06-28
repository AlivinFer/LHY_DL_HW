import numpy as np

matrixA = []
for i in open('./01-Data/matrixA.txt'):
    row = [int(x) for x in i.split(',')]
    matrixA.append(row)

matrixB = []
for j in open('./01-Data/matrixB.txt'):
    row = [int(x) for x in j.split(',')]
    matrixB.append(row)

matrixA = np.array(matrixA)
matrixB = np.array(matrixB)

ans = matrixA.dot(matrixB)
ans.sort(axis=1)
print(ans)
# 参数：保存路径， 数组名，保存的文件格式（这里是十进制），分隔符（这里是 换行）
np.savetxt("./02-Output/Q1_ans.txt", ans, fmt='%d', delimiter='\n')
