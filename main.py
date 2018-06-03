import DT
import numpy as np

X = np.loadtxt('X.csv', dtype=np.int32, delimiter=',')
y = np.loadtxt('y.csv', dtype=np.int32, delimiter=',')

tree = DT.decision_tree()
tree.fit(X, y)
tree.predict(X)

pause = 0
