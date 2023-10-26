#Coverage error
import numpy as np
from sklearn.metrics import coverage_error
y_true = np.array([[1, 0, 0], [0, 0, 1]])
y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
coverage_error(y_true, y_score)

#Label ranking average precision
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score
y_true = np.array([[1, 0, 0], [0, 0, 1]])
y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
label_ranking_average_precision_score(y_true, y_score)

#Ranking loss
import numpy as np
from sklearn.metrics import label_ranking_loss
y_true = np.array([[1, 0, 0], [0, 0, 1]])
y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
label_ranking_loss(y_true, y_score)
# With the following prediction, we have perfect and minimal loss
y_score = np.array([[1.0, 0.1, 0.2], [0.1, 0.2, 0.9]])
label_ranking_loss(y_true, y_score)
