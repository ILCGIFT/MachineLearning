from sklearn import set_config
set_config(display='text')  
# displays text representation in a jupyter context
column_trans  

from sklearn.utils import estimator_html_repr
with open('my_estimator.html', 'w') as f:  
    f.write(estimator_html_repr(clf))
