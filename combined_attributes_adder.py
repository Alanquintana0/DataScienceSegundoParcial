import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

cholesterol_ix = heart_data_num.columns.get_loc('Cholesterol')
bmi_ix = heart_data_num.columns.get_loc('BMI')
diabetes_ix = heart_data_num.columns.get_loc('Diabetes')
tryglicerides_ix = heart_data_num.columns.get_loc('Triglycerides')
print(cholesterol_ix, bmi_ix, diabetes_ix, tryglicerides_ix)
cholesterol_ix, bmi_ix, diabetes_ix, tryglicerides_ix = 1, 14, 3, 15

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
  def __init__(self, add_cholesterol_and_tryg = True):
    self.add_cholesterol_and_tryg = add_cholesterol_and_tryg
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    diabetes_n_triglycerides = (X[:, diabetes_ix] + 1e-10) / X[:, tryglicerides_ix]
    cholesterol_bmi_ratio = (X[:, cholesterol_ix] + 1e-10) / (X[:, bmi_ix] +1e-10)
    if self.add_cholesterol_and_tryg:
      cholesterol_and_tryg = (X[:, cholesterol_ix] + 1e-10) / (X[:, tryglicerides_ix]+1e-10)
      return np.c_[X, diabetes_n_triglycerides, cholesterol_bmi_ratio, cholesterol_and_tryg]
    else:
      return np.c_[X, diabetes_n_triglycerides, cholesterol_bmi_ratio]