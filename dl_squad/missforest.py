import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class MyImputer(TransformerMixin):
    """
    Impute missing values
    Categorial columns are imputed with the most frequent value
    in column.
    Numeric columns are imputed with mean of column.
    """

    def fit(self, X):
        values = [X[c].value_counts().index[0]
                  if X[c].dtype == 'object' else X[c].mean() for c in X]
        self.fill = pd.Series(values, index=X.columns)
        return self

    def transform(self, X):
        return X.fillna(self.fill)


def split_columns_into_two_types(X):
    mask = np.array([True if X[c].dtype == 'object' else False for c in X])
    columns_cate = X.columns[mask]
    columns_num = X.columns[~mask]
    return columns_cate, columns_num, mask


def split_df_into_two_types(X):
    columns_cate, columns_num = split_columns_into_two_types(X)
    return X[columns_cate], X[columns_num]


class MissForestImputer(TransformerMixin):
    """
    Miss Forest Algorithm Implementation

    Impute missing values in the mixed-type dataframe (both numerical and categorical).

    It directly predicts the missing values using a random forest classification(for categorical variable)
    and regression(for numerical variable), trained on the observed values.
    """

    def __init__(self, max_iter=10, tol=1e-4):
        self.max_iter = max_iter
        self.tol = tol
        self.clf_params = {
            'n_estimators': 10,
            'n_jobs': -1,
            'random_state': 0,
            'max_features': 0.33
        }

    def fit_transform(self, X, cat_columns=None):
        # Boolean dataframe mask for missing values
        X_nan = X.isnull()

        # Number of missing values for each column
        nan_cnt = X_nan.sum(axis=0)

        # Number of columns that contain any missing value
        num_nan_vars = np.count_nonzero(nan_cnt)

        # nan 갯수가 적은 column부터 처리해야 하므로
        most_by_nan = list(nan_cnt.argsort())
        most_by_nan = np.array(most_by_nan[-num_nan_vars:])

        # Split mixed-type dataframe into two types such as categorical and numeric
        columns_cate, columns_num, mask_cate = split_columns_into_two_types(X)
        mask_cate_most_by_nan = np.array([mask_cate[m] for m in most_by_nan])
        most_by_nan_cate = (most_by_nan[mask_cate_most_by_nan])
        most_by_nan_num = (most_by_nan[~mask_cate_most_by_nan])
        total_nans_cate = X.iloc[:, mask_cate].isnull().values.sum()

        # Initial imputer (built-in Imputer는 범주형 변수에 대응하지 않으므로 신규 생성)
        X_init = MyImputer().fit_transform(X)
        X_init.iloc[:, mask_cate] = X_init.iloc[:, mask_cate].apply(lambda x: x.astype('category'))
        X_init.iloc[:, mask_cate] = X_init.iloc[:, mask_cate].apply(lambda x: x.cat.codes)
        # X_init[cat_columns] = X_init[cat_columns].apply(lambda x: x.astype('category'))
        # X_init[cat_columns] = X_init[cat_columns].apply(lambda x: x.cat.codes)

        # Initial_guess
        X_old = X_init
        NUMERIC_TYPES = ['int16', 'int32', 'int32', 'float16', 'float32', 'float64']

        for iter in range(self.max_iter):
            if iter > 0:
                X_old = X_new
            X_new = X_old.copy()

            # 결측값 빈도가 낮은 변수부터 수행
            for i in most_by_nan:

                # 결측값 예측 대상 변수 제외
                X_s = X_old.drop(X_old.columns[i], axis=1)

                # nan 여부 (true/false)
                i_mis = X_nan.iloc[:, i]
                i_misidx = list(np.where(i_mis == True)[0])
                i_obsidx = list(np.where(i_mis == False)[0])

                # 결측값을 제외한 관측값들만 추출
                X_obs = X_s.iloc[i_obsidx, :]
                y_obs = X_old.iloc[i_obsidx, i]
                dtype = X.dtypes[i]

                # Fit Random Forest s.t. y_obs ~ X_obs
                if (dtype in NUMERIC_TYPES):  # numeric variable
                    clf = RandomForestRegressor(**self.clf_params)
                else:  # categorical variable
                    clf = RandomForestClassifier(**self.clf_params)

                    # Training
                clf.fit(X_obs, y_obs)

                # Predict y_mis using x_mis
                X_mis = X_s.iloc[i_misidx, :]
                y_mis = clf.predict(X_mis)
                X_new.iloc[i_misidx, i] = y_mis

            # Compute the difference between X_old and X_new
            diff_num = (X_new.iloc[:, most_by_nan_num] - X_old.iloc[:, most_by_nan_num]) ** 2
            diff_cate = ~(X_new.iloc[:, most_by_nan_cate] == X_old.iloc[:, most_by_nan_cate])

            sum_num = diff_num.values.sum()
            sum_cate = diff_cate.values.sum()
            denom = X_new.iloc[:, most_by_nan_num].values.sum()
            delta_num = sum_num / denom
            delta_cate = sum_cate / total_nans_cate
            delta = delta_num + delta_cate

            # Stopping criterion
            if delta < self.tol or iter == self.max_iter - 1:
                X_imp = X_new
                # print('delta_final:{0}, {1}, {2}'.format(delta_num, delta_cate, delta))
                break

            # print('delta:{0}, {1}, {2}'.format(delta_num, delta_cate, delta))

        return X_imp