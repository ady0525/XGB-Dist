import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def data_proc_mrs(df_comb, df_num, df_cat, groupname):
  ## data filtering
  
  if groupname == 'group postop':
    num_names = df_num[ df_num['percs']>0.7 ]['names']  # perc of DC_NIHSS = 0.744
    num_names = list(num_names)

    cat_names = df_cat[ df_cat['percs']>0.9 ]['names']
    cat_names = list(cat_names)
  else:
    num_names = df_num[ (df_num[groupname]==1) & (df_num['percs']>0.7) ]['names']  # perc of DC_NIHSS = 0.744
    num_names = list(num_names)

    cat_names = df_cat[ (df_cat[groupname]==1) & (df_cat['percs']>0.9) ]['names']
    cat_names = list(cat_names)

  names_to_delete = ['INTRA_PROC_COMP_NEW_9',
                     'INTRA_PROC_COMP_NEW_1',
                     'INTRA_PROC_COMP_NEW_99',
                     'PASSES_PERFORMED',
                     'INTRA_PROC_COMP_NEW_6',
                     'THROMBOLYTIC',
                     'PRE_ADMIN',
                     'INTRA_PROC_COMP_NEW_2',
                     'PUNC_COMPLICATION',
                     'DEAD',
                     'PROC_SURVIVALDAYS']

  cat_names = [x for x in cat_names if x not in names_to_delete]
  num_names = [x for x in num_names if x not in names_to_delete]

  sel_names = num_names + cat_names + ['R_DAY_90_MRS']
  df_sel = df_comb[sel_names]
  df_sel = df_sel.loc[~np.isnan(df_sel['R_DAY_90_MRS'])]

  X0 = df_sel[num_names + cat_names]
  y0 = df_sel['R_DAY_90_MRS']

  ## X proc (Imputation)

  numeric_pipeline = Pipeline(
      steps=[("impute", SimpleImputer(strategy="mean")),
             # ("scale", StandardScaler())
            ]
  )

  categorical_pipeline = Pipeline(
      steps=[
          ("impute", SimpleImputer(strategy="most_frequent")),
          # ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
      ]
  )

  full_processor = ColumnTransformer(
      transformers=[
          ("numeric", numeric_pipeline, num_names),
          ("categorical", categorical_pipeline, cat_names),
      ]
  )

  X = full_processor.fit_transform(X0)

  X_proc = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

  ## y proc (Re-grouping)

  y = y0.astype(np.int64)

  y_grp = y.copy()
  y_grp = np.where((y_grp>=0) & (y_grp<=2), 0, y_grp)
  y_grp = np.where((y_grp>=3), 1, y_grp)
  y_proc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(
      y_grp.reshape(-1, 1)
  )

  return X_proc, y_proc, num_names, cat_names

def data_proc_mrs6(df_comb, df_num, df_cat, groupname):
  ## data filtering
  
  if groupname == 'group postop':
    num_names = df_num[ df_num['percs']>0.7 ]['names']  # perc of DC_NIHSS = 0.744
    num_names = list(num_names)

    cat_names = df_cat[ df_cat['percs']>0.9 ]['names']
    cat_names = list(cat_names)
  else:
    num_names = df_num[ (df_num[groupname]==1) & (df_num['percs']>0.7) ]['names']  # perc of DC_NIHSS = 0.744
    num_names = list(num_names)

    cat_names = df_cat[ (df_cat[groupname]==1) & (df_cat['percs']>0.9) ]['names']
    cat_names = list(cat_names)

  names_to_delete = ['INTRA_PROC_COMP_NEW_9',
                     'INTRA_PROC_COMP_NEW_1',
                     'INTRA_PROC_COMP_NEW_99',
                     'PASSES_PERFORMED',
                     'INTRA_PROC_COMP_NEW_6',
                     'THROMBOLYTIC',
                     'PRE_ADMIN',
                     'INTRA_PROC_COMP_NEW_2',
                     'PUNC_COMPLICATION',
                     'DEAD',
                     'PROC_SURVIVALDAYS']

  cat_names = [x for x in cat_names if x not in names_to_delete]
  num_names = [x for x in num_names if x not in names_to_delete]

  sel_names = num_names + cat_names + ['R_DAY_90_MRS']
  df_sel = df_comb[sel_names]
  df_sel = df_sel.loc[~np.isnan(df_sel['R_DAY_90_MRS'])]

  X0 = df_sel[num_names + cat_names]
  y0 = df_sel['R_DAY_90_MRS']

  ## X proc (Imputation)

  numeric_pipeline = Pipeline(
      steps=[("impute", SimpleImputer(strategy="mean")),
             # ("scale", StandardScaler())
            ]
  )

  categorical_pipeline = Pipeline(
      steps=[
          ("impute", SimpleImputer(strategy="most_frequent")),
          # ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
      ]
  )

  full_processor = ColumnTransformer(
      transformers=[
          ("numeric", numeric_pipeline, num_names),
          ("categorical", categorical_pipeline, cat_names),
      ]
  )

  X = full_processor.fit_transform(X0)

  X_proc = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

  ## y proc (Re-grouping)

  y = y0.astype(np.int64)

  y_grp = y.copy()
  # y_grp = np.where((y_grp>=0) & (y_grp<=2), 0, y_grp)
  # y_grp = np.where((y_grp>=3), 1, y_grp)
  y_proc = y_grp.to_numpy()
  y_proc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(
      y_grp.to_numpy()[:,None].reshape(-1, 1)
  )

  return X_proc, y_proc, num_names, cat_names

def data_proc_nihss(df_data, df_num, df_cat, groupname):
    num_names = df_num[ (df_num[groupname]==1) & (df_num['percs']>0.8) ]['names']
    num_names = list(num_names)
    len(num_names)
    
    cat_names = df_cat[ (df_cat[groupname]==1) & (df_cat['percs']>0.9) ]['names']
    cat_names = list(cat_names)
    len(cat_names)
    
    names_to_delete = ['INTRA_PROC_COMP_NEW_9', 
                       'INTRA_PROC_COMP_NEW_1', 
                       'INTRA_PROC_COMP_NEW_99', 
                       'PASSES_PERFORMED', 
                       'INTRA_PROC_COMP_NEW_6',
                       'THROMBOLYTIC',
                       'PRE_ADMIN',
                       'INTRA_PROC_COMP_NEW_2',
                       'PUNC_COMPLICATION',
                       'DEAD',
                       'PROC_SURVIVALDAYS']
                       
    cat_names = [x for x in cat_names if x not in names_to_delete]
    num_names = [x for x in num_names if x not in names_to_delete]

    len(cat_names)
    
    select_names = num_names + cat_names + ['DELTA_NIHSS']
    df_sel = df_data[select_names]
    df_sel = df_sel.dropna()
    
    X = df_sel.drop('DELTA_NIHSS', axis=1)
    y = df_sel['DELTA_NIHSS']
    
    X_data = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    X_data = X_data.to_numpy()
    
    y_data = y.to_numpy()
    
    return X_data, y_data, num_names, cat_names 

def data_proc_nihss2(df_data, df_num, df_cat, groupname):
    num_names = df_num[ (df_num[groupname]==1) & (df_num['percs']>0.7) ]['names']
    num_names = list(num_names)
    len(num_names)
    
    cat_names = df_cat[ (df_cat[groupname]==1) & (df_cat['percs']>0.7) ]['names']
    cat_names = list(cat_names)
    len(cat_names)
    
    names_to_delete = ['INTRA_PROC_COMP_NEW_9', 
                       'INTRA_PROC_COMP_NEW_1', 
                       'INTRA_PROC_COMP_NEW_99', 
                       'PASSES_PERFORMED', 
                       'INTRA_PROC_COMP_NEW_6',
                       'THROMBOLYTIC',
                       'PRE_ADMIN',
                       'INTRA_PROC_COMP_NEW_2',
                       'PUNC_COMPLICATION',
                       'DEAD',
                       'PROC_SURVIVALDAYS']
                       
    cat_names = [x for x in cat_names if x not in names_to_delete]
    num_names = [x for x in num_names if x not in names_to_delete]

    len(cat_names)
    
    select_names = num_names + cat_names + ['DC_NIHSS']
    df_sel = df_data[select_names]
    df_sel = df_sel.dropna()
    
    X = df_sel.drop('DC_NIHSS', axis=1)
    y = df_sel['DC_NIHSS']
    
    X_data = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    X_data = X_data.to_numpy()
    
    y_data = y.to_numpy()
    
    return X_data, y_data, num_names, cat_names 