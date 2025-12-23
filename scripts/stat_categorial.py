import pandas as pd


data = pd.read_parquet('/share_data/data1/fanshengda/DEvo/data/solver_1221/solver_questioner_350_train.parquet')


print(data.head())