import pandas as pd
df1 = pd.read_csv('/Users/jiashichao/Desktop/Edinburgh/valerio_project/data/Transfer_Data.csv')
df2 = pd.read_csv('/Users/jiashichao/Desktop/Edinburgh/valerio_project/data/ekko_customers_info.csv')

# 合并两个数据集，按照ID列进行合并
df = pd.merge(df1, df2, on='Sender_customer_Id')

selected_columns = df[["Sender_customer_Id", "Bank_of_Receiver", "address.townOrCity"]]

# 保存合并后的数据集为新的CSV文件
selected_columns.to_csv('/Users/jiashichao/Desktop/Edinburgh/valerio_project/data/merged.csv', index=False)