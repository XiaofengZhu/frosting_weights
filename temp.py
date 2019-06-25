import pandas as pd
df = pd.DataFrame([{'tags':'Daniel Tucker', 'content':'is cute'}, \
	{'tags':'Danbeast','content':'Does Danbeast like Xenoid?'}])
new_df_list = []
for index, row in df.iterrows():
	for word in str(row['tags']).split():
		new_row = {}
		new_row['tags'] = word
		new_row['content'] = row['content']
		new_df_list.append(new_row)
new_df = pd.DataFrame(new_df_list)
print(new_df)