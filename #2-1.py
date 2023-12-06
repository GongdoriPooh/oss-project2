import numpy as np
import pandas as pd

data_df = pd.read_csv('/Users/jaewook/OpenSource/oss-prj2/2019_kbo_for_kaggle_v2.csv')
def rank_by_year_and_keyword(year, keyword):
    top10_of_year = data_df[data_df['year'] == year].sort_values(by=keyword, ascending=False).head(10)
    top10_of_year = top10_of_year.loc[:, ['batter_name', keyword]]
    print("<Rank of " + keyword + ">")
    print(top10_of_year)
    print()

def highest_war_by_position_in2018(cp):
    print(f"{cp}:\n{data_df[(data_df['year'] == 2018) & (data_df['cp'] == cp)].sort_values(by='war', ascending=False)[['batter_name', 'war']].head(1)}\n")

# 2-1 1)
keywordList_record = ['H', 'avg', 'HR', 'OBP']
for year in range(2015, 2019):
    print("[Record of %d]" % year)
    for key in keywordList_record:
        rank_by_year_and_keyword(year, key)

# 2-1 2)
cpList = ["포수", "1루수", "2루수", "3루수", "유격수", "좌익수", "중견수", "우익수"]
for cp in cpList:
    highest_war_by_position_in2018(cp)

# 2-1 3)
slicedColumns = data_df.loc[:, ["R", "HR", "RBI", "SB", "war", "avg"]]
corr_result = slicedColumns.corrwith(data_df["salary"])
print("\n<corr with salary>")
print(corr_result)
print(f"따라서 가장 연관이 깊은 항목은 {corr_result.max()} 로 가장 1에 가까운 값을 가지는 {corr_result.idxmax()}이다.")