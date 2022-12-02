from random import sample
import pandas as pd

df = pd.read_csv('NO16.csv',encoding='UTF-8')
country = df['country']

g1_index=sample(range(0,16),8)
group1=pd.Series(country[g1_index]).reset_index(drop = True)

g2_index=[i for i in range(0,16) if i not in g1_index]
group2=pd.Series(country[g2_index])

for i in range(0,8):
    print("组1：第",i+1,"队")
    print(group1.iloc[i])
    print("组2：第",i+1,"队")
    print(group2.iloc[i])
