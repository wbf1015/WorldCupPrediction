import requests
import json
import pandas as pd

play_raw = requests.get("http://api.sports.sina.com.cn/?p=sports&s=sport_client&a=index&_sport_t_=livecast&_sport_a_=matchesbytype&type=108&fields=group%2Clivecast_id%2Cdate%2Cstatus%2CTeam1%2CTeam2%2Ctime%2CTeam1Id%2CTeam2Id%2CScore1%2CScore2%2CMatchCity%2CNewsUrl%2CVideoUrl%2CLiveUrl%2Cmatch_url%2CImgUrl%2COptaId%2CRound&limit=64&season=2017&dpc=1")

play_json = json.loads(play_raw.text)
print(type(play_json['result']['data']))
play_dumps=(json.dumps(play_json['result']['data'],ensure_ascii=False))
print(type(play_dumps))

play_excel=pd.read_json(play_dumps)
#print(play_excel)
res=pd.DataFrame(play_excel,columns=['date','Team1','Score1','Score2','Team2'])
res.to_csv('2018.csv', mode='w', encoding='utf_8_sig', index=False)
# print(play_json2)
#pd.read_json(play_dumps)

print(res)