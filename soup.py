from bs4 import BeautifulSoup
import requests
import pandas as pd


def get_html(year):
    # 获取URL
    url = 'http://%d.sina.com.cn/fixtures/timeline.html' % year
    response = requests.get(url)
    response.encoding = 'gb2312'
    # 页面编码gb2312
    return response.text


def get_data(html):
    soup = BeautifulSoup(html, 'lxml', )
    tables = soup.findAll('table', attrs={'class': 'tb_01'})
    # 使用soup获取所有table
    data = []
    for table in tables:
        trs = table.findAll('tr')
        del trs[0]
        last_date = ' '
        for tr in trs:
            tds = tr.findAll('td')
            date = tds[0].get_text()
            if date == ' ':
                date = last_date
            else:
                last_date = date
            #为并列项补全日期

            team1 = tds[2].find('span', attrs={'class': 'tar t_c'}).find('a').get_text()
            vs = tds[2].find('span', attrs={'class': 'vs'}).find('a').get_text()
            team1_score = vs.split('-')[0]
            team2_score = vs.split('-')[1]
            team2 = tds[2].find('span', attrs={'class': 'tal t_c'}).find('a').get_text()

            # 获取各分量值
            dic = {'date': date,'team1': team1,'team2': team2,
                    'team1_score': team1_score, 'team2_score': team2_score}
            # 创建字典
            pd_data = pd.DataFrame(dic, index=[0])
            data.append(pd_data)
    #         使用pandas进行简单数据处理
    return data

year = 2014
data = []
html = get_html(year)
data.extend(get_data(html))
pd_data2 = pd.concat(data)
pd_data2.to_csv('res.csv', mode='w', encoding='utf_8_sig', index=False)