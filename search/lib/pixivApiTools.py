import time
import json
sleepTime = 3

""" 指定の日付に投稿されたイラストの端を検索 """
def searchIllustData(api, word, sort, date) -> json:
    # date_desc: 新しい順, date_asc: 古い順
    json_result = api.search_illust(word=word, search_target='partial_match_for_tags', sort=sort, start_date=date, end_date=date, search_ai_type=1)
    time.sleep(sleepTime)

    return json_result

# 指定日の最新のイラストIDを取得
def getNewIllustId(api, word, date) -> int:
    results = searchIllustData(api, word, "date_desc", date)
    newId = results.illusts[0].id

    return newId

# 指定日の最古のイラストIDを取得
def getOldIllustId(api, word, date) -> int:
    results = searchIllustData(api, word, "date_asc", date)
    oldId = results.illusts[0].id

    return oldId

def getIllustData(api, illustId) -> json:
    illustData = api.illust_detail(illustId)
    time.sleep(sleepTime)
    return illustData