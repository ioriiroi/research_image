import requests
import os
import json
import time

from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pytz import timezone
from pixivpy3 import *

import config

UTC = timezone("UTC")
JST = timezone("Asia/Tokyo")
timeFormat = "%Y-%m-%d %H:%M:%S"
illustId = "1"
downloadDir = "./data/image"
illustJsonDir = "./data/illustData.json"
searchedJsonDir = "./data/searched.json"
maxCount = 100
sleepTime = 3
tagsNG = ["R-18", "R-18G", "漫画", "AI生成"]

def getContents(link) -> json:
    r = requests.get(link)
    time.sleep(1)

    soup = BeautifulSoup(r.content, "html.parser")
    contents = soup.find_all("meta", id="meta-preload-data")[0].get("content")
    contents = json.loads(contents)

    return contents

"いいね: likeCount, ブックマーク: bookmarkCount, 閲覧: viewCount, 投稿日時: createDate (グリニッジ標準時, +9hで日本時)"
class PixivGetTools:
    def getLikeCount(illustData, illustId) -> int:
        likeCount = illustData["illust"][illustId]["likeCount"]

        return int(likeCount)

    def getBookmarkCount(illustData, id) -> int:
        bookmarkCount = illustData["illust"][str(id)]["bookmarkCount"]

        return bookmarkCount

    def getViewCount(illustData, id) -> int:
        viewCount = illustData["illust"][str(id)]["viewCount"]

        return viewCount

    def getAiType(illustData, id) -> int:
        # not AI: 1, AI: 2
        aiType = illustData["illust"][str(id)]["aiType"]

        return aiType
    
    def isIncludeTags(illustData, targetTags, id) -> bool:
        tags = illustData["illust"][str(id)]["tags"]["tags"]
        for tag in tags:
            if tag["tag"] in targetTags:
                return True
        return False

    def isManga(illustData, id) -> bool:
        illustType = illustData["illust"][str(id)]["illustType"]
        if illustType == 1:
            return True
        return False

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

def searchDownload(api, id, detailData):
    dict = {}
    illustData = getContents("https://www.pixiv.net/artworks/{}".format(id))

    if str(id) in detailData:
        print("id {} has already been downloaded".format(id))
        return

    try:
        illustData["illust"]
    except:
        print("id {} is not found".format(id))
        return

    if PixivGetTools.getAiType(illustData, id) == 2:
        print("id {} is AI illust".format(id))
        return
    
    if PixivGetTools.isManga(illustData, id):
        print("id {} is manga".format(id))
        return

    if PixivGetTools.isIncludeTags(illustData, tagsNG, id):
        print("id {} include NG tags".format(id))
        return

    url = illustData["illust"][str(id)]["urls"]["original"]

    if url == None:
        print("id {} is sensitive illust".format(id))
        return

    api.download(url, path = downloadDir, fname = "{}.jpg".format(id))
    time.sleep(sleepTime)

    bookmark = PixivGetTools.getBookmarkCount(illustData, id)
    view = PixivGetTools.getViewCount(illustData, id)

    dict["id"] = id
    dict["bookmark"] = bookmark
    dict["view"] = view
    detailData[str(id)] = dict

    print("id {} is downloaded".format(id))

""" jsonファイルからidを読み込み、ダウンロード"""
def downloadFromJson(api, detailData):
    for id in detailData:
        searchDownload(api, id, detailData)

def openJson(jsonDir) -> json:
    with open(jsonDir, "r") as f:
        data = json.load(f)

    return data

def saveJson(jsonDir, data):
    with open(jsonDir, "w") as f:
        json.dump(data, f)

""" ログイン用 """
def apiLogin() -> AppPixivAPI:
    api = AppPixivAPI()
    api.auth(refresh_token=config.REFRESH_TOKEN)
    return api

def main():
    api = apiLogin()

    # 1日前の日付を取得
    yesterday = datetime.now() - timedelta(days=1)
    date_str = yesterday.strftime('%Y-%m-%d')

    detailData = openJson(illustJsonDir)
    searched = openJson(searchedJsonDir)

    dataMaxId = searched["id"] + 1
    
    minId = max(getOldIllustId(api, "", date_str), dataMaxId)

    # for id in range(minId, minId + maxCount):
    #     searchDownload(api, id, detailData)

    # saveJson(illustJsonDir, detailData)
    # searched = {"id": minId + maxCount - 1}
    # saveJson(searchedJsonDir, searched)

    print("illusts: {} files".format(len(detailData)))

if __name__ == "__main__":
    main()