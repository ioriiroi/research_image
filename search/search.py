import requests
import os
import json
import time

from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pytz import timezone
from pixivpy3 import *

import tools.config as config
import tools.pixivGetTools as pixivGetTools
import tools.pixivApiTools as pixivApi
import tools.jsonLoadAndWrite as jsonLoadAndWrite

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

    detailData = jsonLoadAndWrite.openJson(illustJsonDir)
    searched = jsonLoadAndWrite.openJson(searchedJsonDir)

    dataMaxId = searched["id"] + 1
    
    minId = max(pixivApi.getOldIllustId(api, "", date_str), dataMaxId)

    # for id in range(minId, minId + maxCount):
    #     searchDownload(api, id, detailData)

    # saveJson(illustJsonDir, detailData)
    # searched = {"id": minId + maxCount - 1}
    # saveJson(searchedJsonDir, searched)

    print("illusts: {} files".format(len(detailData)))

if __name__ == "__main__":
    main()