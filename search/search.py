import requests
import os
import json
import time
import sys

from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pytz import timezone
from pixivpy3 import *

import setting.config as config
import lib.pixivGetTools as pixivGetTools
import lib.pixivApiTools as pixivApi
import lib.jsonLoadAndWrite as jsonLoadAndWrite

UTC = timezone("UTC")
JST = timezone("Asia/Tokyo")
timeFormat = "%Y-%m-%d %H:%M:%S"
illustId = "1"
downloadDir = config.DOWNLOAD_DIR
illustJsonDir = "../data/illustData.json"
searchedJsonDir = "../data/searched.json"
maxCount = 100
sleepTime = 3
tagsNG = ["R-18", "R-18G", "漫画", "AI生成", "うごイラ"]

# def getContents(link) -> json:
#     r = requests.get(link)
#     time.sleep(1)

#     soup = BeautifulSoup(r.content, "html.parser")
#     try:
#         contents = soup.find_all("meta", id="meta-preload-data")[0].get("content")
#     except:
#         return None
#     print(contents)

#     contents = json.loads(contents)

#     return contents

def searchDownload(api, id, detailData):
    dict = {}
    illustData = api.illust_detail(id)
    time.sleep(1)

    if str(id) in detailData:
        print("id {} has already been downloaded".format(id))
        return False
    
    try:
        illustData["illust"]
    except:
        print("not found")
        return False

    if illustData["illust"]["user"]["account"] == "":
        print("id {} is not found".format(id))
        return False

    if pixivGetTools.getAiType(illustData, id) == 2:
        print("id {} is AI illust".format(id))
        return False
    
    if pixivGetTools.isManga(illustData, id):
        print("id {} is manga".format(id))
        return False

    if pixivGetTools.isIncludeTags(illustData, tagsNG, id):
        print("id {} include NG tags".format(id))
        return False

    url = illustData["illust"]["image_urls"]["large"]

    if illustData["illust"]["sanity_level"] > 2:
        print("id {} is sensitive illust".format(id))
        return False

    api.download(url, path = downloadDir, fname = "{}.jpg".format(id))
    time.sleep(sleepTime)

    bookmark = pixivGetTools.getBookmarkCount(illustData, id)
    view = pixivGetTools.getViewCount(illustData, id)

    dict["id"] = id
    dict["bookmark"] = bookmark
    dict["view"] = view
    detailData[str(id)] = dict

    print("id {} is downloaded".format(id))
    return True

""" ログイン用 """
def apiLogin() -> AppPixivAPI:
    api = AppPixivAPI()
    api.auth(refresh_token=config.REFRESH_TOKEN)
    return api

def main():
    api = apiLogin()
    args = sys.argv

    if len(args) > 1:
        maxCount = int(args[1])

    # 1日前の日付を取得
    yesterday = datetime.now() - timedelta(days=1)
    date_str = yesterday.strftime('%Y-%m-%d')

    detailData = jsonLoadAndWrite.openJson(illustJsonDir)
    searched = jsonLoadAndWrite.openJson(searchedJsonDir)

    dataMaxId = searched["id"] + 1
    
    minId = max(pixivApi.getOldIllustId(api, "", date_str), dataMaxId)
    
    count = 0
    nowId = minId

    # 指定した枚数分だけ保存
    while (count < maxCount):
        if searchDownload(api, nowId, detailData):
            count += 1
        nowId += 1

    # まとめてJSONを更新
    jsonLoadAndWrite.saveJson(illustJsonDir, detailData)
    searched = {"id": nowId - 1}
    jsonLoadAndWrite.saveJson(searchedJsonDir, searched)

    print("illusts: {} files".format(len(detailData)))

if __name__ == "__main__":
    main()