import requests
import os
import json
import time
import sys
import random

from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pytz import timezone
from pixivpy3 import *


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import setting.config as config
import lib.pixivGetTools as pixivGetTools
import lib.pixivApiTools as pixivApi
from src.JsonLoadAndWrite import openJson, saveJson

UTC = timezone("UTC")
JST = timezone("Asia/Tokyo")
FILE = "遊戯王"
WORD = f"{FILE}5000users入り"
timeFormat = "%Y-%m-%d %H:%M:%S"
downloadDir = config.ILLUST_GOOD_DIR
illustJsonDir = config.DATA_JSON_DIR
maxCount = 10
sleepTime = 3
tagsNG = ["R-18", "R-18G", "漫画", "AI生成", "うごイラ"]

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

    #api.download(url, path = illustMikuDir, fname = f"{id}.jpg")
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
""" --------- """
def main_archive(api):
    # 1日前の日付を取得
    yesterday = datetime.now() - timedelta(days=365)
    date_str = yesterday.strftime('%Y-%m-%d')

    # detailData = openJson(illustJsonDir)
    # searched = openJson(searchedJsonDir)
    detailData = openJson(mikuJsonDir)

    # dataMaxId = searched["id"] + 1
    dataMaxId = 1
    
    minId = max(pixivApi.getOldIllustId(api, "初音ミク", date_str), dataMaxId)
    
    count = 0
    nowId = minId

    # 指定した枚数分だけ保存
    while (count < maxCount):
        if searchDownload(api, nowId, detailData):
            count += 1
        nowId += 1

    # まとめてJSONを更新
    saveJson(mikuJsonDir, detailData)
    searched = {"id": nowId - 1}
    # saveJson(searchedJsonDir, searched)

def download(api, start_date, end_date, word, sort, DLfile, data_json, json_dir, limit):
    count = 0
    next_qs = None
    while (count < limit):
        if next_qs:
            search_results = api.search_illust(**next_qs)
        else:
            search_results = api.search_illust(word=word, search_target='partial_match_for_tags', sort=sort, start_date=start_date, end_date=end_date, search_ai_type=1)
        time.sleep(1)
        for illust in search_results.illusts:
            # if random.randint(1, 30) != 10:
            #     continue
            id = illust.id
            bookmark = illust.total_bookmarks
            view = illust.total_view
            illust_url = illust.image_urls.large
            illust_type = illust.type

            if bookmark < 5000:
                continue

            if str(id) in data_json or illust_type != "illust":
                continue

            api.download(illust_url, path = DLfile, fname = f"{id}.jpg")
            time.sleep(1)

            data_json[str(id)] = {"id": id, "bookmark": bookmark, "view": view}
            print(f"downloaded! id: {id}")
            saveJson(json_dir, data_json)

            count += 1
            if count >= limit:
                break
        next_qs = api.parse_qs(search_results.next_url)
        if next_qs == None:
            break


def main():
    api = apiLogin()
    DLfile = downloadDir
    data_json = openJson(illustJsonDir)
    word = WORD
    sort = "date_desc"
    start_date = "2024-6-15"
    end_date = "2025-12-15"
    json_dir = illustJsonDir
    download(api, start_date, end_date, word, sort, DLfile, data_json, json_dir, maxCount)

    print("illusts: {} files".format(len(data_json)))

if __name__ == "__main__":
    main()