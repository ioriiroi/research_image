import json

""" jsonファイルからidを読み込み、ダウンロード"""
def openJson(jsonDir) -> json:
    with open(jsonDir, "r") as f:
        data = json.load(f)

    return data

def saveJson(jsonDir, data):
    with open(jsonDir, "w") as f:
        json.dump(data, f)