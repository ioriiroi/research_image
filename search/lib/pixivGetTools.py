# def getLikeCount(illustData, illustId) -> int:
#     likeCount = illustData["illust"][illustId]["likeCount"]
#     return int(likeCount)

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