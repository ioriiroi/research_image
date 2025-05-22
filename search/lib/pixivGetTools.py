# def getLikeCount(illustData, illustId) -> int:
#     likeCount = illustData["illust"][illustId]["likeCount"]
#     return int(likeCount)

def getBookmarkCount(illustData, id) -> int:
    bookmarkCount = illustData["illust"]["total_bookmarks"]

    return bookmarkCount

def getViewCount(illustData, id) -> int:
    viewCount = illustData["illust"]["total_view"]

    return viewCount

def getAiType(illustData, id) -> int:
    # not AI: 1, AI: 2
    aiType = illustData["illust"]["illust_ai_type"]

    return aiType

def isIncludeTags(illustData, targetTags, id) -> bool:
    tags = illustData["illust"]["tags"]
    for tag in tags:
        if tag["name"] in targetTags:
            return True
    return False

def isManga(illustData, id) -> bool:
    illustType = illustData["illust"]["type"]
    if illustType == "manga":
        return True
    return False