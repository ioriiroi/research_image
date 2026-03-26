import sys
import os
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setting import config

def split_array(ar, n_group):
    for i_chunk in range(n_group):
        yield ar[i_chunk * len(ar) // n_group:(i_chunk + 1) * len(ar) // n_group]

def plot_hist(x):
    x = np.array(x)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.hist(x, bins=30, range=[0, 1000])
    ax.set_title('Bookmark Histgram')
    ax.set_xlabel('bookmark')
    ax.set_ylabel('num')
    plt.show()

class SetLabel:
    def get_bookmark(image_paths, data):
        paths_bookmarks = {}
        for name in image_paths:
            filename = os.path.basename(name)
            id = os.path.splitext(filename)[0]
            if id in data:
                paths_bookmarks[id] = {"path": str(name), "bookmark": data[id]["bookmark"], "view": data[id]["view"]}
        return paths_bookmarks
                
    def split_label(bookmark):
        if bookmark < 10:
            return 0
        elif 10 <= bookmark < 40:
            return 1
        elif 40 <= bookmark < 100:
            return 2
        else:
            return 3

    def set_label(data):
        all_image_labels = []
        class_num = 4
        tmp = [0] * class_num
        label_limit = float("inf")
        for num in data:
            bookmark = data[num]['bookmark']
            label = SetLabel.split_label(bookmark)
            if tmp[label] < label_limit:
                tmp[label] += 1
                all_image_labels.append(label)
        return all_image_labels, class_num

    def set_label_devide2(paths_bookmarks):
        image_labels = []
        paths = []
        class_num = 2
        tmp = [0] * class_num
        label_limit = float('inf')
        for id, item in paths_bookmarks.items():
            image_path, bookmark = item["path"], item["bookmark"]
            if bookmark <= 10 and tmp[0] < label_limit:
                label = 0
                tmp[0] += 1
            elif bookmark >= 1000  and tmp[1] < label_limit:
                label = 1
                tmp[1] += 1
            else:
                continue
            image_labels.append(label)
            paths.append(image_path)
            # print(image_path, bookmark, label)
        return paths, image_labels, class_num

    def set_label_interval(data):
        all_image_labels = []
        class_num = 3
        tmp = [0] * class_num
        label_limit = float('inf')
        for num in data:
            bookmark = data[num]['bookmark']
            if bookmark <= 5:
                label = 0
            elif 30 <= bookmark <= 50:
                label = 1
            elif 500 <= bookmark:
                label = 2
            else:
                continue
            if tmp[label] < label_limit:
                tmp[label] += 1
                all_image_labels.append(label)
        return all_image_labels, class_num
    
    def set_label_front_and_back(bookmarks):
        t = []
        for id, item in bookmarks.items():
            image_path, bookmark, view = item["path"], item["bookmark"], item["view"]
            t.append((bookmark, view, image_path)) # bookmarkとviewを分ける t.sort()で勝手にbookmark同数ならview順になる
        t.sort()

        image_labels = []
        paths = []
        num = int(len(t) * 0.1)

        for i in range(num):
            _, _, path = t[len(t)-num-i-1]
            _, _, rev_path = t[len(t)-i-1]
            image_labels.append(0)
            paths.append(path)
            image_labels.append(1)
            paths.append(rev_path)
        
        return paths, image_labels, 2
    
    def set_label_percent(bookmarks):
        t = []
        d = defaultdict()
        for id, item in bookmarks.items():
            image_path, bookmark, view = item["path"], item["bookmark"], item["view"]
            t.append((bookmark, view, image_path)) # bookmarkとviewを分ける t.sort()で勝手にbookmark同数ならview順になる
        t.sort(key=lambda x: x[0])
        n = len(t)
        for i in range(n):
            bookmark, _, path = t[i]
            per = (n-i) / n
            score = 1 - per
            d[path] = score
        t.sort(key=lambda x: x[1])
        for i in range(n):
            _, _, path = t[i]
            per = (n-i) / n
            score = (1 - per)
            d[path] += score
        t = sorted(d.items(), key=lambda x:x[1])
        image_labels = []
        paths = []
        num = int(len(t) * 0.1)
        for i in range(num):
            path, _ = t[len(d)-num-i]
            rev_path, _ = t[len(d)-i-1]
            image_labels.append(0)
            paths.append(path)
            image_labels.append(1)
            paths.append(rev_path)
        return paths, image_labels, 2

    def set_label_all(bookmarks):
        t = []
        for id, item in bookmarks.items():
            image_path, bookmark, view = item["path"], item["bookmark"], item["view"]
            t.append((bookmark, view, image_path)) # bookmarkとviewを分ける t.sort()で勝手にbookmark同数ならview順になる
        t.sort()

        split = split_array(t, 10)
        cnt = 1
        image_labels = []
        paths = []
        bm = []
        for part in split:
            for bookmark, view, path in part:
                if cnt <= 5:
                    paths.append(path)
                    image_labels.append(0)
                if cnt >= 6:
                    paths.append(path)
                    image_labels.append(1)
            cnt += 1
        return paths, image_labels, 2
