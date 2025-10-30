import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setting import config

class SetLabel:
    def get_bookmark(image_dir, data, image_names):
        paths_bookmarks = {}
        for name in image_names:
            id = name.split("_")[0]
            if id in data:
                paths_bookmarks[str(name)] = {"path": f"{image_dir}/{name}.jpg", "bookmark": data[id]["bookmark"]}
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
        label_limit = 400
        for id, item in paths_bookmarks.items():
            image_path, bookmark = item["path"], item["bookmark"]
            if bookmark <= 0 and tmp[0] < label_limit:
                label = 0
                tmp[0] += 1
            elif bookmark >= 5000  and tmp[1] < label_limit:
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
            image_path, bookmark = item["path"], item["bookmark"]
            t.append((bookmark, image_path))
        t.sort()

        image_labels = []
        paths = []
        num = 400

        for i in range(num):
            bookmark, path = t[i]
            rev_bookmark, rev_path = t[len(t)-i-1]
            image_labels.append(0)
            paths.append(path)
            image_labels.append(1)
            paths.append(rev_path)
        return paths, image_labels, 2