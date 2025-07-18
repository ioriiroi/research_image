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
        label = split_label(bookmark)
        if tmp[label] < label_limit:
            tmp[label] += 1
            all_image_labels.append(label)
    return all_image_labels, class_num

def set_label_devide2(image_paths, bookmarks):
    image_labels = []
    paths = []
    class_num = 2
    tmp = [0] * class_num
    label_limit = float("inf")
    for image_paths, bookmark in zip(image_paths, bookmarks):
        if bookmark <= 10 and tmp[0] < label_limit:
            label = 0
            tmp[0] += 1
        elif bookmark >= 50  and tmp[1] < label_limit:
            label = 1
            tmp[1] += 1
        else:
            continue
        image_labels.append(label)
        paths.append(image_paths)
        # print(image_paths, bookmark)
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