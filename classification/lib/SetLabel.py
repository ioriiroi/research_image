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

def set_label_devide2(data):
    all_image_labels = []
    class_num = 2
    for num in data:
        bookmark = data[num]['bookmark']
        if bookmark <= 10:
            label = 0
        elif bookmark >= 1000:
            label = 1
        else:
            continue
        all_image_labels.append(label)
    return all_image_labels, class_num

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