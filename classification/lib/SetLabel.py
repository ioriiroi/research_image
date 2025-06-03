def split_label(bookmark):
    if bookmark < 3:
        return 0
    elif 3 <= bookmark < 20:
        return 1
    elif 20 <= bookmark < 150:
        return 2
    else:
        return 3

def set_label(data):
    all_image_labels = []
    tmp = [0] * 4
    for num in data:
        bookmark = data[num]['bookmark']
        label = split_label(bookmark)
        tmp[label] += 1
        all_image_labels.append(label)
    return all_image_labels