import sys
import matplotlib.pyplot as plt
import numpy as np
import PIL
import csv
import glob
import cv2
import os
import collections

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch
import torchvision
import torchvision.models as models
from tqdm import tqdm

from natsort import natsorted

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setting import config
from lib.set_label import SetLabel
from src.JsonLoadAndWrite import openJson


json_path = "/Users/haru0126/pixiv-image-data/experience/illustData.json"
image_dir = "/Users/haru0126/pixiv-image-data/experience"
#　パラメータ設定
epochs = 30      #学習回数
lr = 1e-3        #学習率
seed = 42

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, json_path, transform=None):
        x = []
        y = []
        paths = road_image_path(image_dir)
        data_json = openJson(json_path)
        for p in paths:
            filename = os.path.splitext(os.path.basename(p))[0]  # 拡張子を除去
            label = data_json[filename]["bookmark"]
            if label <= 30:
                x.append(p)
                y.append(np.log1p(float(label)))  # log(1+x)で正規化

        # 外れ値をパーセンタイルでクリップして過学習を抑制
        y_arr = np.array(y, dtype=np.float32)
        p_low, p_high = np.percentile(y_arr, [1, 99])
        y_arr = np.clip(y_arr, p_low, p_high)

        self.x = x
        self.y = torch.from_numpy(y_arr).float().view(-1, 1)
     
        self.transform = transform
  
  
    def __len__(self):
        return len(self.x)
  
  
    def __getitem__(self, i):
        img = PIL.Image.open(self.x[i]).convert('RGB')
        if self.transform is not None:
              img = self.transform(img)
    
        return img, self.y[i]


def road_image_path(image_dir):
    all_image_paths = list(glob.glob("{}/*/*.jpg".format(image_dir))) # 画像パスを全て取得
    return all_image_paths


def flatten(l):   #リストの1次元化
        for el in l:
            if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
                yield from flatten(el)
            else:
                yield el


# 訓練用: データ拡張あり
train_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

# 検証・テスト用: データ拡張なし
val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

def main():
    # 訓練用と検証用でtransformを分ける
    train_dataset_full = MyDataset(image_dir, json_path, transform=train_transform)
    val_dataset_full = MyDataset(image_dir, json_path, transform=val_transform)

    # シャッフルして分割（並び順の偏りを防ぐ）
    np.random.seed(seed)
    indices = np.random.permutation(len(train_dataset_full))
    train_num = int(len(train_dataset_full) * 0.7)
    val_num = int(len(train_dataset_full) * 0.1)

    train_dataset = torch.utils.data.Subset(train_dataset_full, indices[0:train_num])   #学習用データ（拡張あり）
    val_dataset = torch.utils.data.Subset(val_dataset_full, indices[train_num:train_num+val_num])  #検証用データ（拡張なし）
    test_dataset = torch.utils.data.Subset(val_dataset_full, indices[train_num+val_num:])     #テストデータ（拡張なし）

    print(f"full: {len(train_dataset_full)} -> train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weights = models.VGG16_Weights.IMAGENET1K_V1
    model = models.vgg16(weights=weights)

    # 特徴抽出部分を凍結
    for param in model.features.parameters():
        param.requires_grad = False

    # 分類層を回帰用に置き換え
    model.classifier[6] = nn.Linear(4096, 1)  # 回帰タスクなので出力を1にする
    model = model.to(device)
    print(device)

    #損失関数と最適化関数
    criterion = torch.nn.HuberLoss(delta=5.0)    # deltaを広げて学習の安定化
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=1e-3)  # 学習率を上げて収束を加速
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5  # 5エポック改善がなければ学習を止める

    train_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        epoch_loss = 0
        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss / len(train_loader)              

        with torch.no_grad():
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                epoch_val_loss += val_loss / len(valid_loader)

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - val_loss : {epoch_val_loss:.4f}\n"
        )

        train_loss_list.append(epoch_loss.cpu().detach().numpy())
        val_loss_list.append(epoch_val_loss.cpu().detach().numpy())
        
        # 学習率スケジューラを更新
        scheduler.step(epoch_val_loss)
        
        # 早期停止の判定
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            print(f"✓ Validation loss improved to {epoch_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"✗ Validation loss did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        print()
    
    plt.plot(train_loss_list, label='train')
    plt.plot(val_loss_list, label='valid')
    plt.legend()
    plt.show()

    test_loss=[]
    running_test_loss = 0.0
    pred=[]
    ans=[]
    with torch.set_grad_enabled(False):
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            list1 = labels.tolist()
            outputs = model(inputs)
            list2 = outputs.tolist()
            for i in range(len(list1)):
                ans.append(list1[i])
            for i in range(len(list2)):
                pred.append(list2[i])
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()

    test_loss.append(running_test_loss / len(testloader))

    print('test loss (log scale): {:.4f}'.format(running_test_loss / len(testloader)))

    #予測値と正解値の取得
    import collections
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    ans= list(flatten(ans))
    pred= list(flatten(pred))
    
    # log スケールから元のスケールに戻す
    ans = [np.expm1(a) for a in ans]
    pred = [np.expm1(p) for p in pred]
    
    # テスト結果の評価指標を計算
    ans_np = np.array(ans)
    pred_np = np.array(pred)
    
    mae = mean_absolute_error(ans_np, pred_np)
    mse = mean_squared_error(ans_np, pred_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(ans_np, pred_np)
    
    print("\n" + "="*50)
    print("テスト結果評価指標")
    print("="*50)
    print(f"MAE (平均絶対誤差): {mae:.4f}")
    print(f"MSE (平均二乗誤差): {mse:.4f}")
    print(f"RMSE (二乗平均平方根): {rmse:.4f}")
    print(f"R² (決定係数): {r2:.4f}")
    print("="*50 + "\n")

    x = []
    y = []
    paths = road_image_path(image_dir)
    data_json = openJson(json_path)
    for p in paths:
        filename = os.path.splitext(os.path.basename(p))[0]  # 拡張子を除去
        label = data_json[filename]["bookmark"]
        if label <= 30:
            x.append(p)
            y.append(float(label))

    imagelist=[]
    labellist=[]
    for i in range(40):
        imagelist.append(PIL.Image.open(x[train_num+val_num+i]))
        labellist.append(ans)

    fig = plt.figure(figsize=(10,6))
    for i, im in enumerate(imagelist):
        fig.add_subplot(4,10,i+1).set_title('{}\n{}'.format(int(pred[i]),int(ans[i])))
        plt.axis('off')
        plt.imshow(im)
    plt.show()


if __name__ == "__main__":
    main()