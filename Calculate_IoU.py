import numpy as np
from sklearn.metrics import accuracy_score
import nibabel as nib

# img_path2='F:\MedicalData\Task054_RenjiLiver\enjiliver_009.nii'          #Segm.nii跑出来的结果
# img_path ='F:\MedicalData\Task054_RenjiLiver\FFFF_label.nii'          #Mask.nii标的结果
img_path2='F:\MedicalData\Task054_RenjiLiver\enjiliver_008.nii'          #Segm.nii跑出来的结果
img_path ='F:\MedicalData\Task054_RenjiLiver\FFFF_label701.nii'          #Mask.nii标的结果
img = nib.load(img_path).get_fdata()  # 载入
img = np.array(img)
img2 = nib.load(img_path2).get_fdata()  # 载入
img2 = np.array(img2)
# 计算iou,tp,fp,tn,fn
c, w, h = img.shape
TP=0
TN=0
FN=0
FP=0
for i in range(0, c):
    for j in range(0, w):
        for k in range(0, h):
            # print(TP)
            TP += ((img2[i][j][k] == 1) & (img[i][j][k] == 1)).sum()
            # TN predict 和 label 同时为0
            TN += ((img2[i][j][k] == 0) & (img[i][j][k] == 0)).sum()
            # print(TN)
            # FN predict 0 label 1
            FN += ((img2[i][j][k] == 0) & (img[i][j][k] == 1)).sum()
            # FP predict 1 label 0
            FP += ((img2[i][j][k] == 1) & (img[i][j][k] == 0)).sum()

p = TP / (TP + FP)
r = TP / (TP + FN)
F1 = 2 * r * p / (r + p)
acc = (TP + TN) / (TP + TN + FP + FN)
iou=TP/(TP+FP+FN)

Dice=2*TP/(FP+2*TP+FN)
PA=(TP+TN)/(TP+TN+FP+FN)            #像素精度（Pixel Accuracy）标记正确的像素占总像素的百分比
print("iou: ")
print(iou)
print("acc: ")
print(acc)
print("precision: ")
print(p)
print("Dice: ")
print(Dice)
print("Pixel Accuracy: ")
print(PA)
