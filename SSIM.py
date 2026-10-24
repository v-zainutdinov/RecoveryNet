import cv2
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

#512,512,3
img_array: np.ndarray = cv2.imread('5.png')
img_chw = img_array.transpose(2, 0, 1) # С, H, W
tensor_image = torch.from_numpy(img_chw) / 255.0
C, H, W = img_chw.shape

W_Gausse = np.array([
    [0.000058, 0.000275, 0.000834, 0.001625, 0.002030, 0.001625, 0.000834, 0.000275, 0.000058],
    [0.000275, 0.001301, 0.003953, 0.007700, 0.009616, 0.007700, 0.003953, 0.001301, 0.000275],
    [0.000834, 0.003953, 0.012009, 0.023391, 0.029211, 0.023391, 0.012009, 0.003953, 0.000834],
    [0.001625, 0.007700, 0.023391, 0.045559, 0.056896, 0.045559, 0.023391, 0.007700, 0.001625],
    [0.002030, 0.009616, 0.029211, 0.056896, 0.071054, 0.056896, 0.029211, 0.009616, 0.002030],
    [0.001625, 0.007700, 0.023391, 0.045559, 0.056896, 0.045559, 0.023391, 0.007700, 0.001625],
    [0.000834, 0.003953, 0.012009, 0.023391, 0.029211, 0.023391, 0.012009, 0.003953, 0.000834],
    [0.000275, 0.001301, 0.003953, 0.007700, 0.009616, 0.007700, 0.003953, 0.001301, 0.000275],
    [0.000058, 0.000275, 0.000834, 0.001625, 0.002030, 0.001625, 0.000834, 0.000275, 0.000058],
])
W_Gausse_tensor=torch.from_numpy(W_Gausse).float().unsqueeze(dim=0).unsqueeze(dim=0).repeat(C, 1, 1, 1)

stride: int = 1
padding: int = 4
kernel_size: tuple[int, int] = (9, 9)
        
# H_out = int((H + 2 * padding - kernel_size[0]) / stride[0] + 1), для 9x9 
H_out: int = int((H + 2 * padding - kernel_size[0]) / stride + 1)
W_out: int = int((W + 2 * padding - kernel_size[1]) / stride + 1)
print(f"Финальные размеры: высота:{H_out} ; ширина:{W_out}")
        
C1=(0.01*1)**2
C2=(0.03*1)**2


img_pad = np.pad(img_chw, ((0, 0), (padding, padding), (padding, padding)), mode='reflect')
#print(f"Размер исходного X: {img_pad.shape}")


#x=torch.from_numpy(img_pad).unsqueeze(dim=0) / 255.0
#print(f"Размер X готовый к подачи через torch: {x.shape}")

mu_x=F.conv2d(input=tensor_image, weight=W_Gausse_tensor, stride=stride, padding=padding, groups=C)
mu_y=F.conv2d(input=tensor_image, weight=W_Gausse_tensor, stride=stride, padding=padding, groups=C)

var_x=F.conv2d(tensor_image*tensor_image, weight=W_Gausse_tensor, stride=stride, padding=padding, groups=C) - mu_x**2
var_y=F.conv2d(tensor_image*tensor_image, weight=W_Gausse_tensor, stride=stride, padding=padding, groups=C) - mu_y**2
covxy=F.conv2d(tensor_image*tensor_image, weight=W_Gausse_tensor, stride=stride, padding=padding, groups=C) - mu_x * mu_y

SSIM=((2*mu_x*mu_y+C1)*(2*covxy+C2))/((mu_x**2+mu_y**2+C1)*(var_x+var_y+C2))
M = SSIM.mean()
print(M.item())

M=0
SSIM=0

for i in range(0, H_out):
    for j in range(0, W_out):
        SSIM0=0
        SSIM1=0
        SSIM2=0
        for _ in range(C):
            if _ == 0:
                x_ij0=img_pad[_, i*stride:i*stride+kernel_size[0], j*stride:j*stride+kernel_size[1]]
                mmean_0=np.sum(x_ij0*W_Gausse)
                disp_0=np.sum(W_Gausse*(x_ij0-mmean_0)**2)
                cov_0=np.sum(W_Gausse*((x_ij0-mmean_0)*(x_ij0-mmean_0)))
                
                SSIM0+=((2*mmean_0*mmean_0+C1)*(2*cov_0+C2))/((mmean_0**2+mmean_0**2+C1)*(disp_0+disp_0+C2))
                
            elif _ == 1:
                x_ij1=img_pad[_, i*stride:i*stride+kernel_size[0], j*stride:j*stride+kernel_size[1]]
                mmean_1=np.sum(x_ij1*W_Gausse)
                disp_1=np.sum(W_Gausse*(x_ij1-mmean_1)**2)
                cov_1=np.sum(W_Gausse*((x_ij1-mmean_1)*(x_ij1-mmean_1)))
                                
                SSIM1+=((2*mmean_1*mmean_1+C1)*(2*cov_1+C2))/((mmean_1**2+mmean_1**2+C1)*(disp_1+disp_1+C2))
            else:
                x_ij2=img_pad[_, i*stride:i*stride+kernel_size[0], j*stride:j*stride+kernel_size[1]]
                mmean_2=np.sum(x_ij2*W_Gausse)
                disp_2=np.sum(W_Gausse*(x_ij2-mmean_2)**2)
                cov_2=np.sum(W_Gausse*((x_ij2-mmean_2)*(x_ij2-mmean_2)))
                                
                SSIM2+=((2*mmean_2*mmean_2+C1)*(2*cov_2+C2))/((mmean_2**2+mmean_2**2+C1)*(disp_2+disp_2+C2))
                
        SSIM+=(SSIM0+SSIM1+SSIM2)/3
        M+=1
        
print(f"Циклом: {SSIM/M}")


            
            
            
    