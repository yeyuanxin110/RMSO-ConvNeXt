# 论文附带快速傅里叶加速SSD
# 输入：参考图像特征，模板图像特征，搜索域；输出：SSD相关图
import numpy as np
import torch


def fft_match_batch(feature_ref, feature_t, search_rad=0):
    search_rad = int(search_rad)

    b, c, w, h = np.shape(feature_ref)
    if search_rad == 0:
        B, C, W, H = np.shape(feature_t)
        temp1 = w - W
        temp2 = h - H
    else:
        temp1 = 2 * search_rad
        temp2 = 2 * search_rad
    # torch.set_default_tensor_type(torch.DoubleTensor)
    # T = torch.zeros(np.shape(feature_ref))
    T = torch.zeros_like(feature_ref)
    T[:, :, 0:h - temp1, 0:w - temp2] = 1
    # T = T.cuda()

    sen_x = feature_ref ** 2
    tmp1 = torch.fft.fft2(sen_x)
    tmp2 = torch.fft.fft2(T)
    tmp_sum = torch.sum(tmp1 * torch.conj(tmp2), 1)

    ssd_f_1 = torch.fft.ifft2(tmp_sum)
    # ssd_fr_1 = torch.real(ssd_f_1)
    ssd_fr_1 = ssd_f_1.real
    ssd_fr_1 = ssd_fr_1[:, 0:temp1 + 1, 0:temp2 + 1]
    if search_rad != 0:
        ref_T = feature_t[:, :, search_rad:w - search_rad,
                search_rad:h - search_rad]
    else:
        ref_T = feature_t
    # ref_Tx = torch.zeros(np.shape(feature_ref))
    ref_Tx = torch.zeros_like(feature_ref)
    ref_Tx[:, :, 0:w - temp1, 0:h - temp2] = ref_T

    # ref_Tx = ref_Tx.cuda()

    tmp1 = torch.fft.fft2(feature_ref)
    tmp2 = torch.fft.fft2(ref_Tx)
    tmp_sum = torch.sum(tmp1 * torch.conj(tmp2), 1)

    ssd_f_2 = torch.fft.ifft2(tmp_sum)
    # tmp2x = torch.conj(tmp2)
    # ssd_fr_2 = torch.real(ssd_f_2)
    ssd_fr_2 = ssd_f_2.real
    # ssd_fr_2 = ssd_f_2[:, :, :, 0]
    ssd_fr_2 = ssd_fr_2[:, 0:temp1 + 1, 0:temp2 + 1]

    # ssd_fr_3 = torch.sum()

    ssd_batch = (ssd_fr_1 - 2 * ssd_fr_2) / w / h

    return ssd_batch


