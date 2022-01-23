import torch

epsilon = torch.tensor(0.1)
alpha_1 = 10 / 255
alpha_2 = 16 / 255


FREE_OPT_NUM = 4
PGD_OPT_NUM = 4