import torch
import numpy as np
from .basic_module import BasicModule
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

from .unet import UNet


def dataConsistency_NullSpace(R_output, original_img, P, mu, xscale, iter_cg=5):
    """
    Args:
        R_output: (batch, C, H, W) - 네트워크 출력
        original_img: (batch, C, H, W) - 원본 이미지 x
        P: (batch, H*W, 2) or list of (H*W, 2) - 미리 저장된 N matrix
        mu: regularization parameter
    """
    batch_size = R_output.shape[0]
    H, W = R_output.shape[2], R_output.shape[3]
    results = []

    for b in range(batch_size):
        net_output = R_output[b:b+1]  # (1, C, H, W)
        
        # P 가져오기
        P_b = P[b] if isinstance(P, list) else P[b]  # (H*W, 2)
        
        x_tilde = foreground_projection(original_img[b].view(-1), P_b)  # (H*W,)
        AT_x_tilde = foreground_projection(x_tilde, P_b)

        AT_x_tilde = x_tilde.reshape(1, 1, H, W)

        AT_x_tilde = AT_x_tilde.to(net_output.device) # GPU로 이동

        rhs = AT_x_tilde + mu * xscale * net_output
        
        # CG
        s = torch.zeros_like(net_output)
        r = rhs - PtP_operation(s, P_b, mu)
        p = r.clone()
        rsold = torch.sum(r * r)
        
        for i in range(iter_cg):
            Ap = PtP_operation(p, P_b, mu)
            alpha = rsold / torch.sum(p * Ap)
            s = s + alpha * p
            r = r - alpha * Ap
            rsnew = torch.sum(r * r)
            
            if torch.sqrt(rsnew) < 1e-6:
                break
            
            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew
        
        results.append(s)

    return torch.cat(results, dim=0)

def PtP_operation(s, P, mu):
    """
    (I - PP^T + μI) s 연산 수행
    
    Args:
        s : (1, C, H, W)
        P : (H*W, 2) - N matrix
        mu : scalar
    """
    # (I - PP^T) @ s
    
    foreground_s = foreground_projection(s, P)
    
    foreground_s2 = foreground_projection(foreground_s, P)
    # (I - PP^T) @ s + μs = foreground_s + μs
    result = (foreground_s2 + mu * s.flatten()).reshape_as(s)
    
    return result


def foreground_projection(x, P):
    """
    전경 계산 : (I - PP^T) @ x

    Args : 
        x : (H*W, ) or (batch, C, H, W)
        P : (H*W, 2) - U[:, :2] = N
    
    Returns :
        foreground : same shape as x
    """

    x_flat = x.flatten()  # numpy array

    background = P @ (P.T @ x_flat) #(H*W, 2) @ (2, 1) = (H*W, )
    
    foreground = x_flat - background # (I - PP^T) @ x

    return foreground

class UnrollNet(BasicModule):

    def __init__(self, in_channels=1, unroll_iter=10, iter_cg=10):
        super(UnrollNet, self).__init__()
        self.R = nn.ModuleList([UNet(in_chns=in_channels, class_num=1) for i in range (unroll_iter)])
        # self.R = UNet(in_chns=in_channels, class_num=1)
        # self.U = UNet(in_chns=in_channels, class_num=1)
        self.iter = unroll_iter
        self.iter_cg = iter_cg
        self.mu = nn.ParameterList([nn.Parameter(torch.tensor(1.0, dtype=torch.float32)) for i in range(unroll_iter)])
        self.xscale = nn.ParameterList([nn.Parameter(torch.tensor(1.0, dtype=torch.float32)) for i in range(unroll_iter)])

    def forward(self, x, original_img, P):
        """
        Args:
            x: (batch, 1, H, W) - 초기 입력 (보통 original_img와 같음)
            original_img: (batch, 1, H, W) - 원본 ultrasound 이미지
            P: (batch, H*W, 2) or list of (H*W, 2) - 미리 저장된 N matrix
        
        Returns:
            dc_output: (batch, 1, H, W) - 최종 needle segmentation
        """
        dc_output = x.clone()
        net_output_list = []
        dc_output_list = []
        
        for i in range(self.iter):
            # Regularization: Unet 통과
            
            dc_output = self.R[i](dc_output)
            net_output_list.append(dc_output)
            # Data Consistency: Null space projection 기반
            dc_output = dataConsistency_NullSpace(
                dc_output, 
                original_img, 
                P, 
                self.mu[i], 
                self.xscale[i],
                iter_cg=self.iter_cg
            )
            
            dc_output_list.append(dc_output)
            
        
        return dc_output, [net_output_list, dc_output_list]
    


