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
    



# def coilComb(x, sens_map):  #(b, c, y, z) -> (b, 1, y, z)
#     mul = x * torch.conj(sens_map)  #(batch, c, y, z)
#     result = torch.sum(mul, dim=1, keepdim=True)  #(batch, 1, y, z)
#     return result  

# def coilDeComp(x, sens_map):  #(b, 1, y, z) -> (b, c, y, z)
#     # 동적으로 coil 수 가져오기

#     mc_img = x * sens_map  #(b, c, y, z)
#     return mc_img #(b, c, y, z)

# def fft2c(x):

#     # BART -u 스타일: 1/√N 정규화
#     H, W = x.shape[-2], x.shape[-1]
#     N = H * W
#     norm_factor = torch.sqrt(torch.tensor(N, dtype=torch.float32, device=x.device))
        
#     return torch.fft.fftshift(
#             torch.fft.fft2(
#                 torch.fft.ifftshift(x, dim=(-2, -1)), 
#                 dim=(-2, -1), norm='backward'
#             ),
#             dim=(-2, -1)
#     ) / norm_factor
    
#     # # 기본 PyTorch 방식
#     # return torch.fft.fftshift(
#     #     torch.fft.fft2(
#     #         torch.fft.ifftshift(x, dim=(-2, -1)), 
#     #         dim=(-2, -1)
#     #     ),
#     #     dim=(-2, -1)
#     # )

# def ifft2c(x):

    
#     # BART -u 스타일: 1/√N 정규화
#     H, W = x.shape[-2], x.shape[-1]
#     N = H * W  
#     norm_factor = torch.sqrt(torch.tensor(N, dtype=torch.float32, device=x.device))
        
#     return torch.fft.ifftshift(
#                     torch.fft.ifft2(
#                         torch.fft.fftshift(x, dim=(-2, -1)),
#                         dim=(-2, -1), norm='backward'
#                     ),
#                     dim=(-2, -1)
#             ) * norm_factor
            
#             #     # 기본 PyTorch 방식
#             # return torch.fft.ifftshift(
#             #         torch.fft.ifft2(
#             #             torch.fft.fftshift(x, dim=(-2, -1)),
#             #             dim=(-2, -1)
#             #         ),
#             #         dim=(-2, -1)
#             # )

# def dataConsistency(R_output, sens_map, dc_kspace, mask):
#     """
#     Improved data consistency with proper normalization and cleaner implementation
    
#     Args:
#         R_output: (batch, 2, y, z) - network output [real, imag]
#         sens_map: (batch, c, y, z) - sensitivity maps (complex)
#         dc_kspace: (batch, c, y, z) - measured k-space data (complex)
#         mask: (batch, c, y, z) or (batch, 1, y, z) - sampling mask
#     """
#     # Convert network output to complex
#     cpx_output = R_output[:, :1, ...] + 1j * R_output[:, 1:, ...]  # (batch, 1, y, z)
    
#     # Decompose to multi-coil images
#     mc_img = coilDeComp(cpx_output, sens_map)  # (b, 1, y, z) -> (b, c, y, z)

#     # Transform to k-space with proper normalization
#     mc_kspace = fft2c(mc_img)
    
#     # Data consistency: replace measured k-space locations
#     # mask should be 1 where data is measured, 0 where not measured
#     if mask.dim() == 3:  # (batch, y, z)
#         mask = mask.unsqueeze(1)  # (batch, 1, y, z)
    
#     # Broadcast mask to all coils if needed
#     if mask.shape[1] == 1 and mc_kspace.shape[1] > 1:
#         mask = mask.repeat(1, mc_kspace.shape[1], 1, 1)
    
#     # Data consistency: keep measured data, replace unmeasured with network prediction
#     mc_dc = torch.where(mask.bool(), dc_kspace, mc_kspace)

#     # Transform back to image space
#     mc_img_dc = ifft2c(mc_dc)

#     # Combine coils
#     cpx_img = coilComb(mc_img_dc, sens_map)  # (b, c, y, z) -> (b, 1, y, z)
    
#     # Convert back to real format
#     result_img = torch.cat((torch.real(cpx_img), torch.imag(cpx_img)), dim=1).to(torch.float32)
#     result_kspace = torch.cat((torch.real(mc_dc), torch.imag(mc_dc)), dim=1).to(torch.float32)
    
#     return result_img, result_kspace

# # Alternative: CG-based data consistency (more sophisticated)
# def dataConsistency_CG(R_output, sens_map, dc_kspace, mask, mu, init_img, iter_cg=5):
#     """
#     CG-based data consistency for better global consistency
#     """
#     batch_size = R_output.shape[0]
    
    
#     results_img = []
    
    
#     for b in range(batch_size):
#         # Convert to complex
#         cpx_output = R_output[b:b+1, :1, ...] + 1j * R_output[b:b+1, 1:, ...]  # (1, 1, y, z)
#         sens_b = sens_map[b:b+1]  # (1, c, y, z)
#         dc_kspace_b = dc_kspace[b:b+1]  # (1, c, y, z)
#         mask_b = mask[b:b+1] if mask.dim() == 4 else mask[b:b+1].unsqueeze(1)  # (1, c, y, z)
        
#         # Compute RHS: E^h * y (measured k-space transformed to image)
        
#         init_img2 = init_img[b:b+1, :1, ...] + 1j * init_img[b:b+1, 1:, ...]
#         rhs = init_img2 + mu*cpx_output # (1, 1, y, z)
        
#         # Initialize CG
#         # x = cpx_output.clone()  # (1, 1, y, z)  warmstart
#         x = torch.zeros_like(cpx_output) #coldstart
        

#         r = rhs - EhE_operation(x, sens_b, mask_b, mu)
#         p = r.clone()
#         rsold = torch.sum(torch.conj(r) * r).real
        
#         # CG iterations
#         for i in range(iter_cg):
#             Ap = EhE_operation(p, sens_b, mask_b, mu)
#             alpha = rsold / torch.sum(torch.conj(p) * Ap).real
#             x = x + alpha * p
#             r = r - alpha * Ap
#             rsnew = torch.sum(torch.conj(r) * r).real
            
#             if torch.sqrt(rsnew) < 1e-6:
#                 break
                
#             beta = rsnew / rsold
#             p = r + beta * p
#             rsold = rsnew
        
#         # Get final k-space
#         # mc_img = coilDeComp(x, sens_b)
#         # mc_kspace = fft2c(mc_img)
        
#         # Convert to real
#         result_img = torch.cat((torch.real(x), torch.imag(x)), dim=1).to(torch.float32)
#         # result_kspace = torch.cat((torch.real(mc_kspace), torch.imag(mc_kspace)), dim=1).to(torch.float32)
        
#         results_img.append(result_img)
#         # results_kspace.append(result_kspace)
    
#     # return torch.cat(results_img, dim=0), torch.cat(results_kspace, dim=0)
#     return torch.cat(results_img, dim=0)

# def EhE_operation(img, sens_map, mask, mu):
#     """
#     Performs E^h * E operation: image -> k-space -> mask -> image
#     """
#     # img: (1, 1, y, z), sens_map: (1, c, y, z), mask: (1, c, y, z)
#     mc_img = coilDeComp(img, sens_map)  # (1, c, y, z)
#     kspace = fft2c(mc_img)  # (1, c, y, z)
#     masked_kspace = kspace * mask  # (1, c, y, z)
#     img_back = ifft2c(masked_kspace)  # (1, c, y, z)
#     combined = coilComb(img_back, sens_map)  # (1, 1, y, z)
#     return combined + mu * img