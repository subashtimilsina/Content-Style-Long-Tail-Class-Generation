# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
import time

#Import the VGG model for perceptual loss
from torchvision import models
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image


#Make class to use a vgg-16 to calculate perceptual loss
class PerceptualLoss:
    def __init__(self, device):
        self.device = device
        self.vgg = models.vgg16(pretrained=True).features.to(device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.loss_fn = torch.nn.MSELoss()
    
    def __call__(self, img1, img2):
        """
        Calculate perceptual loss between two batches of images using VGG-16 features.
        img1, img2: Input images of shape (N, C, H, W)
        Returns: Perceptual loss value
        """
        # Ensure img1 and img2 are batches of images
        if img1.ndim == 4:  # Batch of images
            img1_list = [self.transform(to_pil_image(img)) for img in img1]  # Convert each image to PIL and apply transform
            img1 = torch.stack(img1_list).to(self.device)  # Stack back into a batch tensor

        if img2.ndim == 4:  # Batch of images
            img2_list = [self.transform(to_pil_image(img)) for img in img2]  # Convert each image to PIL and apply transform
            img2 = torch.stack(img2_list).to(self.device)  # Stack back into a batch tensor

        # Extract features from VGG-16
        features1 = self.vgg(img1)
        features2 = self.vgg(img2)

        # Calculate perceptual loss
        loss = self.loss_fn(features1, features2)
        return loss

def fixed_sample_noise(indices, z_dim, datatype):
    """
    indices: List or 1D tensor of data indices in the batch
    shape: Shape of each noise sample, e.g., (3, 32, 32)
    fixed_indices_set: Set of indices where noise must be fixed
    """
    batch_noise = []
    for idx in indices:
        g = torch.Generator()
        g.manual_seed(idx.item())  # Deterministic seed
        noise = torch.randn(z_dim, generator=g, dtype=datatype)

        batch_noise.append(noise)
    
    # Stack into a batch tensor
    return torch.stack(batch_noise)

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, class_info, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, w_dim=512):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)


        self.head_classes = class_info[0]
        self.tail_classes = class_info[1]
        self.reconstruct_loss_func = torch.nn.MSELoss()
        self.id_mat = torch.eye(w_dim).to(device)

    def run_G(self, z, c, sync, mode=None):
        with misc.ddp_sync(self.G_mapping, sync):
            ws, style = self.G_mapping(z, c, mode=mode)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:], style = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True, mode=mode)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws, style

    def run_D(self, img, c, sync, mode=None):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c, mode=mode)
        return logits
    
    def normalize_embedding(self, embedding):
        with torch.no_grad():
            mu = embedding.mean(dim=0, keepdim=True)
            # std = embedding.std(dim=0, keepdim=True, unbiased=False) + 1e-3
        # embedding = (embedding - mu) / (std + 1e-3)
        return embedding - mu

    def accumulate_gradients(self, phase, real_img, real_c, indcs, gen_z, gen_c, sync, gain, cur_nimg, mode=None):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)


        if do_Gmain:
            # record_time = time.time()
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img_1, _gen_ws, style = self.run_G(gen_z, gen_c, mode=mode, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img_1, gen_c, mode=mode, sync=False)
                training_stats.report('Loss/scores/fake_1', gen_logits)
                training_stats.report('Loss/signs/fake_1', gen_logits.sign())
                loss_Gmain_1 = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                
                reduced_bs = gen_img_1.shape[0]//2
                
                # s_n_epsilon1 = _gen_ws[:reduced_bs, 0, :]
                # s_n_epsilon2 = _gen_ws[reduced_bs:, 0, :]
                s_n_epsilon1 = style[:reduced_bs, :]
                s_n_epsilon2 = style[reduced_bs:, :]
                loss_style_encoder = torch.linalg.norm(s_n_epsilon1 - s_n_epsilon2)

                #-------------------------------------------------------------------------------
                
                loss_Gmain = loss_Gmain_1 + 0.1 * loss_style_encoder

                training_stats.report('Loss/G/loss', loss_Gmain)
                
                training_stats.report('Loss/G/style_encoder', loss_style_encoder)
            # print("Gmain forward time:", time.time() - record_time)
            # record_time = time.time()
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()
            # print("Gmain backward time:", time.time() - record_time)

        # Gpl: Apply path length regularization.
        if do_Gpl:
            # record_time = time.time()
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws, _ = self.run_G(gen_z[:batch_size], gen_c[:batch_size], mode=mode,  sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()
            # print("Gpl backward time:", time.time() - record_time)
        # record_time = time.time()
        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws, _ = self.run_G(gen_z, gen_c, mode=mode,  sync=False)
                gen_logits = self.run_D(gen_img, gen_c, mode=mode,  sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, mode=mode, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()
        # print("Dreal backward time:", time.time() - record_time)
#----------------------------------------------------------------------------