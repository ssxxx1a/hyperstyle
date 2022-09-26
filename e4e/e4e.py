import matplotlib

matplotlib.use('Agg')
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch import nn
from e4e.encoders import psp_encoders
from e4e.stylegan2.model import Generator
from options.path_option import model_paths, stylegan_weights
from time import time
import math
from models.mapper import LevelsMapper
from e4e.stylegan2.model import EqualLinear, PixelNorm


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class eFe(nn.Module):

    def __init__(self, opts):
        super(eFe, self).__init__()
        self.opts = opts
        # Define architecture
        self.encoder = self.set_encoder()
        self.decoder = Generator(opts.stylegan_size, 512, 8, channel_multiplier=2)

        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        if opts.use_au:
            if opts.space_type == 'w+':
                self.au2W = LevelsMapper(opts=opts, input_c=17, n_layer=1)
            elif opts.space_type == 'w':
                self.au2W = EqualLinear(512 + 17, 512, lr_mul=0.01, activation="fused_lrelu")
        # self.load_weights()

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'Encoder4Editing':
            encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self, map_location='cpu'):

        if self.opts.e4e_checkpoint_path is not None:
            print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.opts.e4e_checkpoint_path))
            ckpt = torch.load(self.opts.e4e_checkpoint_path, map_location=map_location)  # 'map_location='cpu'

            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=False)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=False)

            self.__load_latent_avg(ckpt)
        else:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('Loading decoder weights from pretrained!')
            ckpt = torch.load(model_paths['stylegan_ffhq'])
            self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
            self.__load_latent_avg(ckpt, repeat=self.encoder.style_count)

    def forward(self, x, aus=None, isfusion=False, resize=True, latent_mask=None, input_code=False,
                randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None, need_mapper=True, get_w=False, skip_feature=None):
        # input是code 同时fusion存在
        if input_code:
            codes = x
            if isfusion:
                assert skip_feature != None, "why skip_f is None??"

                fusion_f = skip_feature
        else:
            if get_w:
                # 只是为了得到w
                if isfusion:
                    codes, fusion_f = self.encoder(x, isfusion=isfusion)
                    return codes, fusion_f
                else:
                    codes = self.encoder(x, isfusion=isfusion)
                    return codes
            else:
                if isfusion:
                    codes, fusion_f = self.encoder(x, isfusion=isfusion)
                else:
                    codes = self.encoder(x, isfusion=isfusion)
            # normalize with respect to the center of an average face
            if self.opts.start_from_latent_avg:
                if codes.ndim == 2:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
        # # for au change
        # W_au = self.AuToW(aus)
        #
        # W_au = W_au.repeat(codes.shape[1], 1)
        #
        # codes=codes+W_au # add au into codes
        # bsx14x512

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0
        if need_mapper:
            if self.opts.space_type == 'w+':
                # 保证输入是bs  17
                aus = aus.unsqueeze(1)
                # bs 18 512
                # bs 1 17 ~ bs 18 17
                aus = aus.repeat(1, codes.shape[1], 1)
                au_w = self.au2W(aus)
                codes = codes + self.opts.lambda_auw * au_w
                # codes[:, :4, :] += self.opts.lambda_auw * au_w[:, :4, :]
                # codes[:, 4: 8, :] += self.opts.lambda_auw * au_w[:, 4:8, :]
                # codes[:, 8:, :] += self.opts.lambda_auw * au_w[:, 8:, :]
                # n_c = torch.cat([codes, aus], dim=2)
                # codes += 0.2 * self.au2W(n_c)
                # n_code = []
                # for i in range(self.n_latent):
                #     n_code.append(self.au2W[i](codes[:, i]))
                # codes = torch.stack(n_code, dim=1)

        input_is_latent = not input_code
        if isfusion:
            images, result_latent = self.decoder([codes], skp_feature=fusion_f,
                                                 input_is_latent=input_is_latent,
                                                 randomize_noise=randomize_noise,
                                                 return_latents=return_latents)
        else:
            images, result_latent = self.decoder([codes],
                                                 input_is_latent=input_is_latent,
                                                 randomize_noise=randomize_noise,
                                                 return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None
