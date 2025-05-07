from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


Sample = namedtuple('Sample', 'trajectories values chains')

from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

Sample = namedtuple('Sample', 'trajectories values chains')

# 모델 타입 선택을 위한 글로벌 변수
FLOWMATCHING_MODE = False

def set_model_mode(prefix):
    """
    모델 모드를 설정하는 함수
    train.py, plan_guided.py 등에서 호출하여 전역 모드 설정
    """
    global FLOWMATCHING_MODE
    FLOWMATCHING_MODE = prefix.startswith('flowmatching')
    print(f"모델 모드 설정: {'Flowmatching' if FLOWMATCHING_MODE else 'Diffusion'}")
    return FLOWMATCHING_MODE

@torch.no_grad()
def default_sample_fn(model, x, cond, t):
    """
    통합된 샘플링 함수로, 전역 모드에 따라 다른 샘플링 방식 사용
    """
    if FLOWMATCHING_MODE:
        # flowmatching 방식
        x_less_noisy = model.p_mean_variance(x=x, cond=cond, t=t)
        values = torch.zeros(len(x), device=x.device)
        return x_less_noisy, values
    else:
        # 기존 diffusion 방식
        model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
        model_std = torch.exp(0.5 * model_log_variance)

        # no noise when t == 0
        noise = torch.randn_like(x)
        noise[t == 0] = 0

        values = torch.zeros(len(x), device=x.device)
        return model_mean + model_std * noise, values

def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values

def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
        n_sample_timesteps=1
    ):
        super().__init__()
        
        # 현재 모드 출력 (디버깅용)
        if FLOWMATCHING_MODE:
            print("imported diffusion.py for flowmatching")
        else:
            print("imported diffusion.py for diffusion")
        
        self.mode = FLOWMATCHING_MODE
        self.model = model
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.n_timesteps = n_timesteps
        self.n_sample_timesteps = n_sample_timesteps
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon


        # 모델 타입에 따라 초기화 방식 분기
        if FLOWMATCHING_MODE:
            # Flowmatching 설정
            self.register_flowmatching_parameters()
        else:
            # 기존 Diffusion 설정
            self.register_diffusion_parameters()

        # 손실 가중치 설정
        loss_weights = self.get_loss_weights(
            action_weight=action_weight,
            discount=loss_discount,
            weights_dict=loss_weights,
        )
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def register_diffusion_parameters(self):
        """기존 diffusion에 필요한 파라미터 등록"""
        betas = cosine_beta_schedule(self.n_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        # register buffers for diffusion parameters
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        
        sample_alphas = alphas.view(self.n_sample_timesteps, -1).prod(dim=1)
        sample_betas = 1. - sample_alphas

        sample_alphas_cumprod = torch.cumprod(sample_alphas, axis=0)
        sample_alphas_cumprod_prev = torch.cat([torch.ones(1), sample_alphas_cumprod[:-1]])
        
        
        self.sample_betas = sample_betas
        self.sample_alphas_cumprod = sample_alphas_cumprod
        self.sample_alphas_cumprod_prev = sample_alphas_cumprod_prev
        
        self.sample_sqrt_alphas_cumprod = torch.sqrt(sample_alphas_cumprod)
        self.sample_sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - sample_alphas_cumprod)
        self.sample_log_one_minus_alphas_cumprod = torch.log(1. - sample_alphas_cumprod)
        self.sample_sqrt_recip_alphas_cumprod = torch.sqrt(1. / sample_alphas_cumprod)
        self.sample_sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / sample_alphas_cumprod - 1)

        sample_posterior_variance = sample_betas * (1. - sample_alphas_cumprod_prev) / (1. - sample_alphas_cumprod)
        self.sample_posterior_variance = sample_posterior_variance
        self.sample_posterior_log_variance_clipped = torch.log(torch.clamp(sample_posterior_variance, min=1e-20))
        self.sample_posterior_mean_coef1 = sample_betas * np.sqrt(sample_alphas_cumprod_prev) / (1. - sample_alphas_cumprod)
        self.sample_posterior_mean_coef2 = (1. - sample_alphas_cumprod_prev) * np.sqrt(sample_alphas) / (1. - sample_alphas_cumprod)

        self.device = self.betas.device

    def register_flowmatching_parameters(self):
        """flowmatching에 필요한 파라미터 등록"""
        # flowmatching에는 최소한의 매개변수만 필요
        print("register_flowmatching_parameters")
        betas = torch.ones(self.n_timesteps)  # 단순화된 파라미터
        self.betas = betas
        self.device = self.betas.device
        self.theta_min = 0.0

    def get_loss_weights(self, action_weight, discount, weights_dict):
        """
        손실 가중치 설정 - 두 모델 타입 모두 사용하는 공통 메서드
        """
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        # 관측값 손실의 차원 별 가중치 설정
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        # 시간에 따른 손실 감쇠: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        # 첫 행동의 가중치 수동 설정
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #=====================================================
    #                     planning
    #=====================================================

    def predict_start_from_noise(self, x_t, t, noise):
        """
        noise에서 x_0 예측 (diffusion 모델용)
        """
        # 현재 텐서의 디바이스 확인
        current_device = x_t.device
        
        # 필요한 텐서들을 현재 디바이스로 이동
        sample_sqrt_recip_alphas_cumprod = self.sample_sqrt_recip_alphas_cumprod.to(current_device) 
        sample_sqrt_recipm1_alphas_cumprod = self.sample_sqrt_recipm1_alphas_cumprod.to(current_device)
        
        if self.predict_epsilon:
            return (
                extract(sample_sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(sample_sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        """
        posterior q(x_{t-1} | x_t, x_0) 계산 (diffusion 모델용)
        """
        # 현재 텐서의 디바이스 확인
        current_device = x_t.device
        
        # 필요한 텐서들을 현재 디바이스로 이동
        sample_posterior_mean_coef1 = self.sample_posterior_mean_coef1.to(current_device)
        sample_posterior_mean_coef2 = self.sample_posterior_mean_coef2.to(current_device)
        sample_posterior_variance = self.sample_posterior_variance.to(current_device)
        sample_posterior_log_variance_clipped = self.sample_posterior_log_variance_clipped.to(current_device)
        
        sample_posterior_mean = (
            extract(sample_posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(sample_posterior_mean_coef2, t, x_t.shape) * x_t
        )
        sample_posterior_variance = extract(sample_posterior_variance, t, x_t.shape)
        sample_posterior_log_variance_clipped = extract(sample_posterior_log_variance_clipped, t, x_t.shape)
        return sample_posterior_mean, sample_posterior_variance, sample_posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        """
        예측 평균 및 분산 계산
        모델 타입에 따라 다른 구현 사용
        """
        # 모든 입력을 동일한 디바이스로 이동
        device = x.device
        t = t.to(device)
        
        if FLOWMATCHING_MODE:
            # flowmatching 방식의 구현
            # 모델을 현재 디바이스로 이동
            self.model = self.model.to(device)
            model_output = self.model(x, cond,  t * (self.n_timesteps // self.n_sample_timesteps))
            
            if self.predict_epsilon:
                # x_start를 직접 예측하는 방식
                # v_t(x)를 (f(x,t)-x)/(1-t) 형태로 표현
                t_normalized = t.float() / self.n_sample_timesteps
                t_normalized = t_normalized.view(-1, 1, 1).to(device)
                x_less_noisy = x + (model_output - x)/(1.0 - t_normalized) * (1.0/self.n_sample_timesteps)
            else:
                # velocity를 예측하는 방식
                x_less_noisy = x + model_output * (1.0/self.n_sample_timesteps)
                
            x_less_noisy = apply_conditioning(x_less_noisy, cond, self.action_dim)
            return x_less_noisy
        else:
            # 기존 diffusion 방식의 구현
            # 모델을 현재 디바이스로 이동
            self.model = self.model.to(device)
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond,  ((t+1) * (self.n_timesteps // self.n_sample_timesteps) -1)))

            if self.clip_denoised:
                x_recon.clamp_(-1., 1.)

            model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                    x_start=x_recon, x_t=x, t=t)
            return model_mean, posterior_variance, posterior_log_variance

    #=====================================================

    def q_sample(self, x_start, t, noise=None):
        """
        모델 타입에 따라 다른 구현 사용
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        if FLOWMATCHING_MODE:
            # flowmatching 방식의 구현
            theta_min = self.theta_min
            t_normalized = t / self.n_timesteps
            t_normalized = t_normalized.view(-1, 1, 1)
            return (1 - (1 - theta_min) * t_normalized) * noise + t_normalized * x_start
        else:
            # 기존 diffusion 방식의 구현
            current_device = x_start.device
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(current_device)
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(current_device)
            
            sample = (
                extract(sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
            )
            return sample

    def p_losses(self, x_start, cond, t):
        """
        손실 계산
        모델 타입에 따라 다른 구현 사용
        """
        noise = torch.randn_like(x_start)
        
        if FLOWMATCHING_MODE:
            # flowmatching 방식의 손실 계산
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

            theta_min = self.theta_min
            v_t_true = x_start - (1 - theta_min) * noise
            
            if self.predict_epsilon:
                target = v_t_true
            else:
                target = x_start
            
            model_output = self.model(x_noisy, cond, t)
            model_output = apply_conditioning(model_output, cond, self.action_dim)
            target = apply_conditioning(target, cond, self.action_dim)

            assert model_output.shape == target.shape
            loss, info = self.loss_fn(model_output, target)
        else:
            # 기존 diffusion 방식의 손실 계산
            x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_t = apply_conditioning(x_t, cond, self.action_dim)

            if self.predict_epsilon:
                target = noise
            else:
                target = x_start

            model_output = self.model(x_t, cond, t)
            model_output = apply_conditioning(model_output, cond, self.action_dim)

            loss, info = self.loss_fn(model_output, target)
        
        return loss, info

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t)

    #=====================================================

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None

        if FLOWMATCHING_MODE:
            # Flowmatching: 순방향 샘플링 (0→T)
            n_steps = self.n_sample_timesteps
            timesteps = range(0, n_steps)
        else:
            # Diffusion: 역방향 샘플링 (T→0)
            n_steps = self.n_sample_timesteps
            timesteps = reversed(range(0, n_steps))

        progress = utils.Progress(n_steps) if verbose else utils.Silent()
        
        # 모델을 현재 디바이스로 이동
        self.model = self.model.to(device)
        
        for i in timesteps:
            t = make_timesteps(batch_size, i, device)
            # 모든 텐서가 동일한 디바이스에 있는지 확인
            t = t.to(device)
            x = x.to(device)
            
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()

        x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, **sample_kwargs)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)


class ValueDiffusion(GaussianDiffusion):
    """
    가치 함수용 diffusion 모델
    """
    def p_losses(self, x_start, cond, target, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        pred = self.model(x_noisy, cond, t)

        loss, info = self.loss_fn(pred, target)
        return loss, info

    def forward(self, x, cond, t):
        return self.model(x, cond, t)
# diffuser/models/diffusion.py (통합 버전)
