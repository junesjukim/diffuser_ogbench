import torch

from diffuser.models.helpers import (
    extract,
    apply_conditioning,
)
from diffuser.models.diffusion import FLOWMATCHING_MODE, set_model_mode

@torch.no_grad()
def n_step_guided_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,
):
    """
    통합된 가이드 샘플링 함수 (diffusion 및 flowmatching 모두 지원)
    
    인자:
      model: diffusion 또는 flowmatching 방식의 모델
      x: 현재 샘플 텐서
      cond: 조건 정보
      t: 현재 timestep (각 배치마다 동일 혹은 다르게)
      guide: gradient 정보를 제공하는 객체 (guide.gradients 메서드가 있어야 함)
      scale: gradient 업데이트 스케일 (기본값: 0.001)
      t_stopgrad: t 미만에서는 가이드 gradient를 무시 (기본값: 0)
      n_guide_steps: 가이드 업데이트를 몇 번 수행할지 (기본값: 1)
      scale_grad_by_std: posterior variance로 스케일링 여부 (기본값: True)
    """

    # 모든 텐서를 동일한 디바이스로 이동
    device = x.device
    t = t.to(device)

    # 모델의 mode 속성을 우선 확인하고, 없는 경우에만 전역 변수 사용
    is_flowmatching = getattr(model, 'mode', FLOWMATCHING_MODE)
    print("FLOWMATCHING_MODE in functions.py: ", is_flowmatching, flush=True)
    
    # 모델과 가이드를 현재 디바이스로 이동
    model = model.to(device)
    guide.model = guide.model.to(device)
    
    if not is_flowmatching:
        # Diffusion 방식
        model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
        model_std = torch.exp(0.5 * model_log_variance)
        model_var = torch.exp(model_log_variance)
    
    for _ in range(n_guide_steps):
        with torch.enable_grad():
            if is_flowmatching:
                sample_t = t * (model.n_timesteps // model.n_sample_timesteps)
            else:
                sample_t = (t+1) * (model.n_timesteps // model.n_sample_timesteps) -1
            
            y, grad = guide.gradients(x, cond, sample_t)
            # 그래디언트를 현재 디바이스로 이동
            grad = grad.to(device)
        
        if scale_grad_by_std and not is_flowmatching:
            # diffusion에서만 variance로 스케일링
            grad = model_var * grad
        
        if is_flowmatching:
            theta_min = getattr(model, "theta_min", 0.0)   # 없는 경우 0으로
            sample_t_normalized = sample_t / model.n_timesteps
            sample_t_normalized = sample_t_normalized.view(-1, 1, 1).to(device)
            b_t     = (1.0 - (1.0 - theta_min) * sample_t_normalized) / (sample_t_normalized + 1e-8)
            grad    = b_t * grad                 #   −β b_t ∇E

        # t < t_stopgrad 인 경우 gradient 업데이트를 중단
        grad[t < t_stopgrad] = 0
        
        x = x + scale * grad
        x = apply_conditioning(x, cond, model.action_dim)
    
    if is_flowmatching:
        # Flowmatching: 직접 결정론적 업데이트 반환
        x_updated = model.p_mean_variance(x=x, cond=cond, t=t)
        return x_updated, y
    else:
        # Diffusion: 평균과 표준편차로 노이즈 추가
        model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
        
        # no noise when t == 0
        noise = torch.randn_like(x)
        noise[t == 0] = 0
        
        return model_mean + model_std * noise, y
