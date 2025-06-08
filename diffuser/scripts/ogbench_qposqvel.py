import os, sys, inspect
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import ogbench

###############################################################################
# 1) 환경 + 데이터셋을 “스테이트 포함” 으로 로드
###############################################################################

ENV_NAME = "scene-play-singletask-task2-v0"

# 현재 ogbench 버전에 따라 인자 이름이 조금 다릅니다.
# 가장 쉬운 방법: 서명 확인 후 kwargs 동적으로 전달
sig = inspect.signature(ogbench.make_env_and_datasets)
kw = dict(render_mode="rgb_array")
for k in ("include_state", "include_infos", "with_state", "with_infos", "return_state"):
    if k in sig.parameters:
        kw[k] = True

env, train_ds, _ = ogbench.make_env_and_datasets(ENV_NAME, **kw)
print("[ OK ] dataset keys:", train_ds.keys())

###############################################################################
# 2) 'infos/qpos', 'infos/qvel', 'infos/button_states' 존재 확인
###############################################################################

def get_state_arrays(dataset, t):
    """
    t 번째 transition(=데이터셋 인덱스)의 qpos/qvel/button_states 반환
    -> 없으면 None
    """
    for key in ("infos/qpos", "info/qpos", "qpos"):
        if key in dataset:
            return dataset[key][t], dataset[key.replace("qpos", "qvel")][t], \
                   dataset.get(key.replace("qpos", "button_states"), None)
    return None, None, None

qpos, qvel, buttons = get_state_arrays(train_ds, 0)
if qpos is None:
    raise RuntimeError(
        "데이터셋에 qpos/qvel 이 없습니다.\n"
        " - ogbench==0.3.0 이상이라면 make_env_and_datasets(..., include_state=True)\n"
        " - 구버전이라면 datasets/scene-play-singletask-task2-v0-state-*.npz 를 받아야 합니다."
    )

print("qpos shape:", qpos.shape, "| qvel shape:", qvel.shape)

###############################################################################
# 3) MuJoCo state 세팅 & 렌더
###############################################################################

# (i) 아무 state 로 리셋
env.reset(seed=0)
sim = env.unwrapped.sim        # TimeLimit 래퍼 해제
assert sim.model.nq == qpos.size and sim.model.nv == qvel.size

# (ii) 버튼(락) 상태까지 포함한 새로운 MjSimState 생성
state = mujoco.MjSimState(
    time=0.0,
    qpos=np.asarray(qpos, dtype=np.float64).copy(),
    qvel=np.asarray(qvel, dtype=np.float64).copy(),
    act=None,
    udd_state={}
)
sim.set_state(state)

# SceneEnv 전용: 현재 버튼 상태 덮어쓰기
if hasattr(env.unwrapped, "_cur_button_states") and buttons is not None:
    env.unwrapped._cur_button_states = np.asarray(buttons, dtype=np.int8).copy()

sim.forward()          # 반드시 호출

# (iii) 렌더링
frame = env.render(camera_name="corner", width=640, height=640)

###############################################################################
# 4) 확인용 시각화 / 파일 저장
###############################################################################

plt.imshow(frame)
plt.axis("off")
plt.tight_layout()
plt.show()

out_path = "first_frame.png"
plt.imsave(out_path, frame)
print(f"[ SAVED ] {out_path}")
