import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import cv2
import subprocess
import time
import imageio
import ogbench

# diffuser_ogbench 경로를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import diffuser.utils as utils
from diffuser.datasets.ogbench import OGBenchGoalDataset

class Parser(utils.Parser):
    dataset: str = 'scene-play-singletask-task2-v0'
    config: str = 'config.locomotion'
    horizon: int = 64
    normalizer: str = 'LimitsNormalizer'
    use_padding: bool = True
    max_path_length: int = 1000

def get_episode_indices(terminals, episode_idx):
    """terminals 배열에서 원하는 에피소드의 (start, end) 인덱스 반환"""
    ends = np.where(terminals)[0]
    if episode_idx == 0:
        start = 0
    else:
        start = ends[episode_idx - 1] + 1
    end = ends[episode_idx] + 1  # end는 exclusive
    return start, end

def render_episode(episode_idx, dataset, env, output_dir='rendered_episodes'):
    """특정 에피소드를 렌더링하고 비디오로 저장하는 함수"""
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 에피소드 구간 계산
    terminals = dataset['terminals']
    start, end = get_episode_indices(terminals, episode_idx)
    print(f"에피소드 {episode_idx}의 시작 인덱스: {start}, 종료 인덱스: {end}")
    observations = dataset['observations'][start:end]
    actions = dataset['actions'][start:end]
    
    # 비디오 작성자 초기화
    video_writer = imageio.get_writer(
        os.path.join(output_dir, f'episode_{episode_idx}.mp4'),
        fps=30
    )
    
    # 환경 초기화
    obs, _ = env.reset()
    
    # 에피소드 재생
    print(f"에피소드 {episode_idx}의 타임스텝 수: {actions.shape[0]}")
    for t in range(actions.shape[0]):
        action = actions[t]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 현재 프레임 렌더링
        frame = env.render()
        
        # OpenCV를 사용하여 타임스텝 정보 추가 (RGB -> BGR 변환 필요)
        frame_cv = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 텍스트 설정
        text = f"t: {t}"
        print(f"텍스트: {text}")
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 255, 255)  # 흰색
        font_thickness = 2
        
        # 텍스트 위치 (우측 상단)
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = frame.shape[1] - text_size[0] - 10  # 우측 여백 10픽셀
        text_y = text_size[1] + 10  # 상단 여백 10픽셀
        
        # 텍스트 배경 (가독성 향상)
        cv2.rectangle(frame_cv, 
                     (text_x - 5, text_y - text_size[1] - 5), 
                     (text_x + text_size[0] + 5, text_y + 5), 
                     (0, 0, 0), -1)
        
        # 텍스트 추가
        cv2.putText(frame_cv, text, (text_x, text_y), font, font_scale, 
                   font_color, font_thickness, cv2.LINE_AA)
        
        # BGR -> RGB 변환하여 비디오에 추가
        frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        video_writer.append_data(frame_rgb)
        
        if terminated or truncated:
            print(f"에피소드 {episode_idx} 종료: terminated={terminated}, truncated={truncated}")
            break
    
    # 비디오 작성자 종료
    video_writer.close()
    
    # 환경 종료
    env.close()
    print(f"에피소드 {episode_idx}의 비디오가 {output_dir}에 저장되었습니다.")

def main():
    # Xvfb를 사용하여 가상 디스플레이 설정
    subprocess.run(['pkill', 'Xvfb'])
    subprocess.Popen(['Xvfb', ':100', '-screen', '0', '1024x768x24', '-ac'])
    os.environ['DISPLAY'] = ':100.0'
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['EGL_DEVICE_ID'] = '0'
    time.sleep(2)  # Xvfb가 시작될 때까지 대기
    
    # 파서로 설정 가져오기
    args = Parser().parse_args('diffusion')
    
    # 환경과 데이터셋 생성
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        args.dataset,
        render_mode='rgb_array',
        width=640,
        height=480,
        max_episode_steps=1000
    )
    
    # 환경의 최대 에피소드 길이 설정 확인 및 변경
    print(f"원래 max_episode_steps: {env._max_episode_steps}")
    env._max_episode_steps = 1000
    print(f"변경된 max_episode_steps: {env._max_episode_steps}")
    
    #데이터셋 구조 출력
    print('train_dataset keys:', train_dataset.keys())
    for k in train_dataset:
        print(f"{k}: {type(train_dataset[k])}, shape: {getattr(train_dataset[k], 'shape', None)}")
    #10개의 에피소드 렌더링
    
    for i in range(10):
        print(f"\n렌더링 중: 에피소드 {i}")
        render_episode(i, train_dataset, env)
    
    # Xvfb 프로세스 종료
    subprocess.run(['pkill', 'Xvfb'])

if __name__ == "__main__":
    main() 