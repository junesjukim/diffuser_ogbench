import ogbench

dataset_names = [
    'scene-play-v0'
]

# 데이터셋 다운로드
ogbench.download_datasets(
    dataset_names,  # 데이터셋 이름 리스트
    dataset_dir='/home/junseolee/intern/diffuser_ogbench/iql-pytorch/data/ogbench',  # 데이터셋 저장 디렉토리
) 