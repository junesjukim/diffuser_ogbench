#!/usr/bin/env python3
import os
import json
import csv
import math
import statistics
import argparse

def read_score(file_path):
    """
    Attempts to read the JSON file at file_path and return the 'score' field.
    If the file does not exist or the field is missing, returns None.
    """
    if not os.path.exists(file_path):
        print(f"[Warning] File does not exist: {file_path}")
        return None

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            score = data.get("score")
            if score is None:
                print(f"[Warning] 'score' not found in {file_path}")
            return score
    except Exception as e:
        print(f"[Error] Could not read {file_path}: {e}")
        return None

def format_stat(avg, std_err):
    """
    Format average and standard error as 'XX.X±Y.Y'
    Values are multiplied by 100 before formatting
    """
    if avg is None or std_err is None:
        return "N/A"
    avg_scaled = avg * 100
    std_err_scaled = std_err * 100
    return f"{avg_scaled:.1f}±{std_err_scaled:.1f}"

def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='결과를 CSV 파일로 변환합니다.')
    parser.add_argument('--output', type=str, default="results.csv", help='출력 CSV 파일 이름')
    parser.add_argument('--prefix', type=str, default="", help='출력 CSV 파일 이름 앞에 붙일 접두사')
    args = parser.parse_args()
    # Set environments
    env1 = "hopper-medium-replay"
    env2 = "walker2d-medium-replay"
    
    mode = "flowmatching"
    # 출력 파일 이름 설정
    output_csv = args.output
    if args.prefix:
        output_csv = f"{mode}_{args.prefix}_{output_csv}"
    
    # Define the categories with their corresponding file path templates
    categories = {
        # Diffuser Plan Epoch Walk
        "diffuser_epoch0": f"logs/walker2d-medium-replay-v2/diffusion_plan/epoch_walk/stepfixed2_H4_T20_S{{seed}}_d0.99_ST20/0/rollout.json",
        "diffuser_epoch200k": f"logs/walker2d-medium-replay-v2/diffusion_plan/epoch_walk/stepfixed2_H4_T20_S{{seed}}_d0.99_ST20/200000/rollout.json",
        "diffuser_epoch400k": f"logs/walker2d-medium-replay-v2/diffusion_plan/epoch_walk/stepfixed2_H4_T20_S{{seed}}_d0.99_ST20/400000/rollout.json",
        "diffuser_epoch600k": f"logs/walker2d-medium-replay-v2/diffusion_plan/epoch_walk/stepfixed2_H4_T20_S{{seed}}_d0.99_ST20/600000/rollout.json",
        
        # Flowmatching Plan Epoch Walk
        "flowmatching_epoch0": f"logs/walker2d-medium-replay-v2/flowmatching_plan/epoch_walk/stepfixed2_H4_T20_S{{seed}}_d0.99_ST20/0/rollout.json",
        "flowmatching_epoch200k": f"logs/walker2d-medium-replay-v2/flowmatching_plan/epoch_walk/stepfixed2_H4_T20_S{{seed}}_d0.99_ST20/200000/rollout.json",
        "flowmatching_epoch400k": f"logs/walker2d-medium-replay-v2/flowmatching_plan/epoch_walk/stepfixed2_H4_T20_S{{seed}}_d0.99_ST20/400000/rollout.json",
        "flowmatching_epoch600k": f"logs/walker2d-medium-replay-v2/flowmatching_plan/epoch_walk/stepfixed2_H4_T20_S{{seed}}_d0.99_ST20/600000/rollout.json",
    }
    # Dictionary to hold results per seed
    results = {}

    # Loop over seed numbers
    for seed in range(150):
        results[seed] = {}
        for cat_name, path_template in categories.items():
            file_path = path_template.format(seed=seed)
            score = read_score(file_path)
            results[seed][cat_name] = score

    # Compute overall averages and standard errors for each category
    averages = {}
    std_errors = {}
    for cat in categories.keys():
        valid_scores = [results[seed][cat] for seed in range(150) if results[seed][cat] is not None]
        if valid_scores:
            avg = sum(valid_scores) / len(valid_scores)
            averages[cat] = avg
            if len(valid_scores) > 1:
                stdev = statistics.stdev(valid_scores)
                std_errors[cat] = stdev / math.sqrt(len(valid_scores))
            else:
                std_errors[cat] = 0.0
        else:
            averages[cat] = None
            std_errors[cat] = None

    # Write the results to a CSV file
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header row
        header = ["seed"] + list(categories.keys())
        writer.writerow(header)
        
        # Write one row per seed
        for seed in range(150):
            row = [seed] + [results[seed][cat] for cat in categories.keys()]
            writer.writerow(row)
        
        # Write averages
        avg_row = ["Average"] + [averages[cat] for cat in categories.keys()]
        writer.writerow(avg_row)
        
        # Write standard errors
        std_err_row = ["StdError"] + [std_errors[cat] for cat in categories.keys()]
        writer.writerow(std_err_row)
        
        # Write formatted statistics (XX.X±Y.Y)
        formatted_stats = ["Formatted"] + [format_stat(averages[cat], std_errors[cat]) for cat in categories.keys()]
        writer.writerow(formatted_stats)
    
    print(f"CSV 파일 '{output_csv}'이(가) 성공적으로 작성되었습니다.")

if __name__ == '__main__':
    main()
