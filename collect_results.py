import argparse
import json
from pathlib import Path
import os
import pandas as pd
import numpy as np


def main(args):
    folder = Path("/gpfs/data/ranganathlab/singhr36/twisty-diffusion/diffusers_experimentation/geneval_metadata_outputsl/")

    # folder = Path("./results_folder")
    data = []
    # a, b = "20241028-122131", "20241028-122130"
    
    # "20241111"
    print(len(list(folder.iterdir())))
    
    # arr_evals = ["20241111-120127", "20241111-121048", "20241111-121108", "20241111-132706"]
        
    for i, job in enumerate(folder.iterdir()):
        # `for i, job in enumerate([folder / a, folder / b]):     
        
        # if i not in [27, 28, 29, 30]:
        #     continue

        if i != args.job_index:
            continue
         
        # # job = folder / arr_evals[i-27]       
        # job = folder / "20241115-135123"
        # i = 102     
        
        print(job)
                
        # # print(f"Launching job {i}")
        # # print(job)
        # df = pd.read_json(results_filename, orient="records", lines=True)
        
        # job = Path(df.filename.iloc[0]).parent.parent.parent
        args = job / "args.json"
    
        if not args.exists():
            print(f"Args file {args} does not exist")
            continue

        # load and print args
        with open(args, "r") as f:
            args = json.load(f)
        print(args)    
        
        print(job, i)

        print(f"Running job {i}, {job}")
        os.system(f"python evaluation/evaluate_images.py {job} --outfile results_folder/results_{i}.jsonl --model-path object_detector")
        os.system(f"python evaluation/summary_scores.py results_folder/results_{i}.jsonl")
        print(f"Job {i} done")

        ave_score = generate_data(f"results_folder/results_{i}", bon=False)
        bon_score = generate_data(f"results_folder/results_{i}", bon=True)
        print(f"Job {i} done")   

        data.append(
            dict(model_name=args['model_name'],
            use_smc=args['use_smc'],
            lmbda=args['lmbda'],
            resample_freq=args['resample_frequency'],
            t_start=args['resample_t_start'],
            t_end=args['resample_t_end'],
            ave_score=ave_score,
            bon_score=bon_score)
        )

    data = pd.DataFrame(data)
    data = data.sort_values(by='ave_score', ascending=False)
    print(data)

    data.to_csv("new_geneval_scores.csv", index=False)


def generate_data(filename, bon=False):
    with open("./evaluation/object_names.txt") as cls_file:
        classnames = [line.strip() for line in cls_file]
        cls_to_idx = {"_".join(cls.split()):idx for idx, cls in enumerate(classnames)}

    # Load results

    df = pd.read_json(filename, orient="records", lines=True)

    # Measure overall success

    # print("Summary")
    # print("=======")
    print(f"Total images: {len(df)}")
    print(f"Total prompts: {len(df.groupby('metadata'))}")

    assert len(df.groupby('metadata')) == 553, "Number of prompts is not 553"
    # print(f"% correct images: {df['correct'].mean():.2%}")
    # print(f"% correct prompts: {df.groupby('metadata')['correct'].any().mean():.2%}")
    # print()

    # By group
    task_scores = []

    # filter df by filename ending in 0000.png
    if bon:
        df = df[df['filename'].str.endswith("0000.png")]

    # print("Task breakdown")
    # print("==============")
    for tag, task_df in df.groupby('tag', sort=False):
        task_scores.append(task_df['correct'].mean())
        # print(f"{tag:<16} = {task_df['correct'].mean():.2%} ({task_df['correct'].sum()} / {len(task_df)})")

    # print(f"Overall score (avg. over tasks): {np.mean(task_scores):.5f}")    

    return np.mean(task_scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch a job")
    parser.add_argument("--job_index", type=int, help="Job to launch")
    
    args = parser.parse_args()
    main(args)