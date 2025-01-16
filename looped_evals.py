import argparse
import json
from pathlib import Path
import os
import pandas as pd
import numpy as np


def main(args):
    # folder = Path("/gpfs/data/ranganathlab/singhr36/twisty-diffusion/diffusers_experimentation/geneval_outputs/")

    folder = Path("./results_folder")
    data = []
    for i, results_filename in enumerate(folder.rglob("*.jsonl")):     

        # print(results_filename.name)
        # job_id = int(results_filename.name.split("_")[1].split(".")[0])
        # if job_id < 24:
        #     continue        

        # assert False, "Stop here"
        
        # print(f"Launching job {i}")
        print(results_filename)
        try:
            df = pd.read_json(results_filename, orient="records", lines=True)
        except AttributeError:
            print(f"File {results_filename} does not exist")
            continue
        
        try:            
            job = Path(df.filename.iloc[0]).parent.parent.parent
        except AttributeError:
            print(f"File {results_filename} does not exist")
            continue
        
        args = job / "args.json"    
        # add extra line
        print(job)
        
        if not args.exists():
            print(f"Args file {args} does not exist")
            continue

        # load and print args
        with open(args, "r") as f:
            args = json.load(f)
        # print(args)


        div_score = 0
        IR_avg = 0
        IR_max = 0
        
        n_samples = 0
        n_prompts = 0

        for prompt_samples in job.iterdir():
            results_file = prompt_samples / 'results.json'
            if results_file.exists():
                # print(f"Results file {results_file} exists")
                with open(results_file, "r") as f:
                    results = json.load(f)
                    # print(results)
                    IR_max += results['ImageReward']['max']
                    IR_avg += results['ImageReward']['mean']

                    if 'Clip-Score' in results:
                        
                        div_score += results['Clip-Score']['diversity'] if 'diversity' in results['Clip-Score'] else 0
                    else:
                        div_score += 0
                        
                    n_samples += len(results['ImageReward']['result'])
                    n_prompts += 1
        if n_prompts < 500:
            continue

        IR_max /= n_prompts
        IR_avg /= n_samples
        div_score /= n_prompts

        ave_score = generate_data(results_filename, bon=False)
        bon_score = generate_data(results_filename, bon=True)
        print(f"Job {i} done")   
        
        if n_prompts < 550:
            continue

        data.append(
            dict(model_name=args['model_name'],
            use_smc=args['use_smc'],
            lmbda=args['lmbda'],
            resample_freq=args['resample_frequency'],
            t_start=args['resample_t_start'],
            t_end=args['resample_t_end'],
            num_particles=args['num_particles'],
            potential_type=args['potential_type'],
            seed=args['seed'],
            div_score=div_score,
            IR_avg=IR_avg,
            IR_max=IR_max,
            ave_score=ave_score,            
            bon_score=bon_score,
            job=job.name,
            results_filename=results_filename.name,
        ))

    data = pd.DataFrame(data)
    data = data.sort_values(by='ave_score', ascending=False)
        
    return data


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
    parser.add_argument("--job_index", required=False, default=0, type=int, help="Job to launch")
    
    args = parser.parse_args()
    # data = main(args)
    
    # # print(data)
    # # for model_name in data.model_name.unique():
    # #     print(f"Model: {model_name}")
    # #     print(data[data.model_name == model_name].sort_values(by='ave_score', ascending=False))                
    

    # data.to_csv("geneval_scores.csv", index=False)
    # data = pd.read_csv("new_potentials_geneval.csv")
    data = pd.read_csv("geneval_scores.csv")
    # 
    
    # data = pd.read_csv("new_potentials_geneval.csv")
    # # change column loc of potential type to end
    cols = list(data.columns)
    cols.remove("potential_type")
    cols.append("potential_type")
    data = data[cols]
    
    # print(data)
    # for model_name in data.model_name.unique():
    #     print(f"Model: {model_name}")
    #     print(data[data.model_name == model_name].sort_values(by='ave_score', ascending=False))                
    
    # data.to_csv("geneval_scores.csv", index=False)

    # data = pd.read_csv("geneval_scores.csv")
    del data['IR_max']
    del data['IR_avg']
    del data['results_filename']
    del data['t_start']
    del data['t_end']
    
    model_name_change = {
        "stabilityai/stable-diffusion-xl-base-1.0": 'SDXL-base',
        "stabilityai/stable-diffusion-2-1": 'SDv2.1',
        "runwayml/stable-diffusion-v1-5": 'SDv1.5',
        "CompVis/stable-diffusion-v1-4": 'SDv1.4',
        "mhdang/dpo-sdxl-text2image-v1": 'SDXL-base-DPO',
        "mhdang/dpo-sd1.5-text2image-v1": 'SDv1.5-DPO',
    }
    
    
    # apply name change to data
    data['model_name'] = data['model_name'].apply(lambda x: model_name_change[x])
    
    # data.to_csv("consistent_allk_new_potentials_geneval.csv", index=False)

    print(data[data.use_smc == False])
    data = data[data.use_smc == False]
    for model_name in data.model_name.unique():
        print(f"Model: {model_name}")
        print(data[data.model_name == model_name].sort_values(by='ave_score', ascending=False))                    

    # print(data[data.num_particles == 2])
    # print(data[data.num_particles == 3][data.model_name == "SDv2.1"][data.potential_type == 'max'])
    # data[data.num_particles == 2].to_csv("2_particles.csv", index=False)