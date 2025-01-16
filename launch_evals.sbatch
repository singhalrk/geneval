#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=evals
#SBATCH --partition=a100_short,a100_long
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --array=0-200
#SBATCH --gres=gpu:1

# Initialize Conda for bash
eval "$(conda shell.bash hook)"

# Activate the geneval environment
conda activate geneval

python collect_results.py --job_index $SLURM_ARRAY_TASK_ID

# # Run the evaluation script as a loop conditioned on the array index

# python evaluation/evaluate_images.py  /gpfs/data/ranganathlab/singhr36/twisty-diffusion/diffusers_experimentation/geneval_outputs/20241017-114415/   --outfile "results_folder/results_0.jsonl"     --model-path object_detector
# python evaluation/summary_scores.py results_folder/results_0.jsonl 


# # python evaluation/evaluate_images.py  /gpfs/data/ranganathlab/singhr36/twisty-diffusion/diffusers_experimentation/geneval_outputs/20241017-114418/   --outfile "results_folder/results_1.jsonl"     --model-path object_detector
# # python evaluation/summary_scores.py results_folder/results_1.jsonl 

# # python evaluation/evaluate_images.py  /gpfs/data/ranganathlab/singhr36/twisty-diffusion/diffusers_experimentation/geneval_outputs/20241017-114419/   --outfile "results_folder/results_2.jsonl"     --model-path object_detector

# # python evaluation/summary_scores.py results_folder/results_2.jsonl
# # python evaluation/evaluate_images.py  /gpfs/data/ranganathlab/singhr36/twisty-diffusion/diffusers_experimentation/geneval_outputs/20241017-114435/   --outfile "results_folder/results_3.jsonl"     --model-path object_detector
# # python evaluation/evaluate_images.py  /gpfs/data/ranganathlab/singhr36/twisty-diffusion/diffusers_experimentation/geneval_outputs/20241017-114443/   --outfile "results_folder/results_4.jsonl"     --model-path object_detector
# # python evaluation/evaluate_images.py  /gpfs/data/ranganathlab/singhr36/twisty-diffusion/diffusers_experimentation/geneval_outputs/20241017-115641/   --outfile "results_folder/results_5.jsonl"     --model-path object_detector
# # python evaluation/evaluate_images.py  /gpfs/data/ranganathlab/singhr36/twisty-diffusion/diffusers_experimentation/geneval_outputs/20241017-125819/   --outfile "results_folder/results_6.jsonl"     --model-path object_detector
# # python evaluation/evaluate_images.py  /gpfs/data/ranganathlab/singhr36/twisty-diffusion/diffusers_experimentation/geneval_outputs/20241017-125822/   --outfile "results_folder/results_7.jsonl"     --model-path object_detector
# # python evaluation/evaluate_images.py  /gpfs/data/ranganathlab/singhr36/twisty-diffusion/diffusers_experimentation/geneval_outputs/20241017-125828/   --outfile "results_folder/results_8.jsonl"     --model-path object_detector
# # python evaluation/evaluate_images.py  /gpfs/data/ranganathlab/singhr36/twisty-diffusion/diffusers_experimentation/geneval_outputs/20241017-125831/   --outfile "results_folder/results_9.jsonl"     --model-path object_detector
# # python evaluation/evaluate_images.py  /gpfs/data/ranganathlab/singhr36/twisty-diffusion/diffusers_experimentation/geneval_outputs/20241017-125843/   --outfile "results_folder/results_10.jsonl"    --model-path object_detector
# # python evaluation/evaluate_images.py  /gpfs/data/ranganathlab/singhr36/twisty-diffusion/diffusers_experimentation/geneval_outputs/20241017-125847/   --outfile "results_folder/results_11.jsonl"    --model-path object_detector

# # python evaluation/summary_scores.py results_folder/results_3.jsonl
# # python evaluation/summary_scores.py results_folder/results_4.jsonl
# # python evaluation/summary_scores.py results_folder/results_5.jsonl
# # python evaluation/summary_scores.py results_folder/results_6.jsonl
# # python evaluation/summary_scores.py results_folder/results_7.jsonl
# # python evaluation/summary_scores.py results_folder/results_8.jsonl
# # python evaluation/summary_scores.py results_folder/results_9.jsonl
# # python evaluation/summary_scores.py results_folder/results_10.jsonl
# # python evaluation/summary_scores.py results_folder/results_11.jsonl
