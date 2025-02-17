

from glob import glob

configfile: "/home/users/a/algren/work/diffusion/workflow/config.yaml"
container: config["container_path"]
save_path = '/srv/beegfs/scratch/groups/rodem/VIPR/online/VIPR/'

model_name = 'PUPPIML_top_jets_pileup_jet_2024_12_09_22_33_30_673870'
# model_name = 'PUPPIML_top_jets_w_drop_pileup_jet_2025_01_13_09_57_05_749162'


# hyperparameters
probs_cuts = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75, 0.8, 0.825, 0.85]
# probs_cuts = [0.3]
if False:
    mu = [200]
    std = [50]
    size = 99990
    subtraction=10
else:
    mu = [50, 60, 70, 80, 90, 100, 150, 200, 250, 300]
    std = [0]
    size = 9999
    subtraction=1

str_probs_cuts = [str(i).replace(".", "_") for i in probs_cuts]

def convert_to_float(string):
    return float(string.replace("_", "."))

# Plotting rules
rule all:
    input:
        expand(
            [
            '{save_path}/{model_name}/eval_files/gen_jets_single_pileup_mu_{mu}_std_{std}_size_{size}_probs_cut_{str_probs_cut}.csv',
            '{save_path}/{model_name}/eval_files/jet_subs/jet_substructure_single_pileup_mu_{mu}_std_{std}_size_{size}_probs_cut_{str_probs_cut}.h5'
            ], 
        str_probs_cut=str_probs_cuts, 
        save_path=save_path, 
        model_name=model_name,
        mu=mu, std=std, size=size
        )

rule puppi_run_eval:
    output:
        '{save_path}/{model_name}/eval_files/gen_jets_single_pileup_mu_{mu}_std_{std}_size_{size}_probs_cut_{str_probs_cut}.csv'
    params:
        probs_cut = lambda wildcards: convert_to_float(wildcards.str_probs_cut),
        _size = lambda wildcards: int(wildcards.size)+subtraction
    resources:
        gpu=1,
        slurm_extra="--gres=gpu:1,VramPerGpu:20G --exclude=",
        slurm_partition="shared-gpu,private-dpnc-gpu",
    shell:
        # pip install dotmap --user &&
        """
        python /home/users/a/algren/work/github/VIPR/run/run_eval.py eval_clf.probs_cut={params.probs_cut} eval_clf.path_to_model={wildcards.save_path}/{wildcards.model_name} generate_substructure=False data_cfg.pileup_dist_args.mu={wildcards.mu} data_cfg.pileup_dist_args.std={wildcards.std} predict_str=eval_clf size={params._size}
        """

rule puppi_generate_substructure:
    input:
        '{save_path}/{model_name}/eval_files/gen_jets_single_pileup_mu_{mu}_std_{std}_size_{size}_probs_cut_{str_probs_cut}.csv'
    output:
        '{save_path}/{model_name}/eval_files/jet_subs/jet_substructure_single_pileup_mu_{mu}_std_{std}_size_{size}_probs_cut_{str_probs_cut}.h5'
    params:
        probs_cut = lambda wildcards: convert_to_float(wildcards.str_probs_cut),
        _size = lambda wildcards: int(wildcards.size)+subtraction
    shell:
        # pip install tables --user &&
        """
        python /home/users/a/algren/work/github/VIPR/run/run_eval.py eval_clf.probs_cut={params.probs_cut} eval_clf.path_to_model={wildcards.save_path}/{wildcards.model_name} generate_substructure=True data_cfg.pileup_dist_args.mu={wildcards.mu} data_cfg.pileup_dist_args.std={wildcards.std} predict_str=eval_clf size={params._size}
        """
