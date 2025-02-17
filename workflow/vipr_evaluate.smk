

from glob import glob

configfile: "/home/users/a/algren/work/diffusion/workflow/config.yaml"
container: config["container_path"]

###### setting ######

# model_name = 'PUPPIML_top_jets_pileup_jet_2024_12_07_13_28_41_399052'
model_name = 'VIPR_top_jets_w_drop_pileup_jet_2025_01_10_15_40_18_574439'
flow_path = 'p_N_flow_w_10_drop_pileup_jet_2025_01_22_14_15_16_223112' # N(200, 10) correct normalization dp proba

save_sub_path = [f'eval_files/flow_N/{flow_path}']


save_path = f'/srv/beegfs/scratch/groups/rodem/VIPR/online/VIPR/'

if False:
    mu = [200]
    std = [50]
    size = 99990
else:
    mu = [50, 60, 70, 80, 90, 100, 150, 200, 250, 300]
    std = [0]
    size = 9999

# Plotting rules
rule all:
    input:
        expand(
            [
            '{save_path}/{model_name}/{save_sub_path}/gen_cnts_single_pileup_mu_{mu}_std_{std}_size_{size}.csv',
            '{save_path}/{model_name}/{save_sub_path}/jet_subs/jet_substructure_single_pileup_mu_{mu}_std_{std}_size_{size}.h5',
            ], 
        save_path=save_path, 
        model_name=model_name,
        mu=mu, std=std, size=size,
        save_sub_path = save_sub_path
        )

rule VIPR_run_eval:
    output:
        '{save_path}/{model_name}/{save_sub_path}/gen_cnts_single_pileup_mu_{mu}_std_{std}_size_{size}.csv',
    resources:
        gpu=1,
        slurm_extra="--gres=gpu:1,VramPerGpu:20G --exclude= --constraint=COMPUTE_TYPE_AMPERE",
    shell:
        # pip install nflows --user &&
        """
        python /home/users/a/algren/work/github/VIPR/run/run_eval.py generate_substructure=False data_cfg.pileup_dist_args.mu={wildcards.mu} data_cfg.pileup_dist_args.std={wildcards.std} predict_str=eval_VIPR_dp
        """

rule VIPR_generate_substructure:
    input:
        '{save_path}/{model_name}/{save_sub_path}/gen_cnts_single_pileup_mu_{mu}_std_{std}_size_{size}.csv'
    output:
        '{save_path}/{model_name}/{save_sub_path}/jet_subs/jet_substructure_single_pileup_mu_{mu}_std_{std}_size_{size}.h5'
    # params:
    #     probs_cut = lambda wildcards: convert_to_float(wildcards.str_probs_cut),
    resources:
        gpu=1,
        slurm_extra="--gres=gpu:1,VramPerGpu:2G --exclude=",
    shell:
        # pip install tables --user &&
        """
        python /home/users/a/algren/work/github/VIPR/run/run_eval.py generate_substructure=True data_cfg.pileup_dist_args.mu={wildcards.mu} data_cfg.pileup_dist_args.std={wildcards.std} predict_str=eval_VIPR_dp
        """
