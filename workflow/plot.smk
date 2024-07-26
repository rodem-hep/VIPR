# FILEPATH: /home/users/a/algren/work/diffusion/workflow/plot.smk

# default configs
save_figures=True

from tools import misc
config = misc.load_yaml("configs/evaluate.yaml")

# get the save path
save_path = f"{config.eval.path_to_model}/figures"

save_format = config.eval.format

# defining output list
out_plot_scatter = [f"{save_path}/{config.csv_sample_to_load}/imgs/scatter_images_{i}_{idx}.pdf" for idx in range(2) for i in ["obs", "sd", "vipr"]]

out_plot_subs = [f"{save_path}/{config.csv_sample_to_load}/gen_jets_{i}_{save_format}" for i in config.jet_sub_vars]
out_plot_subs += [f"{save_path}/{config.csv_sample_to_load}/pileup_func/{i}_median.pdf" for i in config.jet_sub_vars]
out_plot_subs += [f"{save_path}/{config.csv_sample_to_load}/pileup_func/{i}_IQR.pdf" for i in config.jet_sub_vars]

out_plot_posterior = [f"{save_path}/posteriors/posterior_quantiles_{label}.pdf" for label in config.jet_sub_vars]

# Plotting rules
rule all:
    input:
        out_plot_scatter+out_plot_subs+out_plot_posterior

# Rule for plotting single sampling
rule plot_scatter:
    output:
        out_plot_scatter
    shell:
        f"python /home/users/a/algren/work/diffusion/run/plot_eval.py save_figures={save_figures} plot_images=True"

# Rule for plotting subs
rule plot_subs:
    output:
        out_plot_subs
    shell:
        f"""python /home/users/a/algren/work/diffusion/run/plot_substructure.py \
        save_figures={save_figures} eval.size=100000 data_cfg.pileup_dist_args.mu=200 \
        data_cfg.pileup_dist_args.mu=50
        """

# Rule for plotting subs
rule plot_posteriors:
    output:
        out_plot_posterior
    shell:
        f"""python /home/users/a/algren/work/diffusion/run/plot_posterior.py \
        save_figures={save_figures} eval.size=2000 data_cfg.pileup_dist_args.mu=200 \
        data_cfg.pileup_dist_args.mu=0
        """
