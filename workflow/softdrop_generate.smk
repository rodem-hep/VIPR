

from glob import glob

configfile: "/home/users/a/algren/work/diffusion/workflow/config.yaml"
container: config["container_path"]

path_to_obs_jet = glob('/srv/beegfs/scratch/groups/rodem/pileup_diffusion/data/obs_jets/*.npy')[:None]
save_path = '/srv/beegfs/scratch/groups/rodem/pileup_diffusion/data/softdrop'

# hyperparameters
beta = [0, 0.5, 1, 2, 3]
zcut= [0.05, 0.1, 0.15]

out_sp_eval={}
out_sp_hlv={}
input_sp_hlv = {}

for input_file in path_to_obs_jet:
    if 'pileup_mu_100_std_0' not in input_file:
        continue
    for b in beta:
        for z in zcut:
            z_str = str(z).replace('.','_')
            b_str = str(b).replace('.','_')

            # generate SP file
            name = input_file.split('/')[-1]
            sp_out_file = f"{save_path}/zcut_{z_str}_beta_{b_str}/{name}"
            out_sp_eval[f"{b},{z},{input_file}"] = sp_out_file

            # generate HLV file from SP file
            sp_out_file_hlv = f"{save_path}/zcut_{z_str}_beta_{b_str}/{name}"
            sp_out_file_hlv = sp_out_file_hlv.replace('.npy', '_HLV.h5')

            # output file for hlv
            out_sp_hlv[f"{sp_out_file}"] = sp_out_file_hlv

            # input file for HLV
            input_sp_hlv[f"{sp_out_file}"] = sp_out_file

print(f'out_sp_eval: {out_sp_eval}')
print(f'out_sp_eval: {out_sp_hlv}')

# Plotting rules
rule all:
    input:
        list(out_sp_eval.values())+list(out_sp_hlv.values())

# 
for i,j in out_sp_eval.items():
    args = i.split(',')
    rule:
        name: i
        params:
            beta=args[0],
            zcut=args[1],
            input_file=args[2],
        output:
            j
        shell:
            # pip install fastjet --user &&
            # pip install energyflow --user &&
            # pip install -e /home/users/a/algren/work/tools/. --user &&
            f"""
            python /home/users/a/algren/work/diffusion/run/RunSoftdrop.py --input={{params.input_file}} --output={{output}} --beta={{params.beta}} --zcut={{params.zcut}}"""
    
    # HLV 
    input_file = input_sp_hlv[j]
    output_file = out_sp_hlv[j]
    rule:
        name: f'{i}_HLV'
        input: input_file
        output: output_file
        shell:
            # pip install -e /home/users/a/algren/work/tools/. --user &&
            # pip install fastjet --user &&
            # pip install energyflow --user &&
            # pip install pyjet --user &&
            # pip install tables --user &&
            f"""
            python /home/users/a/algren/work/diffusion/run/LoadJetDumpSubstructure.py --input={{input}} --output={{output}}"""
