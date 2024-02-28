
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import os
from glob import glob

import numpy as np
from datetime import datetime
import hydra
import pandas as pd

# framework
import tools.physics.jet_substructure as sjets

# internal
from tools import misc
from tools.datamodule.prepare_data import matrix_to_point_cloud
from plot_eval import get_pileup_name

@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="evaluate")
def main(config):
    eval_fw = hydra.utils.instantiate(config.eval)

    name = get_pileup_name(eval_fw.data.pileup_dist_args)

    # pileup naming
    name=""
    if eval_fw.data.pileup_dist_args != "":
        name = "_pileup_"+"_".join([f"{i}_{j}" for i,j in eval_fw.data.pileup_dist_args.items()])
        print(f"Evaluating pileup with {name}")

    if False: # create data sample of obs jet and regular jet
        eval_ctxt, eval_truth = eval_fw.data.get_normed_ctxt(return_truth=True)
        eval_ctxt["cnts"] = eval_fw.data.relative_pos(eval_ctxt["cnts"],
                                                jet_vars=eval_ctxt["scalars"][:, :3],
                                                mask=eval_ctxt["mask"],
                                                reverse=True)
        eval_truth["images"] = eval_fw.data.relative_pos(eval_truth["images"],
                                                jet_vars=eval_truth["scalars"][:, :3],
                                                mask=eval_truth["mask"],
                                                reverse=True)

        date_str=f"_{datetime.today():%m_%d_%Y_%H_%M_%S}"

        # Write dictionary to json file
        size = len(eval_ctxt["cnts"])
        np.save(f'/srv/beegfs/scratch/groups/rodem/pileup_diffusion/data/obs_jets/obs_jet{name}_size_{size}{date_str}.npy', eval_ctxt)

        # Write dictionary to json file
        np.save(f'/srv/beegfs/scratch/groups/rodem/pileup_diffusion/data/top_jets/top_jet_size_{size}{date_str}.npy', eval_truth)
    elif False: # generate substructure
        size = config.eval.size

        ### generate sample ###
        gen_data = eval_fw.get_eval_files(should_contain=config.csv_sample_to_load)
        
        # get size 
        size = 9999*config.eval.size//10_000
        if "posterior" in config.csv_sample_to_load:
            size = "99990_2000" # TODO fix this

        # full name
        full_name = f"{config.csv_sample_to_load}{name}_size_{size}"

        # truth_jets = gen_data[f"truth_jets_{config.csv_sample_to_load}"]
        gen_cnts = gen_data[f"gen_cnts_{full_name}"]
        gen_jets = gen_data[f"gen_jets_{full_name}"]

        # # create pc
        gen_cnts, mask = matrix_to_point_cloud(gen_cnts[["eta", "phi", "pt"]].values,
                                                gen_cnts["eventNumber"].values,
                                            #   num_per_event_max=max_cnts
                                                )
        if "eventNumber" not in gen_jets.columns:
            gen_jets=gen_jets.rename(columns={gen_jets.columns[0]:"eventNumber"})

        # generated
        print("Generated predicted substructure")
        out_name = f"{config.eval.path_to_model}/eval_files/jet_substructure_{full_name}.h5"
        
        if not os.path.isfile(out_name):
            if "posterior" in config.csv_sample_to_load:
                size = None # TODO fix this
            sjets.dump_hlvs(gen_cnts[:size], mask[:size],
                            out_path=out_name,
                            addi_col={"eventNumber":gen_jets["eventNumber"].values[:size]}
                            )
            if "posterior" in config.csv_sample_to_load:
                import sys
                sys.exit()

        truth_name = f"{config.eval.path_to_model}/eval_files/jet_substructure_truth_{full_name}.h5"

        if not os.path.isfile(truth_name) and not "posterior" in config.csv_sample_to_load:
            # # substruct for true Top
            print("True substructure")
            sjets.dump_hlvs(eval_fw.data.cnts_vars[:None],
                            eval_fw.data.mask_cnts[:None],
                            out_path=f"{config.eval.path_to_model}/eval_files/jet_substructure_truth_{full_name}.h5",
                            # addi_col={"eventNumber":truth_jets["eventNumber"].values[:len(gen_jets)]}
                            )
        
        obs_jet_file_name = f"{config.eval.path_to_model}/eval_files/jet_substructure_ctxt_{full_name}.h5"
        
        if ("single" in config.csv_sample_to_load
            and not os.path.isfile(obs_jet_file_name)): # substruct for obs. jet
            print("Generated obs. jet substructure")

            # get ctxt 
            ctxt_path = glob(f"/srv/beegfs/scratch/groups/rodem/pileup_diffusion/data/data/obs_jet{full_name.replace('single', '')}*.npy")

            eval_ctxt= np.load([i for i in ctxt_path if "soft" not in i][0], allow_pickle=True).item()

            sjets.dump_hlvs(eval_ctxt["cnts"][:len(gen_jets)], eval_ctxt["mask"][:len(gen_jets)],
                            out_path=obs_jet_file_name,
                            # addi_col={"eventNumber":truth_jets["eventNumber"].values[:len(gen_jets)]}
                            )
    else:
        ### generate sample ###
        eval_fw.load_diffusion()

        # will follow the pileup defined
        file_name = glob(f"/srv/beegfs/scratch/groups/rodem/pileup_diffusion/data/obs_jets/obs_jet*{name}*")
        file_name = [i for i in file_name if "softdrop" not in i][0]

        eval_ctxt = np.load(file_name, allow_pickle=True).item()
        
        # abs -> rel
        eval_ctxt["cnts"] = eval_fw.data.relative_pos(
            eval_ctxt["cnts"],
            jet_vars=pd.DataFrame(eval_ctxt["scalars"][:, :3],
                                    columns=["eta", "phi", "pt"]),
            mask=eval_ctxt["mask"],
            reverse=False)

        saving_name = f"{config.csv_sample_to_load}{name}_size_{len(eval_ctxt['cnts'])}"
        print(f"Save path: {saving_name}")

        # it will save the generates pc
        eval_fw.generate_and_save_post(eval_ctxt, config.n_post_to_gen,
                                       combine_ctxt_size=config.combine_ctxt_size,
                                       saving_name=saving_name)


if __name__ == "__main__":
    main()