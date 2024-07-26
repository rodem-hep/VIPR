
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import os
from glob import glob

import numpy as np
from datetime import datetime
import hydra
import pandas as pd

# framework
from run.plot_eval import get_pileup_name

# internal
import tools.physics.jet_substructure as sjets
from tools.datamodule.prepare_data import matrix_to_point_cloud

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
    elif config.generate_substructure: # generate substructure
        print("Calculate substructures")
        
        size = config.eval.size

        # full name
        full_name = f"{config.csv_sample_to_load}{name}" #_size_{size}"

        ### generate sample ###
        gen_data = eval_fw.get_eval_files(should_contain=config.csv_sample_to_load)
        if len(gen_data)==0 and "posterior" in config.csv_sample_to_load:
            path = (f"{eval_fw.eval_folder}/post/flow_N/*" if eval_fw.run_flow
                    else f"{eval_fw.eval_folder}/post/*" )
            gen_data = eval_fw.get_eval_files(eval_files=[i for i in glob(path) if full_name in i],specific_file=None)
            eval_fw.eval_folder = path.replace("*", "")

        # if "posterior" in config.csv_sample_to_load:
        #     eval_fw.eval_folder+="/post/"

        # get size 
        size = 9999*config.eval.size//10_000
            # size = "99990_2000" # TODO fix this

        # truth_jets = gen_data[f"truth_jets_{config.csv_sample_to_load}"]
        gen_cnts = gen_data[[i for i in gen_data if f"cnts_{full_name}" in i][0]]
        gen_jets = gen_data[[i for i in gen_data if f"jets_{full_name}" in i][0]]

        if isinstance(gen_cnts, pd.DataFrame):
            # TODO need to add indexing
            gen_cnts = gen_cnts[["eta", "phi", "pt", "eventNumber"]].values

        # eventNumber = gen_cnts[:, -1]
        # gen_cnts = gen_cnts[:,:4]

        # # create pc
        gen_cnts, mask = matrix_to_point_cloud(gen_cnts, gen_cnts[:, -2],
                                            #   num_per_event_max=max_cnts
                                                )
        if "eventNumber" not in gen_jets.columns:
            # gen_jets=gen_jets.rename(columns={gen_jets.columns[0]:"eventNumber"})
            gen_jets["eventNumber"] = gen_jets.index

        # generated
        # if "flow_N" in eval_fw.eval_folder:
        save_path = f"{eval_fw.eval_folder}/jet_subs/"

        os.makedirs(f"{save_path}/", exist_ok=True)

        out_name = f"{save_path}/jet_substructure_{full_name}.h5"
        
        if not os.path.isfile(out_name):
            print(f"Generated predicted substructure in {out_name}")
            if "posterior" in config.csv_sample_to_load:
                size = None # TODO fix this
            sjets.dump_hlvs(gen_cnts[..., :3][:size], mask[:size],
                            out_path=out_name,
                            addi_col={"eventNumber":gen_cnts[..., 0, -1][:size]}
                            )
            if "posterior" in config.csv_sample_to_load:
                import sys
                sys.exit()

        truth_name = f"{save_path}/jet_substructure_truth_{full_name}.h5"

        if not os.path.isfile(truth_name) and not "posterior" in config.csv_sample_to_load and False:
            # # substruct for true Top
            print("True substructure")
            sjets.dump_hlvs(eval_fw.data.cnts_vars[:None],
                            eval_fw.data.mask_cnts[:None],
                            out_path=f"{save_path}/jet_substructure_truth_{full_name}.h5",
                            # addi_col={"eventNumber":truth_jets["eventNumber"].values[:len(gen_jets)]}
                            )
        
        obs_jet_file_name = f"{config.obs_jets_path}/jet_subs/jet_substructure_ctxt_{full_name}.h5"
        
        if ("single" in config.csv_sample_to_load
            and not os.path.isfile(obs_jet_file_name) and True): # substruct for obs. jet
            os.makedirs(f"{config.obs_jets_path}/jet_subs/",
                        exist_ok=True)
            
            print(f"Generated obs. jet substructure at {obs_jet_file_name}")

            # get ctxt 
            ctxt_path = glob(f"/srv/beegfs/scratch/groups/rodem/pileup_diffusion/data/obs_jets/*{full_name.replace('single', '')}*.npy")

            eval_ctxt= np.load([i for i in ctxt_path if "soft" not in i][0], allow_pickle=True).item()

            sjets.dump_hlvs(eval_ctxt["cnts"][:len(gen_jets)], eval_ctxt["mask"][:len(gen_jets)],
                            out_path=obs_jet_file_name,
                            # addi_col={"eventNumber":truth_jets["eventNumber"].values[:len(gen_jets)]}
                            )
    else:
        print("Generated Top jets")
        
        ### generate sample ###
        eval_fw.load_diffusion()

        # will follow the pileup defined
        if config.eval.flow_path is not None:
            file_name = glob(f"{config.obs_jets_path}/flow_N/{config.eval.flow_path}/obs_jet*{name}*")
        else:
            file_name = glob(f"{config.obs_jets_path}/obs_jet*{name}*")
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
        
        if ("flow_N" in eval_fw.eval_folder):
            eval_ctxt["true_n_cnts"] = np.int64(eval_ctxt["scalars"][:, -1])
        
        if "posterior" in saving_name and config.n_post_to_gen==1:
            raise ValueError("When generating posterior, n_post_to_gen should be larger than 1.")

        # it will save the generates pc
        eval_fw.generate_and_save_post(eval_ctxt, config.n_post_to_gen,
                                       combine_ctxt_size=config.combine_ctxt_size,
                                       saving_name=saving_name)


if __name__ == "__main__":
    main()