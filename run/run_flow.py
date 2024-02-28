"train flow to estimate p(N)"
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)


import hydra

from tools.flows import Flow, stacked_norm_flow
from tools.modules import IterativeNormLayer


class TransFlow(Flow):
    def __init__(self, data_dims:dict, flow_conf:dict, embed_conf:dict,
                 device="cuda"):
        super().__init__()

        self.data_dims = data_dims
        self.flow_conf = flow_conf
        self.embed_conf = embed_conf
        
    def get_network(self):
        # Initialise the individual normalisation layers
        self.cnsts_normaliser = IterativeNormLayer(self.data_dims["cnts"])
        self.scalars_normaliser = IterativeNormLayer(self.data_dims["scalars"])
        
        self.flow = stacked_norm_flow(**self.flow_conf).to(self.device)
        self.embed = hydra.utils.instantiate(self.embed_conf)

    def get_embed(self, inputs:dict):

        # normalise inputs
        if "cnts" in inputs:
            inputs["cnts"] = self.cnsts_normaliser(inputs["cnts"])

        if "scalars" in inputs:
            inputs["scalars"] = self.cnsts_normaliser(inputs["scalars"])

        # run embedding and return
        return self.embed(**inputs)

@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="flow_config")
def main(config:dict):

    # check/printing config and load new if necessary
    # config, wandb = check_config(config)

    #dataloader
    train_loader = hydra.utils.instantiate(config.data.train,
                                           loader_config=config.data.loader_config)
    if "jet" in config.data.valid._target_.lower():
        test_loader = hydra.utils.instantiate(config.data.valid,
                                            loader_config=config.data.loader_config,
                                            max_cnstits=train_loader.dataset.max_cnstits,
                                            datatype=train_loader.dataset.datatype,
                                            )
    else:
        test_loader = hydra.utils.instantiate(config.data.valid,
                                            loader_config=config.data.loader_config,
                                            )

    # init network
    network=hydra.utils.instantiate(
        config.model,ctxt_dims=train_loader.get_ctxt_shape()
        )

    # init diffusion
    diffusion = hydra.utils.instantiate(
        config.trainer,
        network=network,
        train_loader=train_loader.train_dataloader(),
        test_loader=test_loader.test_dataloader(),
        device=config.device,
        save_path=config.save_path,
        eval_fw=test_loader,
        wandb=wandb,
        )

    train_params = diffusion.network.count_trainable_parameters()
    print(f"Trainable parameters: {train_params}")

    wandb.config.update({"Model Parameters": train_params})

    diffusion.run_training()


if __name__ == "__main__":
    main()