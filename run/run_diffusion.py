import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import hydra
from tools.omegaconf_utils import check_config

@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="config")
def main(config):

    # check/printing config and load new if necessary
    config, wandb = check_config(config)

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
