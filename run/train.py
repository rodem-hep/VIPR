import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging
from omegaconf import OmegaConf

import hydra
import torch as T
from tools.tools.omegaconf_utils import check_config, instantiate_collection, save_config
import tools.tools.misc as misc

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="config")
def main(config):
    T.set_float32_matmul_precision('medium')
    T.autograd.set_detect_anomaly(True)

    # check/printing config and load new if necessary
    config = check_config(config)
    
    log.info("Instantiating the data")
    train_loader = hydra.utils.instantiate(config.data.train,
                                           loader_config=config.data.loader_config)

    log.info("Instantiating the testing data dn evaluation framework")
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
    log.info("Instantiating the callbacks")
    callbacks = instantiate_collection(config.callbacks)
    
    log.info("Instantiating the WandB")
    wandb = hydra.utils.instantiate(config.wandb,
                                    resume=config.get('ckpt_path') is not None)

    log.info("Instantiating the Trainer")
    trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks,
                                      logger=wandb)

    log.info("Instantiating the models")
    with trainer.init_module():

        if config.get('ckpt_path') is not None:
            log.info(f"Loading model weights from checkpoint: {config.ckpt_path}")

            model_class = hydra.utils.get_class(config.model._target_)

            model = model_class.load_from_checkpoint(
                config.ckpt_path, map_location="cuda" if T.cuda.is_available() else "cpu",
                eval_fw=test_loader, strict=False,
                **config.model
                )

        else:
            model = hydra.utils.instantiate(config.model, eval_fw=test_loader)
            model = hydra.utils.instantiate(config.model, eval_fw=test_loader)

    if wandb is not None:
        log.info("log run parameters")
        wandb.experiment.config.update(model.ctp)

        wandb.experiment.config.update(OmegaConf.to_container(
                    config, resolve=True, throw_on_missing=True
                    ))
        
    log.info("Saving config so job can be resumed")
    save_config(config)
    
    # train model
    log.info("Start training:")
    trainer.fit(model=model,
                train_dataloaders=train_loader.train_dataloader(),
                val_dataloaders=test_loader.test_dataloader(),
                )
    
    if trainer.state.status == "finished":
        log.info("Declaring job as finished!")
        misc.save_declaration("train_finished")


        
if __name__ == "__main__":
    main()
