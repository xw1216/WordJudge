import hydra
import wandb
from omegaconf import DictConfig

from train import Train


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    for i in range(cfg.repeat):
        run = wandb.init(
            project=cfg.project,
            entity=cfg.wandb_entity,
            reinit=True,
            group="BrainGNN-fold-" + cfg.train.fold,
            dir=cfg.log.wandb_path,
            config={
                "epochs": cfg.train.epochs,
                "batch-size": cfg.train.batch_size,

                "learning-rate": cfg.model.lr,
                "weight_decay": cfg.model.weight_decay,

                "pool-ratio": cfg.model.pool_ratio,
                "lambda-subject": cfg.model.lamb_subject,
                "lambda-group": cfg.model.lamb_group,
            }
        )

        train = Train(cfg)
        train.start()

        run.finish()


if __name__ == '__main__':
    main()
