import hydra

from omegaconf import  DictConfig

import util
from train import Train


@hydra.main(config_path="conf", config_name="Config", version_base=None)
def main(cfg: DictConfig) -> None:
    log = util.append_logger(cfg)
    util.create_project_dir(cfg, log)

    program = Train(cfg)
    program.run()
    program.test()


if __name__ == '__main__':
    main()
