import copy
import logging
from typing import Optional
from pathlib import Path

import torch
import wandb
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.loader import DataLoader
from wandb.sdk.lib.runid import generate_id
from omegaconf import DictConfig, OmegaConf

import util
import dataset
import model


class Train:
    def __init__(self, cfg: DictConfig):
        self.cfg: DictConfig = cfg
        self.logger_name: str = cfg.log.log_name
        self.group: str = cfg.group
        self.group_id: str = generate_id(5).upper()

        self.seed: int = self.cfg.train.seed
        self.epochs: int = self.cfg.train.epochs
        self.folds: int = self.cfg.train.folds
        self.batch_size: int = self.cfg.train.batch_size
        self.device: torch.device = self.define_device()

        self.model: Optional[model.BrainGNN] = None
        self.optim: Optional[Optimizer] = None
        self.sched: Optional[LRScheduler] = None

        self.final_loader: Optional[DataLoader] = None
        self.model_best_list: list = []
        self.loss_best: float = 1e10
        self.acc_train_best: float = 0.70
        self.acc_valid_best: float = 0.50

    def logger(self):
        return logging.getLogger(self.logger_name)

    def define_device(self):
        if self.cfg.cuda and torch.cuda.is_available():
            self.logger().info('Torch Running on Cuda')
            return torch.device("cuda")
        else:
            self.logger().info('Torch Running on CPU')
            return torch.device("cpu")

    def convert_data(self):
        util.mat2ny(self.cfg)
        self.logger().info('Data preparation complete')

    def build_kfold_graph_loader(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        self.logger().info('Building in-memory dataset')
        conn = dataset.ConnDataset(self.cfg)

        loader = dataset.KFoldGroup(
            conn,
            log=self.logger(),
            fold=self.folds,
            seed=self.seed,
            shuffle=True
        )

        k_cnt = 1

        self.logger().info(f'Splitting data into {self.folds} Fold')
        self.logger().info(f'{self.folds - 2} random as Train Set, 1 random as Valid Set, 1 as Test Set')
        for train_data, valid_data, test_data in loader.split():
            self.logger().info(f'Fold {k_cnt}/{loader.fold - 1} th split complete')
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            valid_loader = DataLoader(valid_data, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
            yield train_loader, valid_loader, test_loader

    def build_model(self):
        self.model = model.BrainGNN(
            dim_in=self.cfg.dataset.n_roi,
            num_class=self.cfg.dataset.n_class,
            num_cluster=self.cfg.model.n_cluster,
            pool_ratio=self.cfg.model.pool_ratio,
            drop_ratio=self.cfg.model.drop_ratio,
            dim_conv1=self.cfg.model.dim_conv1,
            dim_conv2=self.cfg.model.dim_conv2,
            dim_mlp=self.cfg.model.dim_mlp
        ).to(self.device)

        self.logger().info('Printing model construction')
        self.logger().info(self.model)

        self.optim = model.build_optimizer(
            name=self.cfg.train.optim,
            param=self.model.parameters(),
            lr=self.cfg.model.lr,
            weight_decay=self.cfg.model.weight_decay
        )
        self.logger().info('Printing optimizer config')
        self.logger().info(self.optim)

        self.sched = model.build_scheduler(
            name=self.cfg.train.sched,
            optim=self.optim,
            step_size=self.cfg.train.step_size,
            gamma=self.cfg.model.gamma,
            lr=self.cfg.model.lr,
            epoch=self.cfg.train.epochs
        )
        self.logger().info('Printing scheduler type')
        self.logger().info(self.sched.__class__.__name__)

    def get_optim_lr(self) -> float:
        if self.optim is None:
            return .0
        else:
            return self.optim.param_groups[0]['lr']

    def loss_batch(
            self, output: torch.Tensor, labels: torch.Tensor,
            weight1: torch.Tensor, weight2: torch.Tensor,
            score1: torch.Tensor, score2: torch.Tensor
    ) -> model.LossSelector:
        res = model.LossSelector()

        res.loss_ce = model.cross_entropy_loss(output, labels)

        res.loss_unit1 = model.unit_loss(weight1)
        res.loss_unit2 = model.unit_loss(weight2)

        res.loss_top1 = model.top_k_loss(
            score1, self.cfg.model.pool_ratio, self.cfg.model.eps
        )
        res.loss_top2 = model.top_k_loss(
            score2, self.cfg.model.pool_ratio, self.cfg.model.eps
        )

        res.loss_consist = model.consist_loss(
            score1, labels, self.cfg.dataset.n_class, self.device
        )

        res.loss_all = res.loss_ce + (
                res.loss_unit1 + res.loss_unit2) + (
                res.loss_top1 + res.loss_top2) * self.cfg.model.lamb_top + (
                res.loss_consist) * self.cfg.model.lamb_consist

        return res

    def train(self, epoch: int, loader: DataLoader):
        # for param in self.optim.param_groups:
        #     self.logger().info(param)

        self.model.train()
        self.optim.zero_grad()

        score1_list, score2_list = [], []
        weight1_list, weight2_list = [], []
        num_graph_all, loss_sum = 0, 0
        step = 0

        # noinspection PyTypeChecker
        for data in loader:
            # dynamic created DataBatch class
            data = data.to(self.device)
            num_graph_all += data.num_graphs

            output, weight1, weight2, score1, score2 = self.model(data)

            loss = self.loss_batch(output, data.y, weight1, weight2, score1, score2)
            loss.loss_all.backward()
            loss_sum += loss.loss_all.detach().item() * data.num_graphs

            self.optim.step()
            self.optim.zero_grad()

            score1_list.append(score1.view(-1).detach().cpu().numpy())
            score2_list.append(score2.view(-1).detach().cpu().numpy())
            weight1_list.append(weight1.detach().cpu().numpy())
            weight2_list.append(weight2.detach().cpu().numpy())

            with torch.no_grad():
                # self.logger().info(f'Step {step} for epoch {epoch}, loss {loss.loss_all.item()}')
                log_dict = {
                    "train/step": epoch * loader.__len__() + step,
                    "train/loss_ce": loss.loss_ce,
                    "train/loss_avg": loss.loss_all,
                    "train/loss_unit1": loss.loss_unit1,
                    "train/loss_unit2": loss.loss_unit2,
                    "train/loss_top1": loss.loss_top1,
                    "train/loss_top2": loss.loss_top2,
                    "train/loss_consist": loss.loss_consist,
                    # "train/pool_weight_1": torch.from_numpy(weight1_list[-1]),
                    # "train/pool_weight_2": torch.from_numpy(weight2_list[-1]),
                    # "train/top_k_score_1": torch.from_numpy(score1_list[-1]),
                    # "train/top_k_score_2": torch.from_numpy(score2_list[-1])
                }
                wandb.log(log_dict)

            step += 1

        loss_avg = loss_sum / num_graph_all
        return loss_avg, score1_list, score2_list, weight1_list, weight2_list

    def valid_loss(self, epoch: int, loader: DataLoader, is_final: bool = False):
        self.model.eval()
        loss_sum, num_graph_all = 0, 0

        # noinspection PyTypeChecker
        for data in loader:
            data = data.to(self.device)
            num_graph_all += data.num_graphs

            output, weight1, weight2, score1, score2 = self.model(data)
            loss = self.loss_batch(output, data.y, weight1, weight2, score1, score2)

            loss_sum += loss.loss_all.item() * data.num_graphs

            if not is_final:
                with torch.no_grad():
                    log_dict = {
                        "epoch/step": epoch,
                        "epoch/valid_loss_ce": loss.loss_ce,
                        "epoch/valid_loss_avg": loss.loss_all,
                        "epoch/valid_loss_unit1": loss.loss_unit1,
                        "epoch/valid_loss_unit2": loss.loss_unit2,
                        "epoch/valid_loss_top1": loss.loss_top1,
                        "epoch/valid_loss_top2": loss.loss_top2,
                        "epoch/valid_loss_consist": loss.loss_consist,
                    }
                    wandb.log(log_dict)

        loss_avg = loss_sum / num_graph_all
        return loss_avg

    def valid_acc(self, loader: DataLoader, is_print_label: bool = False):
        self.model.eval()
        num_graph_all, correct_sum = 0, 0

        # noinspection PyTypeChecker
        for data in loader:
            data = data.to(self.device)
            num_graph_all += data.num_graphs

            output, _, _, _, _ = self.model(data)
            predict = output.max(dim=1)[1]
            correct_sum += predict.eq(data.y).sum().item()

            if is_print_label:
                self.logger().warning(f'label: {data.y.tolist()}, predict: {predict.tolist()}')
            # self.logger().warning(f'predict: {predict.tolist()}')
            # self.logger().warning(f'label: {label.tolist()}')

        acc = correct_sum / num_graph_all
        return acc

    def run(self):
        self.logger().info('Printing program configurations')
        self.logger().info(OmegaConf.to_yaml(self.cfg))

        self.logger().info('Converting raw data to npy files')
        self.convert_data()

        fold = 0

        self.logger().info('Training Start...')
        for train_loader, valid_loader, test_loader in self.build_kfold_graph_loader():
            self.build_model()
            self.final_loader = test_loader

            run_wandb = self.createWandbRun(self.group, 'train', fold)
            timer = util.Timer()
            for epoch in range(0, self.epochs):
                timer.start()

                loss_train, _, _, _, _ = self.train(epoch, train_loader)
                acc_train = self.valid_acc(train_loader)
                self.logger().warning(f'*** Epoch {epoch}, Train Loss {loss_train}, Train Acc {acc_train}')

                self.sched.step()

                # TODO solve overfit on train dataset
                loss_valid = self.valid_loss(epoch, valid_loader)
                acc_valid = self.valid_acc(valid_loader, True)
                self.logger().warning(f'*** Epoch {epoch}, Valid Loss {loss_valid}, Valid Acc {acc_valid}')

                timer.end()

                wandb.log({
                    "epoch/step": epoch,
                    "epoch/lr": self.get_optim_lr(),
                    "epoch/loss_train": loss_train,
                    "epoch/acc_train": acc_train,
                    "epoch/loss_valid": loss_valid,
                    "epoch/acc_valid": acc_valid,
                    "epoch/timing": timer.last()
                })

                if acc_train >= self.acc_train_best and acc_valid >= self.acc_valid_best and epoch > 5:
                    self.logger().info('Saving best model')
                    self.loss_best = loss_valid
                    self.acc_train_best = acc_train
                    self.acc_valid_best = acc_valid
                    if len(self.model_best_list) == fold + 1:
                        self.model_best_list.pop()
                    self.model_best_list.append((fold, copy.deepcopy(self.model.state_dict())))

                    if self.cfg.train.save_model:
                        self.logger().info('Saving best model')
                        save_path = Path(self.cfg.train.save_path, 'fold-' + str(fold) + '.pth')
                        torch.save(self.model_best_list[-1], save_path)

            fold += 1
            self.loss_best = 1e10
            self.acc_train_best = 0.70
            self.acc_valid_best = 0.50
            run_wandb.finish()

    def test(self):
        if self.cfg.train.load_model:
            self.build_model()
            run_wandb = self.createWandbRun(self.group, 'test')
            for fold in range(self.folds - 1):
                model_path = Path(self.cfg.train.save_path, 'fold-' + str(fold) + '.pth')
                if model_path.exists():
                    model_dict = torch.load(model_path)
                    self.model.load_state_dict(model_dict)
                    self.model.eval()
                    loss_test = self.valid_loss(0, self.final_loader, is_final=True)
                    acc_test = self.valid_acc(self.final_loader, True)

                    self.logger().warning(f'*** Final Test loss {loss_test}, acc {acc_test}')

                    wandb.log({
                        "test/fold": fold,
                        "test/loss_test": loss_test,
                        "test/acc_test": acc_test,
                    })

            run_wandb.finish()
        else:
            run_wandb = self.createWandbRun(self.group, 'test')
            for model_best in self.model_best_list:
                self.model.load_state_dict(model_best[1])
                self.model.eval()
                fold = model_best[0]

                loss_test = self.valid_loss(0, self.final_loader, is_final=True)
                acc_test = self.valid_acc(self.final_loader, True)

                self.logger().warning(
                    f'*** Final Test fold {fold}: loss {loss_test}, acc {acc_test}'
                )

                wandb.log({
                    "test/fold": fold,
                    "test/loss_test": loss_test,
                    "test/acc_test": acc_test,
                })

            run_wandb.finish()

    def createWandbRun(self, group: str, job_type: str, fold: int = 0):
        cfg = self.cfg
        group = f'{group}-{self.group_id}'
        name = f'{group}-{job_type}-{fold}'
        run = wandb.init(
            project=cfg.project,
            entity=cfg.entity,
            reinit=True,
            group=group,
            name=name,
            job_type=job_type,
            dir=cfg.log.wandb_path,
            config={
                "label_type": cfg.dataset.label_type,
                "node_type": cfg.dataset.node_type,
                "edge_type": cfg.dataset.edge_type,

                "lr": cfg.model.lr,
                "gamma": cfg.model.gamma,
                "weight_decay": cfg.model.weight_decay,
                "n_cluster": cfg.model.n_cluster,
                "pool_ratio": cfg.model.pool_ratio,
                "drop_ratio": cfg.model.drop_ratio,
                "lambda_subject": cfg.model.lamb_top,
                "lambda_group": cfg.model.lamb_consist,

                "dim_conv1": cfg.model.dim_conv1,
                "dim_conv2": cfg.model.dim_conv2,
                "dim-mlp": cfg.model.dim_mlp,

                "seed": cfg.train.seed,
                "epochs": cfg.train.epochs,
                "batch_size": cfg.train.batch_size,
                "folds": cfg.train.folds,
                "optimizer": cfg.train.optim,
                "scheduler": cfg.train.sched,
                "step_size": cfg.train.step_size,
            }
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("epoch/step")
        wandb.define_metric("epoch/*", step_metric="epoch/step")
        wandb.define_metric("test/fold")
        wandb.define_metric("test/*", step_metric="test/fold")
        return run
