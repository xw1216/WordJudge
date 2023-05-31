import copy
import logging
import os
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

        self.atlas_table = []
        self.timer = util.Timer()

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

    def read_atlas_table(self):
        self.atlas_table = util.read_atlas_table(cfg=self.cfg)

    def build_kfold_graph_loader(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        self.logger().info('Building in-memory dataset')
        conn = dataset.ConnDataset(self.cfg)

        loader = dataset.KFoldGroup(
            conn,
            log=self.logger(),
            session=self.cfg.dataset.n_session,
            batch_size=self.batch_size,
            fold=self.folds,
            seed=self.seed,
            stratified=self.cfg.train.stratified
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
            dim_conv=self.cfg.model.dim_conv,
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
            self,
            output: torch.Tensor, labels: torch.Tensor,
            weights: torch.Tensor, scores: torch.Tensor
    ) -> model.LossSelector:
        conv_len = len(weights)

        res = model.LossSelector(conv_len)

        res.loss_ce = model.cross_entropy_loss(output, labels)
        res.loss_consist = model.consist_loss(
            scores[0], labels, self.cfg.dataset.n_class, self.device
        )

        for i in range(conv_len):
            res.loss_unit[i] = model.unit_loss(weights[i])
            res.loss_top[i] = model.top_k_loss(
                scores[i], self.cfg.model.pool_ratio, self.cfg.model.eps
            )

        loss_unit = res.loss_unit[0]
        loss_top = res.loss_top[0]
        for i in range(1, conv_len):
            loss_unit = loss_unit + res.loss_unit[i]
            loss_top = loss_top + res.loss_top[i]

        res.loss_all = \
            res.loss_ce + \
            loss_unit + \
            loss_top * self.cfg.model.lamb_top + \
            res.loss_consist * self.cfg.model.lamb_consist

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

            output, weights, scores, _ = self.model(data)

            loss = self.loss_batch(output, data.y, weights, scores)
            loss.loss_all.backward()
            loss_sum += loss.loss_all.detach().item() * data.num_graphs

            self.optim.step()
            self.optim.zero_grad()

            score1_list.append(scores[0].view(-1).detach().cpu().numpy())
            score2_list.append(scores[1].view(-1).detach().cpu().numpy())
            weight1_list.append(weights[0].detach().cpu().numpy())
            weight2_list.append(weights[1].detach().cpu().numpy())

            with torch.no_grad():
                # self.logger().info(f'Step {step} for epoch {epoch}, loss {loss.loss_all.item()}')
                log_dict = {
                    "train/step": epoch * loader.__len__() + step,
                    "train/loss_ce": loss.loss_ce,
                    "train/loss_avg": loss.loss_all,
                    "train/loss_unit1": loss.loss_unit[0],
                    "train/loss_unit2": loss.loss_unit[1],
                    "train/loss_top1": loss.loss_top[0],
                    "train/loss_top2": loss.loss_top[1],
                    "train/loss_consist": loss.loss_consist,
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

            output, weights, scores, _ = self.model(data)
            loss = self.loss_batch(output, data.y, weights, scores)

            loss_sum += loss.loss_all.item() * data.num_graphs

            if not is_final:
                with torch.no_grad():
                    log_dict = {
                        "epoch/step": epoch,
                        "epoch/valid_loss_ce": loss.loss_ce,
                        "epoch/valid_loss_avg": loss.loss_all,
                        "epoch/valid_loss_unit1": loss.loss_unit[0],
                        "epoch/valid_loss_unit2": loss.loss_unit[1],
                        "epoch/valid_loss_top1": loss.loss_top[0],
                        "epoch/valid_loss_top2": loss.loss_top[1],
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

            output, _, _, _ = self.model(data)
            predict: torch.Tensor = output.max(dim=1)[1]
            correct_sum += predict.eq(data.y).sum().item()

            if is_print_label:
                self.logger().warning(f'label: {data.y.tolist()}, predict: {predict.tolist()}')

        acc = correct_sum / num_graph_all
        return acc

    def run(self):
        self.timer.start()
        self.logger().info('Printing program configurations')
        self.logger().info(OmegaConf.to_yaml(self.cfg))

        self.logger().info('Converting raw data to npy files')
        self.convert_data()
        self.read_atlas_table()

        fold = 0

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.logger().info('Training Start...')
        for train_loader, valid_loader, test_loader in self.build_kfold_graph_loader():
            self.build_model()
            self.final_loader = test_loader

            run_wandb = self.createWandbRun('train', fold)
            timer = util.Timer()
            for epoch in range(0, self.epochs):
                timer.start()

                loss_train, _, _, _, _ = self.train(epoch, train_loader)
                acc_train = self.valid_acc(train_loader)
                self.logger().warning(f'*** Epoch {epoch}, Train Loss {loss_train}, Train Acc {acc_train}')

                self.sched.step()

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

                if acc_train >= self.acc_train_best and acc_valid >= self.acc_valid_best and epoch > 20:
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
                        if save_path.exists():
                            os.remove(save_path)
                        torch.save(self.model_best_list[-1], save_path)

            fold += 1
            self.loss_best = 1e10
            self.acc_train_best = 0.70
            self.acc_valid_best = 0.50
            run_wandb.finish()

    def test_specific_model(self, fold: int, acc_best: float):
        loss_test = self.valid_loss(0, self.final_loader, is_final=True)
        acc_test = self.valid_acc(self.final_loader, True)

        if acc_test > acc_best:
            acc_best = acc_test
            self.logger().info('Saving best model on final test set')
            save_path = Path(
                self.cfg.train.save_path,
                self.cfg.train.final_file_name
            )
            if save_path.exists():
                os.remove(save_path)
            torch.save(copy.deepcopy(self.model.state_dict()), save_path)

        self.logger().warning(
            f'*** Final Test fold {fold}: loss {loss_test}, acc {acc_test}'
        )

        wandb.log({
            "test/fold": fold,
            "test/loss_test": loss_test,
            "test/acc_test": acc_test,
        })

        return acc_best, acc_test

    def final_test_interpret(self, loader: DataLoader):
        self.model.eval()
        num_graph_all, correct_sum, loss_sum = 0, 0, 0
        score1_mean_true, score1_mean_false = torch.Tensor(), torch.Tensor()

        # noinspection PyTypeChecker
        for data in loader:
            data = data.to(self.device)

            output, weights, scores, score_uni = self.model(data)
            loss = self.loss_batch(output, data.y, weights, scores)
            predict: torch.Tensor = output.max(dim=1)[1]

            loss_sum += loss.loss_all.item() * data.num_graphs
            correct_sum += predict.eq(data.y).sum().item()
            num_graph_all += data.num_graphs

            self.logger().warning(f'label: {data.y.tolist()}, predict: {predict.tolist()}')

            true_perm = (data.y == 1)
            score1_temp_true = score_uni[true_perm].detach().cpu()
            score1_mean_true = torch.concatenate((score1_mean_true, score1_temp_true), dim=0)

            false_perm = (data.y == 0)
            score1_temp_false = score_uni[false_perm].detach().cpu()
            score1_mean_false = torch.concatenate((score1_mean_false, score1_temp_false), dim=0)

        score1_mean_true = torch.mean(score1_mean_true, dim=0, keepdim=False)
        score1_mean_false = torch.mean(score1_mean_false, dim=0, keepdim=False)
        acc = correct_sum / num_graph_all
        loss_avg = loss_sum / num_graph_all
        community_factor: torch.Tensor = self.model.conv[0].embed_linear[0].weight.detach().cpu()

        self.logger().info(f'community_factor: {community_factor}')
        self.logger().info(f'score_true: {score1_mean_true}')
        self.logger().info(f'score_false: {score1_mean_false}')

        return acc, loss_avg, community_factor, score1_mean_true, score1_mean_false

    def test(self):
        acc_best = .0
        acc_list = []

        # load model for each fold from saved files
        if self.cfg.train.load_model:
            self.build_model()
            run_wandb = self.createWandbRun('test')
            for fold in range(self.folds - 1):
                model_path = Path(self.cfg.train.save_path, 'fold-' + str(fold) + '.pth')
                if model_path.exists():
                    model_dict = torch.load(model_path)
                    self.model.load_state_dict(model_dict)
                    self.model.eval()
                    acc_best, acc = self.test_specific_model(fold, acc_best)
                    acc_list.append(acc)
            run_wandb.finish()
        # load model from current training result
        else:
            run_wandb = self.createWandbRun('test')
            for model_best in self.model_best_list:
                self.model.load_state_dict(model_best[1])
                fold = model_best[0]
                self.model.eval()
                acc_best, acc = self.test_specific_model(fold, acc_best)
                acc_list.append(acc)
            run_wandb.finish()

        save_path = Path(
            self.cfg.train.save_path,
            self.cfg.train.final_file_name
        )
        if not save_path.exists():
            raise RuntimeError(f'Final model {str(save_path)} not found')
        else:
            best_model_dict = torch.load(save_path)
            self.model.load_state_dict(best_model_dict)

        acc_final, loss_final, community_factor, \
            score_uni_true, score_uni_false = self.final_test_interpret(self.final_loader)

        self.timer.end()

        acc_mean = 0
        for acc in acc_list:
            acc_mean += acc
        acc_mean /= len(acc_list)

        self.logger().warning('***** Final Result *****')
        self.logger().warning(f'Average Acc: {acc_mean}')
        self.logger().warning(f'Final Acc: {acc_final}')
        self.logger().warning(f'Final Loss: {loss_final}')
        self.logger().warning(f'Last Time: {util.Timer.to_datetime(self.timer.last())}')

        run_wandb = self.createWandbRun('test', fold=1)
        util.draw_atlas_interpret(
            self.atlas_table, community_factor,
            score_uni_true, score_uni_false,
            Path(self.cfg.train.save_path)
        )
        run_wandb.finish()

    def createWandbRun(self, job_type: str, fold: int = 0):
        cfg = self.cfg
        group = f'{self.cfg.dataset.atlas_table_type}-{self.group_id}'
        name = f'{self.group}-{job_type}-{fold}'
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

                "dim_conv1": cfg.model.dim_conv[0],
                "dim_conv2": cfg.model.dim_conv[1],
                "dim_conv": cfg.model.dim_conv,
                "dim-mlp": cfg.model.dim_mlp,

                "seed": cfg.train.seed,
                "epochs": cfg.train.epochs,
                "batch_size": cfg.train.batch_size,
                "folds": cfg.train.folds,
                "optimizer": cfg.train.optim,
                "scheduler": cfg.train.sched,
                "step_size": cfg.train.step_size,

                "aug_noise": cfg.dataset.aug_noise,
                "aug_noise_bound": cfg.dataset.noise_bound,
                "aug_mixup": cfg.dataset.aug_mixup,
                "aug_mixup_alpha": cfg.dataset.mixup_alpha
            }
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("epoch/step")
        wandb.define_metric("epoch/*", step_metric="epoch/step")
        wandb.define_metric("test/fold")
        wandb.define_metric("test/*", step_metric="test/fold")
        return run
