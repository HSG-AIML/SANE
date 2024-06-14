import torch
import torch.nn as nn
import numpy as np
from .def_AE import AE
from .def_loss import GammaContrastReconLoss
import itertools

import logging

import inspect

from lightning.fabric import Fabric
from lightning.fabric import seed_everything

from pathlib import Path

from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm


class AEModule(nn.Module):
    def __init__(self, config):
        super(AEModule, self).__init__()

        logging.info("Initialize Model")
        self.config = config

        # setting seeds for reproducibility
        seed = config.get("seed", 42)
        seed_everything(seed)

        self.device = config.get("device", torch.device("cpu"))
        if type(self.device) is not torch.device:
            self.device = torch.device(self.device)
        logging.info(f"device: {self.device}")
        if type(self.device) == torch.device("cuda"):
            # device now becomes an integer with the gpu_id
            # torch recognizes this must be a gpu id and handles the backend
            self.device = config.get("gpu_id", 0)

        model = AE(config)

        if config.get("model::compile", False):
            logging.info("compiling the model... (takes a ~minute)")
            # cuda before compile :) https://discuss.pytorch.org/t/torch-compile-before-or-after-cuda/176031
            model = torch.compile(model)  # requires PyTorch 2.0
            logging.info("compiled successfully")
        self.model = model

        #
        self.criterion = GammaContrastReconLoss(
            gamma=config.get("training::gamma", 0.5),
            reduction=config.get("training::reduction", "mean"),
            batch_size=config.get("trainset::batchsize", 64),
            temperature=config.get("training::temperature", 0.1),
            contrast=config.get("training::contrast", "simclr"),
            z_var_penalty=config.get("training::z_var_penalty", 0.0),
        )

        # send model and criterion to device
        self.distributed = config.get("distributed", False)
        if self.distributed == False:
            print(f"Running single-gpu. send model to device: {self.device}")
            self.model.to(self.device)
            self.criterion.to(self.device)
        elif self.distributed == "ddp":
            print(f"Running DDP. send model to device: {self.device}")
            # if model is in distributed data parallel
            self.model = DDP(self.model, device_ids=[self.device])
            self.criterion = DDP(self.criterion, device_ids=[self.device])

        # initialize model in eval mode
        self.model.eval()

        # gather model parameters and projection head parameters
        self.params = self.parameters()

        # set optimizer
        self.set_optimizer(config)

        # automatic mixed precision
        self.use_amp = (
            True if config.get("training::precision", "full") == "amp" else False
        )
        if self.use_amp:
            print(f"++++++ USE AUTOMATIC MIXED PRECISION +++++++")
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # init gradien clipping
        if config.get("training::gradient_clipping", None) == "norm":
            self.clip_grads = self.clip_grad_norm
            self.clipping_value = config.get("training::gradient_clipp_value", 5)
        elif config.get("training::gradient_clipping", None) == "value":
            self.clip_grads = self.clip_grad_value
            self.clipping_value = config.get("training::gradient_clipp_value", 5)
        else:
            self.clip_grads = None

        # init scheduler
        self.set_scheduler(config)

        # flag to reset the optimizer. Defaults to False, can be set to True for specific experiments
        self.reset_optimizer = False

        self._save_model_checkpoint = True

    def clip_grad_norm(
        self,
    ):
        # print(f"clip grads by norm")
        # nn.utils.clip_grad_norm_(self.params, self.clipping_value)
        nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)

    def clip_grad_value(
        self,
    ):
        # print(f"clip grads by value")
        # nn.utils.clip_grad_value_(self.params, self.clipping_value)
        nn.utils.clip_grad_value_(self.parameters(), self.clipping_value)

    def set_transforms(
        self, transforms_train=None, transforms_test=None, transforms_downstream=None
    ):
        if transforms_train is not None:
            self.transforms_train = transforms_train
        else:
            self.transforms_train = torch.nn.Sequential()
        if transforms_test is not None:
            self.transforms_test = transforms_test
        else:
            self.transforms_test = torch.nn.Sequential()
        if transforms_downstream is not None:
            self.transforms_downstream = transforms_downstream
        else:
            self.transforms_downstream = torch.nn.Sequential()

    def forward(self, x, p):
        # pass forward call through to model
        logging.debug(f"x.shape: {x.shape}")
        z = self.forward_encoder(x, p)
        logging.debug(f"z.shape: {z.shape}")
        zp = self.model.projection_head(z)
        logging.debug(f"zp.shape: {zp.shape}")
        y = self.forward_decoder(z, p)
        logging.debug(f"y.shape: {y.shape}")
        return z, y, zp

    def forward_encoder(self, x, p, mask=None):
        # normalize input features
        z = self.model.forward_encoder(x, p, mask)
        return z

    def forward_decoder(self, z, p, mask=None):
        y = self.model.forward_decoder(z, p, mask)
        return y

    def forward_embeddings(self, x, p):
        # for downstream tasks computation -> cast to full
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        z = self.model.forward_embeddings(x, p)
        return z

    def set_optimizer(self, config):
        """
        finds paramters, sets wd for decay and non-decay paramters
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": config.get("optim::wd", 3e-5)},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        if config.get("optim::optimizer", "adamw") == "sgd":
            self.optimizer = torch.optim.SGD(
                params=self.parameters(),
                lr=config.get("optim::lr", 3e-4),
                momentum=config.get("optim::momentum", 0.9),
                weight_decay=config.get("optim::wd", 3e-5),
            )
        elif config.get("optim::optimizer", "adamw") == "adam":
            self.optimizer = torch.optim.Adam(
                params=self.parameters(),
                lr=config.get("optim::lr", 3e-4),
                weight_decay=config.get("optim::wd", 3e-5),
            )
        elif config.get("optim::optimizer", "adamw") == "adamw":
            # Create AdamW optimizer and use the fused version if it is available
            fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and self.device == "cuda"
            extra_args = dict(fused=True) if use_fused else dict()
            print(f"using fused AdamW: {use_fused}")
            self.optimizer = torch.optim.AdamW(
                params=optim_groups,
                lr=config.get("optim::lr", 3e-4),
                # betas=betas,
                **extra_args,
            )

            # torch.optim.AdamW(
            #     params=self.parameters(),
            #     lr=config.get("optim::lr", 3e-4),
            #     weight_decay=config.get("optim::wd", 3e-5),
            # )
        elif config.get("optim::optimizer", "adamw") == "lamb":
            self.optimizer = torch.optim.Lamb(
                params=self.parameters(),
                lr=config.get("optim::lr", 3e-4),
                weight_decay=config.get("optim::wd", 3e-5),
            )
        else:
            raise NotImplementedError(
                f'the optimizer {config.get("optim::optimizer", "adam")} is not implemented. break'
            )

    def set_scheduler(self, config):
        if config.get("optim::scheduler", None) == None:
            self.scheduler = None
        # elif config.get("optim::scheduler", None) == "ReduceLROnPlateau":
        #     mode = config.get("optim::scheduler_mode", "min")
        #     factor = config.get("optim::scheduler_factor", 0.1)
        #     patience = config.get("optim::scheduler_patience", 10)
        #     threshold = config.get("optim::scheduler_threshold", 1e-4)
        #     threshold_mode = config.get("optim::scheduler_threshold_mode", "rel")
        #     cooldown = config.get("optim::scheduler_cooldown", 0)
        #     min_lr = config.get("optim::scheduler_min_lr", 0.0)
        #     eps = config.get("optim::scheduler_eps", 1e-8)

        #     self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         self.optimizer,
        #         mode=mode,
        #         factor=factor,
        #         patience=patience,
        #         threshold=threshold,
        #         threshold_mode=threshold_mode,
        #         cooldown=cooldown,
        #         min_lr=min_lr,
        #         eps=eps,
        #         verbose=False,
        #     )
        elif config.get("optim::scheduler", None) == "OneCycleLR":
            total_steps = (
                config.get("training::epochs_train", 150)
                * config["training::steps_per_epoch"]
                * config.get("training::test_epochs", 1)
            )
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=config["optim::lr"],
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy="cos",
                cycle_momentum=True,
                base_momentum=0.85,
                max_momentum=0.95,
                div_factor=25.0,
                final_div_factor=10000.0,
                three_phase=False,
                last_epoch=-1,
                verbose=False,
            )

    def save_model(self, experiment_dir):
        """
        Saves the model to the given path using fabric routines
        Args:
            path (str): path to save the model
        Returns:
            None
        """
        logging.info(f"save model to {experiment_dir}")
        # assure logging path exists
        Path(experiment_dir).mkdir(exist_ok=True, parents=True)
        # define name for checkpoint
        path = Path(experiment_dir).joinpath("state.pt")
        if self.distributed == False:
            state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            if self.scheduler is not None:
                state["scheduler"] = self.scheduler.state_dict()
            # self.fabric.save(path, state) # remove fabric for now
            torch.save(state, path)
            return None
        # if ddp: only rank==0 saves the model
        elif self.distributed == "ddp" and self.rank == 0:
            state = {
                "model": self.model.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            torch.save(state, path)
            return None

    def load_model(self, experiment_dir):
        """
        Saves the model to the given path using fabric routines
        Args:
            path (str): path to save the model
        Returns:
            None
        """
        ## new fabric way
        path = Path(experiment_dir).joinpath("state.pt")
        # remove fabric for now
        # if self.reset_optimizer:
        #     state = {"model": self.model}
        # else:
        #     state = {"model": self.model, "optimizer": self.optimizer}
        # self.fabric.load(path, state)
        state = torch.load(path)
        self.model.load_state_dict(state["model"])
        if not self.reset_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler", None) is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        return None

    # ##########################
    # one training step / batch
    # ##########################
    def train_step(self, x_i, m_i, p_i, x_j, m_j, p_j):
        """
        performs one training step with a batch of data
        # (not currently) using fabric to distribute the training across multiple gpus
        # instead, use cuda amp to speed up training
        Args:
            x_i (torch.Tensor): batch of input features view 1
            m_i (torch.Tensor): batch of input masks view 1
            p_i (torch.Tensor): batch of input positions
            x_j (torch.Tensor): batch of input features veiw 2
            m_j (torch.Tensor): batch of input masks view 2
            p)j (torch.Tensor): batch of input positions
        Returns:
            perf (dict): dictionary with performance metrics
        """
        with torch.cuda.amp.autocast(enabled=True):
            # zero grads before training steps
            self.optimizer.zero_grad(set_to_none=True)
            # forward pass with both views
            z_i, y_i, zp_i = self.forward(x_i, p_i)
            z_j, y_j, zp_j = self.forward(x_j, p_j)
            # cat y_i, y_j and x_i, x_j, and m_i, m_j
            x = torch.cat([x_i, x_j], dim=0)
            y = torch.cat([y_i, y_j], dim=0)
            m = torch.cat([m_i, m_j], dim=0)
            logging.debug(
                f"train step - x: {x.shape}; y: {y.shape}, m: {m.shape}, z_i {z_i.shape}; z_j {z_j.shape};  zp_i {zp_i.shape}; zp_j {zp_j.shape}"
            )
            # compute loss
            perf = self.criterion(z_i=zp_i, z_j=zp_j, y=y, t=x, m=m)
            # prop loss backwards to
            loss = perf["loss/loss"]
        # technically, there'd need to be a scaler for each loss individually.
        self.scaler.scale(loss).backward()
        # if gradient clipping is to be used...
        if self.clip_grads is not None:
            # # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)
            # Since the gradients of optimizer's assigned params are now unscaled, clips as usual.
            self.clip_grads()
        # update parameters
        self.scaler.step(self.optimizer)
        # update scaler
        self.scaler.update()
        # update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        # return perf
        return perf

    # one training epoch
    def train_epoch(self, trainloader, epoch=0, show_progress=False):
        """
        performs one training epoch, i.e. iterates over all batches in the trainloader and aggregates results
        Args:
            trainloader (torch.utils.data.DataLoader): trainloader
            epoch (int): epoch number (optional)
        Returns:
            perf (dict): dictionary with performance metrics aggregated over all batches
        """
        logging.info(f"train epoch {epoch}")
        # set model to training mode
        self.model.train()

        # init accumulated loss, accuracy
        perf_out = {}
        n_data = 0

        # set epoch for distributed sampler
        if self.distributed == "ddp":
            self.train_data.sampler.set_epoch(epoch)

        # enter loop over batches
        disable = not show_progress
        for idx, data in tqdm.tqdm(
            enumerate(trainloader),
            disable=disable,
            desc=f"training epoch {epoch}",
            total=len(trainloader),
        ):
            # # move data to device
            # data = (ddx.to(self.device) for ddx in data)
            data = [
                ddx.to(self.device) for ddx in data
            ]  # use list instead of tuple, generator objects can cause problems..
            # pass through transforms (if any)
            x_i, m_i, p_i, x_j, m_j, p_j = self.transforms_train(*data)

            """ deprecated way
            data, mask, position p
            x, m, p, _ = data
            move data to device
            x, m, p = (
                x.to(self.device),
                m.to(self.device),
                p.to(self.device),
            )
            pass through transforms (if any)
            x_i, m_i, p_i, x_j, m_j, p_j = self.transforms_train(x, m, p)
            """
            # ### hotfix: cast position to torch.int
            p_i = p_i.to(torch.int).to(self.device)
            p_j = p_j.to(torch.int).to(self.device)
            # ### end hotfix
            # take train step
            perf = self.train_step(x_i, m_i, p_i, x_j, m_j, p_j)
            # scale loss with batchsize (get's normalized later)
            for key in perf.keys():
                if key not in perf_out:
                    perf_out[key] = perf[key].item() * x_i.shape[0]
                else:
                    perf_out[key] += perf[key].item() * x_i.shape[0]
            n_data += x_i.shape[0]
            # hotfix
            del x_i, m_i, p_i, x_j, m_j, p_j

        self.model.eval()
        # compute epoch running losses
        for key in perf_out.keys():
            perf_out[key] /= n_data
            if torch.is_tensor(perf_out[key]):
                perf_out[key] = perf_out[key].item()

        return perf_out

    # test batch
    def test_step(self, x_i, m_i, p_i, x_j, m_j, p_j):
        """
        #TODO
        """
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                # forward pass with both views
                z_i, y_i, zp_i = self.forward(x_i, p_i)
                z_j, y_j, zp_j = self.forward(x_j, p_j)
                # cat y_i, y_j and x_i, x_j, and m_i, m_j
                x = torch.cat([x_i, x_j], dim=0)
                y = torch.cat([y_i, y_j], dim=0)
                m = torch.cat([m_i, m_j], dim=0)
                # compute loss
                perf = self.criterion(z_i=zp_i, z_j=zp_j, y=y, t=x, m=m)
            return perf

    # test epoch
    def test_epoch(self, testloader, epoch=0, show_progress=False):
        logging.info(f"test at epoch {epoch}")
        # set model to eval mode
        self.model.eval()
        # init accumulated loss, accuracy
        perf_out = {}
        n_data = 0
        # enter loop over batches
        disable = not show_progress
        for idx, data in tqdm.tqdm(
            enumerate(testloader),
            disable=disable,
            desc=f"test epoch {epoch}",
            total=len(testloader),
        ):
            # # data, mask, position p
            # x, m, p = data
            # # move data to device
            # data = (ddx.to(self.device) for ddx in data)
            data = [
                ddx.to(self.device) for ddx in data
            ]  # use list instead of tuple, generator objects can cause problems..
            # pass through transforms (if any)
            x_i, m_i, p_i, x_j, m_j, p_j = self.transforms_test(*data)
            ### explicit, old way
            # x, m, p = (
            #     x.to(self.device),
            #     m.to(self.device),
            #     p.to(self.device),
            # )
            # x_i, m_i, p_i, x_j, m_j, p_j = self.transforms_test(x, m, p)
            ### hotfix: cast position to torch.int
            p_i = p_i.to(torch.int).to(self.device)
            p_j = p_j.to(torch.int).to(self.device)
            ### end hotfix

            ###
            # compute loss
            perf = self.test_step(x_i, m_i, p_i, x_j, m_j, p_j)
            # scale loss with batchsize (get's normalized later)
            for key in perf.keys():
                if key not in perf_out:
                    perf_out[key] = perf[key].item() * x_i.shape[0]
                else:
                    perf_out[key] += perf[key].item() * x_i.shape[0]
            n_data += x_i.shape[0]

        # compute epoch running losses
        for key in perf_out.keys():
            perf_out[key] /= n_data
            if torch.is_tensor(perf_out[key]):
                perf_out[key] = perf_out[key].item()

        return perf_out


"""
    # training loop over all epochs
    def train_loop(self, config):
        logging.info("##### enter training loop ####")

        # unpack training_config
        epochs_train = config["training::epochs_train"]
        start_epoch = config["training::start_epoch"]
        output_epoch = config["training::output_epoch"]
        test_epochs = config["training::test_epochs"]
        tf_out = config["training::tf_out"]
        checkpoint_dir = config["training::checkpoint_dir"]
        tensorboard_dir = config["training::tensorboard_dir"]

        if tensorboard_dir is not None:
            tb_writer = SummaryWriter(log_dir=tensorboard_dir)
        else:
            tb_writer = None

        # trainloaders with matching lenghts

        trainloader = config["training::trainloader"]
        testloader = config["training::testloader"]

        ## compute loss_mean
        self.loss_mean = self.criterion.compute_mean_loss(testloader)

        # compute initial test loss
        loss_test, loss_test_contr, loss_test_recon, rsq_test = self.test(
            testloader,
            epoch=0,
            writer=tb_writer,
            tf_out=tf_out,
        )

        # write first state_dict
        perf_dict = {
            "loss_train": 1e15,
            "loss_test": loss_test,
            "rsq_train": -999,
            "rsq_test": rsq_test,
        }

        self.save_model(epoch=0, perf_dict=perf_dict, path=checkpoint_dir)
        self.best_epoch = 0
        self.loss_best = 1e15

        # initialize the epochs list
        epoch_iter = range(start_epoch, start_epoch + epochs_train)
        

        self.last_checkpoint = self.model.state_dict()
        return self.loss_best
"""
