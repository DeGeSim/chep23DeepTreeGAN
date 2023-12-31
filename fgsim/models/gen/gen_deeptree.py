from math import prod
from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.models.common import DynHLVsLayer, FtxScaleLayer, MPLSeq
from fgsim.models.common.deeptree import BranchingLayer, Tree, TreeGraph
from fgsim.monitoring import logger

# from fgsim.plot.model_plotter import model_plotter


class ModelClass(nn.Module):
    def __init__(
        self,
        n_global: int,
        n_cond: int,
        ancestor_mpl: Dict,
        child_mpl: Dict,
        branching_param: Dict,
        final_layer_scaler: bool,
        connect_all_ancestors: bool,
        dim_red_in_branching: bool,
        pruning: str,
        equivar: bool,
        **kwargs,
    ):
        super().__init__()
        self.n_global = n_global
        self.n_cond = n_cond
        self.batch_size = conf.loader.batch_size
        self.final_layer_scaler = final_layer_scaler
        self.ancestor_mpl = ancestor_mpl
        self.child_mpl = child_mpl
        self.dim_red_in_branching = dim_red_in_branching
        self.pruning = pruning
        self.equivar = equivar

        self.features = OmegaConf.to_container(conf.tree.features)
        self.branches = OmegaConf.to_container(conf.tree.branches)
        n_levels = len(self.features)
        self.num_part_idx = conf.loader.y_features.index("num_particles")

        # Shape of the random vector
        self.z_shape = conf.loader.batch_size, 1, self.features[0]

        # Calculate the output points
        self.output_points = prod(self.branches)
        assert self.output_points == conf.loader.n_points
        assert self.features[-1] == conf.loader.n_features

        logger.debug(f"Generator output will be {self.output_points}")
        if conf.loader.n_points > self.output_points:
            raise RuntimeError(
                "Model cannot generate a sufficent number of points: "
                f"{conf.loader.n_points} < {self.output_points}"
            )

        self.tree = Tree(
            batch_size=conf.loader.batch_size,
            connect_all_ancestors=connect_all_ancestors,
            branches=self.branches,
            features=self.features,
        )

        self.dyn_hlvs_layers = nn.ModuleList(
            [
                DynHLVsLayer(
                    n_features=self.features[-1],
                    n_cond=self.n_cond,
                    n_global=n_global,
                    batch_size=self.batch_size,
                )
                for _ in self.features
            ]
        )

        self.branching_layers = nn.ModuleList(
            [
                BranchingLayer(
                    tree=self.tree,
                    level=level,
                    n_global=n_global,
                    n_cond=self.n_cond,
                    dim_red=self.dim_red_in_branching,
                    **branching_param,
                )
                for level in range(n_levels - 1)
            ]
        )

        if self.ancestor_mpl["n_mpl"] > 0:
            self.ancestor_conv_layers = nn.ModuleList(
                [
                    self.wrap_layer_init(ilevel, type="ac")
                    for ilevel in range(n_levels - 1)
                ]
            )

        if self.child_mpl["n_mpl"] > 0:
            self.child_conv_layers = nn.ModuleList(
                [
                    self.wrap_layer_init(ilevel, type="child")
                    for ilevel in range(n_levels - 1)
                ]
            )

        if self.final_layer_scaler:
            self.ftx_scaling = FtxScaleLayer(self.features[-1])

        # Allocate the Tensors used later to construct the batch
        self.presaved_batch: Optional[Batch] = None
        self.presaved_batch_indexing: Optional[torch.Tensor] = None

    def wrap_layer_init(self, ilevel, type: str):
        if type == "ac":
            conv_param = self.ancestor_mpl
        elif type == "child":
            conv_param = self.child_mpl
        else:
            raise Exception

        return MPLSeq(
            in_features=(
                self.features[ilevel + int(self.dim_red_in_branching)]
                if type == "ac"
                else self.features[ilevel + 1]
            ),
            out_features=self.features[ilevel + 1],
            n_cond=self.n_cond,
            n_global=self.n_global,
            # batch_size=self.batch_size,
            **conv_param,
        )

    def forward(
        self, random_vector: torch.Tensor, cond: torch.Tensor, n_pointsv
    ) -> Batch:
        batch_size = self.batch_size
        features = self.features
        device = random_vector.device
        n_levels = len(self.features)

        # Init the graph object
        graph_tree = TreeGraph(
            tftx=random_vector.reshape(batch_size, features[0]),
            global_features=torch.empty(
                batch_size, self.n_global, dtype=torch.float, device=device
            ),
            tree=self.tree,
        )
        cond = n_pointsv.reshape(-1, 1).float().clone()
        num_vec = n_pointsv.long()

        # model_plotter.save_tensor("input noise", graph_tree.tftx)
        print_dist("initial", graph_tree.tftx_by_level(0))
        # Do the branching
        for ilevel in range(n_levels - 1):
            # assert graph_tree.tftx.shape[1] == self.tree.features[ilevel]
            # assert graph_tree.tftx.shape[0] == (
            #     self.tree.tree_lists[ilevel][-1].idxs[-1] + 1
            # )
            # Assign the global features
            graph_tree.global_features = self.dyn_hlvs_layers[ilevel](
                x=graph_tree.tftx_by_level(ilevel)[..., : features[-1]],
                cond=cond,
                batch=self.tree.tbatch_by_level[ilevel][
                    self.tree.idxs_by_level[ilevel]
                ],
            )

            graph_tree = self.branching_layers[ilevel](graph_tree, cond)
            assert not torch.isnan(graph_tree.tftx).any()

            edge_index = self.tree.ancestor_ei(ilevel + 1)
            edge_attr = self.tree.ancestor_ea(ilevel + 1)
            batchidx = self.tree.tbatch_by_level[ilevel + 1]

            # Assign the global features
            graph_tree.global_features = self.dyn_hlvs_layers[ilevel](
                x=graph_tree.tftx_by_level(ilevel)[..., : features[-1]],
                cond=cond,
                batch=self.tree.tbatch_by_level[ilevel][
                    self.tree.idxs_by_level[ilevel]
                ],
            )

            print_dist("branching", graph_tree.tftx_by_level(ilevel + 1))
            assert (
                graph_tree.tftx.shape[1]
                == self.tree.features[ilevel + int(self.dim_red_in_branching)]
            )
            assert graph_tree.tftx.shape[0] == (
                self.tree.tree_lists[ilevel + 1][-1].idxs[-1] + 1
            )
            # model_plotter.save_tensor(
            #     f"branching output level{ilevel+1}",
            #     graph_tree.tftx_by_level(ilevel + 1),
            # )

            if self.ancestor_mpl["n_mpl"] > 0:
                graph_tree.tftx = self.ancestor_conv_layers[ilevel](
                    x=graph_tree.tftx,
                    cond=cond,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batchidx,
                    global_features=graph_tree.global_features,
                )
                print_dist("ac", graph_tree.tftx_by_level(ilevel + 1))
                assert graph_tree.tftx.shape[1] == self.tree.features[ilevel + 1]
                assert graph_tree.tftx.shape[0] == (
                    self.tree.tree_lists[ilevel + 1][-1].idxs[-1] + 1
                )
                # model_plotter.save_tensor(
                #     f"ancestor conv output level{ilevel+1}",
                #     graph_tree.tftx_by_level(ilevel + 1),
                # )

            if self.child_mpl["n_mpl"] > 0:
                graph_tree.tftx = self.child_conv_layers[ilevel](
                    x=graph_tree.tftx,
                    cond=cond,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batchidx,
                )
                assert graph_tree.tftx.shape[1] == self.tree.features[ilevel + 1]
                assert graph_tree.tftx.shape[0] == (
                    self.tree.tree_lists[ilevel + 1][-1].idxs[-1] + 1
                )

                # model_plotter.save_tensor(
                #     f"child conv output level{ilevel+1}",
                #     graph_tree.tftx_by_level(ilevel + 1),
                # )
            assert not torch.isnan(graph_tree.tftx).any()

        batch = self.construct_batch(graph_tree, num_vec)

        if self.final_layer_scaler:
            batch.x = self.ftx_scaling(batch.x)

        return batch

    def construct_batch(self, graph_tree: TreeGraph, num_vec: torch.Tensor):
        device = num_vec.device
        batch_size = self.batch_size
        n_features = self.features[-1]
        if self.presaved_batch is None:
            (
                self.presaved_batch,
                self.presaved_batch_indexing,
            ) = graph_tree.get_batch_skeleton()

        batch = self.presaved_batch.clone()
        batch.x = graph_tree.tftx_by_level(-1)[self.presaved_batch_indexing]

        x = batch.x.reshape(self.output_points, batch_size, n_features).transpose(
            0, 1
        )

        sel_point_idx = self.get_sel_idxs(x, num_vec)

        batch.x, batch.xnot = batch.x[sel_point_idx], batch.x[~sel_point_idx]
        batch.batch, batch.batchnot = (
            batch.batch[sel_point_idx],
            batch.batch[~sel_point_idx],
        )

        # Set the slice_dict to allow splitting the batch again
        batch._slice_dict["x"] = torch.concat(
            [torch.zeros(1, device=device), num_vec.cumsum(0)]
        )
        batch._slice_dict["xnot"] = torch.concat(
            [
                torch.zeros(1, device=device),
                (self.output_points - num_vec).cumsum(0),
            ]
        )
        batch._inc_dict["x"] = torch.zeros(self.batch_size, device=device)
        batch._inc_dict["xnot"] = torch.zeros(self.batch_size, device=device)
        batch.num_nodes = len(batch.x)
        batch.n_pointsv = num_vec
        batch.n_multihit = torch.zeros_like(num_vec)

        assert (
            len(batch.x) + len(batch.xnot) == self.output_points * self.batch_size
        )
        assert batch.x.shape[-1] == self.features[-1]
        assert batch.num_graphs == self.batch_size
        assert not torch.isnan(batch.x).any()

        return batch

    def get_sel_idxs(self, x: torch.Tensor, num_vec: torch.Tensor):
        device = x.device
        gidx = torch.zeros(x.shape[0] * x.shape[1], device=device).bool()
        shift = 0

        if self.pruning == "cut":
            for ne in num_vec:
                gidx[shift : shift + ne] = True
                shift += self.output_points
        elif self.pruning == "topk":
            for xe, ne in zip(x, num_vec):
                idxs = (
                    xe[..., conf.loader.x_ftx_energy_pos]
                    .topk(k=int(ne), dim=0, largest=True, sorted=False)
                    .indices
                ) + shift
                gidx[idxs] = True
                shift += self.output_points
        else:
            raise Exception

        return gidx

    def to(self, device):
        super().to(device)
        self.tree.to(device)
        return self


def print_dist(name, x):
    return
    if not conf.debug:
        return
    x = x.detach().cpu().numpy()
    print(
        f"{name}:\n\tmean\n\t{x.mean(0)} global {x.mean():.2f}\n"
        f"\tstd\n\t{x.std(0)} global {x.std():.2f}"
    )
