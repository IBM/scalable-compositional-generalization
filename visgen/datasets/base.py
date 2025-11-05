import itertools
from abc import ABC, abstractmethod
from pathlib import Path
from random import sample
from typing import List, Tuple, Union
import matplotlib.pyplot as plt, numpy as np, torch
from ortools.sat.python import cp_model
from torch.utils.data import Dataset, Subset


class BaseDataset(Dataset, ABC):
    "\n    Base class for all datasets.\n"

    def __init__(
        self,
        path: Union[Path, str],
        dataset_subset: str = None,
        targets: List[str] = None,
        split_attributes: List[str] = None,
        split: str = "composition",
        split_difficulty: float = None,
        train: bool = True,
        test_complement: bool = True,
        shuffle: bool = True,
        downsample: int = 0,
        upsample_strategy: str = None,
        *args,
        **kwargs,
    ):
        "\n        :param dataset_path: Path to the dataset.\n        :param dataset_subset: Specify a subset or version of the dataset, e.g. a constellation for the IRAVEN dataset.\n        :param target_names: Names of the attributes that are to be predicted.\n        :param split_attributes: Names of the attributes that are relevant for splitting the dataset into training and\n                                 testing sets.\n        :param split: Split of the dataset. One of 'random', 'composition', 'interpolation' and 'extrapolation'.\n                      See Schott et al. (2022) for more details.\n        :param split_difficulty: A number between 0 and 1 for the fraction of combinations that will be excluded. The\n        fraction is with respect to the maximum number of combinations that can be excluded for a certain split.\n        :param train: If True, the dataset will be used for training.\n        :param test_complement: If True, the test set is the complement of the training set. Otherwise, the test set\n        consists of the combinations that are excluded across all difficulty levels.\n        :param shuffle: If True, the dataset will be shuffled.\n        :param downsample: If > 0, the dataset will be downsampled to the specified number of samples.\n"
        super().__init__()
        self._split = split
        self._split_difficulty = split_difficulty
        self._test_complement = test_complement
        self.ood_val_combinations = None
        if split_attributes is None:
            split_attributes = targets
        if targets is not None:
            targets = targets.split("_")
        split_attributes = split_attributes.split("_")
        self._dataset_images, self._dataset_targets = self._load_data(
            path, dataset_subset
        )
        self.original_size = len(self._dataset_targets)
        if split_attributes is not None:
            self._split_data(
                split=split,
                split_attributes=split_attributes,
                split_difficulty=split_difficulty,
                train=train,
                **kwargs,
            )
        if targets is not None:
            target_indices = [self._attribute_to_index(target) for target in targets]
            target_indices.sort()
            self._split_attribute_indices = []
            num_attributes = self._dataset_targets.shape[-1]
            excluded_indices = [
                i for i in range(num_attributes) if i not in target_indices
            ]
            for attribute in split_attributes:
                idx = self._attribute_to_index(attribute)
                if idx in target_indices:
                    n_smaller = (np.array(excluded_indices) < idx).sum()
                    new_idx = idx - n_smaller
                    self._split_attribute_indices.append(new_idx)
            self._split_attribute_indices.sort()
            self._dataset_targets = self._dataset_targets[..., target_indices]
            if self._attribute_values is not None:
                self._attribute_values = [
                    self._attribute_values[i] for i in target_indices
                ]
        if train and upsample_strategy:
            current_size = len(self._dataset_images)
            num_additional_samples = self.original_size - current_size
            if upsample_strategy == "balanced":
                incl_pairs = {
                    tuple(x) for x in np.unique(self._dataset_targets, axis=0).squeeze()
                }
                excl_pairs = (
                    set(itertools.product(*self._attribute_values)) - incl_pairs
                )
                tuples_array = np.array(list(excl_pairs))
                np.where(
                    np.logical_or.reduce(
                        [
                            np.isin(self._dataset_targets[..., i], tuples_array[:, i])
                            for i in range(tuples_array.shape[1])
                        ]
                    )
                )[0]
                extra_samples = sample(
                    list(range(current_size)), num_additional_samples
                )
            elif upsample_strategy == "random":
                extra_samples = sample(
                    list(range(current_size)), num_additional_samples
                )
            self._dataset_targets = np.concatenate(
                [self._dataset_targets, self._dataset_targets[extra_samples]]
            )
            self._dataset_samples = np.concatenate(
                [self._dataset_images, self._dataset_images[extra_samples]]
            )
        if shuffle:
            shuffling = torch.randperm(self._dataset_images.shape[0])
            self._dataset_images = self._dataset_images[shuffling]
            self._dataset_targets = self._dataset_targets[shuffling]
        if train and downsample:
            self._dataset_images = self._dataset_images[:downsample]
            self._dataset_targets = self._dataset_targets[:downsample]

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError("Subclasses must implement __getitem__()")

    @abstractmethod
    def _attribute_to_index(self, attribute: str) -> int:
        "\n        Returns the index of the attribute.\n        :param attribute: Name of the attribute.\n"
        raise NotImplementedError("Subclasses must implement _attribute_to_index()")

    @abstractmethod
    def _load_data(
        self, dataset_path: Union[Path, str], dataset_subset: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        "\n        Loads the dataset into a tuple: (images, targets). Targets are assumed to be of dimension (N, M) where N is the\n        number of objects and M is the number of attributes.\n        :param dataset_path: Path to the dataset.\n        :param dataset_subset: Subset or version of the dataset, e.g. a constellation for the IRAVEN dataset.\n"
        raise NotImplementedError("Subclasses must implement _load_data()")

    def __len__(self):
        return len(self._dataset_targets)

    def _get_attribute_values(self) -> List[List[int]]:
        num_attributes = self._dataset_targets.shape[-1]
        object_attributes = self._dataset_targets.reshape(-1, num_attributes)
        padding_mask = ~np.all(object_attributes == -1, axis=1)
        object_attributes = object_attributes[padding_mask]
        return [
            np.unique(object_attributes[:, i]).tolist() for i in range(num_attributes)
        ]

    def _get_attribute_difficulties(
        self, split_attribute_indices, split_difficulty, train
    ):
        factor_sizes = [self._factor_sizes[idx] for idx in split_attribute_indices]
        if train or self._test_complement:
            difficulties, self.actual_difficulty = self._optimize_difficulties(
                factor_sizes, split_difficulty
            )
        else:
            difficulties = [1 for _ in split_attribute_indices]
        return difficulties

    @staticmethod
    def _get_threshold_values(attribute_values, split_attribute_indices, difficulties):
        return [
            attribute_values[attr_idx][-difficulty]
            for (attr_idx, difficulty) in zip(split_attribute_indices, difficulties)
        ]

    @staticmethod
    def _get_mask_from_combinations(target_values, included_combinations):
        return np.array(
            [
                all(tuple(val) in included_combinations for val in vals)
                for vals in target_values
            ]
        )

    def _get_mask(self, included_combinations, cartesian_product, indices, train):
        self._included_combinations = set(map(tuple, included_combinations))
        if not train:
            cartesian_product_set = set(map(tuple, cartesian_product))
            self._included_combinations = cartesian_product_set.difference(
                self._included_combinations
            )
        target_values = self._dataset_targets[..., indices]
        return self._get_mask_from_combinations(
            target_values, self._included_combinations
        )

    def _optimize_difficulties(
        self, factor_sizes, split_difficulty
    ) -> (List[int], float):
        model = cp_model.CpModel()
        min_num_vals = 2 if self._split == "interpolation" else 1
        difficulties = [
            model.new_int_var(1, size - min_num_vals, f"d_{i}")
            for (i, size) in enumerate(factor_sizes)
        ]
        if "composition" in self._split:
            max_excluded = np.product([size - 1 for size in factor_sizes])
            num_excluded = model.new_int_var(0, max_excluded, "num_excluded")
            model.add_multiplication_equality(num_excluded, difficulties)
        else:
            total = np.product(factor_sizes)
            if self._split == "interpolation":
                max_excluded = total - 2 ** len(factor_sizes)
            else:
                max_excluded = total - 1
            sizes_minus_difficulties = [
                size - difficulty
                for (size, difficulty) in zip(factor_sizes, difficulties)
            ]
            num_included = model.new_int_var(0, total, "num_excluded")
            model.add_multiplication_equality(num_included, sizes_minus_difficulties)
            num_excluded = model.new_int_var(0, max_excluded, "num_excluded")
            model.add(num_excluded == total - num_included)
        percent_excluded = model.new_int_var(0, 100, "percent_excluded")
        model.add_division_equality(percent_excluded, num_excluded * 100, max_excluded)
        difficulty = int(100 * split_difficulty)
        abs_diff = model.new_int_var(0, 100, "abs_diff")
        model.add(abs_diff >= percent_excluded - difficulty)
        model.add(abs_diff >= difficulty - percent_excluded)
        relative_difficulties = [
            model.new_int_var(0, 100, f"rel_diff_{i}") for i in range(len(factor_sizes))
        ]
        for i, (difficulty, size) in enumerate(zip(difficulties, factor_sizes)):
            model.add_division_equality(
                relative_difficulties[i], difficulty * 100, size
            )
        max_rel_diff = model.new_int_var(0, 100, "max_rel_diff")
        min_rel_diff = model.new_int_var(0, 100, "min_rel_diff")
        model.add_max_equality(max_rel_diff, relative_difficulties)
        model.add_min_equality(min_rel_diff, relative_difficulties)
        rel_diff_range = model.new_int_var(0, 100, "rel_diff_range")
        model.add(rel_diff_range == max_rel_diff - min_rel_diff)
        model.minimize(5 * abs_diff + rel_diff_range)
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            raise RuntimeError("No feasible solution found.")
        difficulty_values = [solver.value(d) for d in difficulties]
        final_num_excluded = solver.value(num_excluded)
        actual_difficulty = final_num_excluded / max_excluded
        return difficulty_values, actual_difficulty

    def _pairwise_composition(
        self,
        split_attributes: List[str],
        attribute_values: List[List[int]],
        split_difficulty: float,
        train: bool,
        **args,
    ):
        indices = [self._attribute_to_index(attr) for attr in split_attributes]
        difficulties = self._get_attribute_difficulties(
            indices, split_difficulty, train
        )
        threshold_values = self._get_threshold_values(
            attribute_values, indices, difficulties
        )
        split_attribute_values = [attribute_values[i] for i in indices]
        cartesian_product = np.array(list(itertools.product(*split_attribute_values)))
        included_combinations = cartesian_product[
            np.any(cartesian_product < np.array(threshold_values), axis=1)
        ]
        self.volume = included_combinations.shape[0] / cartesian_product.shape[0]
        return self._get_mask(included_combinations, cartesian_product, indices, train)

    def _orthotopic_composition(
        self,
        split_attributes: List[str],
        attribute_values: List[List[int]],
        train: bool,
        c: int = 1,
        split_difficulty: float = None,
        attr_difficulty: List[float] = None,
        **kwargs,
    ):
        indices = [self._attribute_to_index(attr) for attr in split_attributes]
        split_attribute_values = [attribute_values[i] for i in indices]
        if attr_difficulty is not None:
            if attr_difficulty[0] < 1:
                att_vals = [len(att) - 1 for att in split_attribute_values]
                threshold_values = np.ceil(
                    att_vals - np.multiply(attr_difficulty, att_vals)
                )
            else:
                threshold_values = attr_difficulty
        elif split_difficulty is not None:
            difficulties = self._get_attribute_difficulties(
                indices, split_difficulty, train
            )
            threshold_values = self._get_threshold_values(
                attribute_values, indices, difficulties
            )
        else:
            raise ValueError("p and c cannot be both None!")
        cartesian_product = np.array(list(itertools.product(*split_attribute_values)))
        print(threshold_values)
        included_combinations = cartesian_product[
            np.sum(cartesian_product >= threshold_values, axis=1) <= c
        ]
        self.volume = included_combinations.shape[0] / cartesian_product.shape[0]
        print(self.volume)
        return self._get_mask(included_combinations, cartesian_product, indices, train)

    def _split_data(
        self,
        split_attributes: List[str],
        split: str,
        train: bool,
        split_difficulty: Union[float, int] = None,
        **kwargs,
    ) -> None:
        if split_difficulty < 0 or split_difficulty > 1:
            raise ValueError("split_difficulty needs to be a number between 0 and 1.")
        self._split_attributes = sorted(
            split_attributes, key=lambda attr: self._attribute_to_index(attr)
        )
        self._attribute_values = self._get_attribute_values()
        self._factor_sizes = [len(attr_vals) for attr_vals in self._attribute_values]
        split_methods = {
            "composition": self._pairwise_composition,
            "general_composition": self._orthotopic_composition,
        }
        if split in split_methods:
            mask = split_methods[split](
                split_attributes=self._split_attributes,
                attribute_values=self._attribute_values,
                split_difficulty=split_difficulty,
                train=train,
                **kwargs,
            )
        else:
            raise ValueError(f"Split {split} not recognized.")
        self._dataset_images = self._dataset_images[mask]
        self._dataset_targets = self._dataset_targets[mask]

    def ood_validation_split(self, num_ood_val) -> (Subset, Subset):
        "Out-of-domain validation split."
        self.ood_val_combinations = sample(
            sorted(self._included_combinations), num_ood_val
        )
        self._included_combinations = self._included_combinations.difference(
            self.ood_val_combinations
        )
        target_values = self._dataset_targets[..., self._split_attribute_indices]
        train_mask = self._get_mask_from_combinations(
            target_values, self._included_combinations
        )
        train_indices = np.where(train_mask)[0]
        train_set = Subset(self, train_indices)
        ood_masks = [
            self._get_mask_from_combinations(target_values, [comb])
            for comb in self.ood_val_combinations
        ]
        ood_indices = [np.where(ood_mask)[0] for ood_mask in ood_masks]
        ood_sets = [Subset(self, ind) for ind in ood_indices]
        return train_set, ood_sets

    def plot_attribute_histogram(self, save_path=None):
        indices = self._split_attribute_indices
        dataset_targets = self._dataset_targets[..., indices]
        x, y = dataset_targets[..., 0].reshape(-1), dataset_targets[..., 1].reshape(-1)
        bins = [len(self._attribute_values[idx]) for idx in indices]
        range_ = [
            (np.min(arr), np.max(arr))
            for arr in np.array(self._attribute_values, dtype=object)[indices]
        ]
        fig, ax = plt.subplots()
        hist, xbins, ybins, im = ax.hist2d(x, y, bins=bins, density=False, range=range_)
        ax.set_title("Attribute Histogram")
        ax.set_xlabel(self._split_attributes[0])
        ax.set_ylabel(self._split_attributes[1])
        for i in range(len(ybins) - 1):
            for j in range(len(xbins) - 1):
                color = "w" if hist.T[i][j] <= np.mean(hist) / 2 else "k"
                ax.text(
                    xbins[j] + 0.5,
                    ybins[i] + 0.5,
                    int(hist.T[i, j]),
                    color=color,
                    ha="center",
                    va="center",
                )
        plt.xticks(xbins), plt.yticks(ybins)
        if save_path is not None:
            plt.savefig(save_path)
        plt.close()
        return fig
