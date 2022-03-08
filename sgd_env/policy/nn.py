from itertools import tee
from typing import List, Optional

import torch
from torch import nn

from dac4automlcomp.policy import DACPolicy, DeterministicPolicy


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def create_ffn_layers(
    layer_sizes: List[int], act: Optional[nn.Module] = None, bias: bool = True
) -> List[nn.Module]:
    layers: List[nn.Module] = []
    for idx, (layer1, layer2) in enumerate(pairwise(layer_sizes)):
        layers.append(nn.Linear(layer1, layer2, bias=bias))
        if act is not None:
            if idx < len(layer_sizes) - 2:
                layers.append(act())
    return layers


class AbstractNetworkPolicy(DeterministicPolicy, DACPolicy):
    def save(self, path):
        file_path = path / f"{self.__class__.__name__}.pt"
        torch.save(self, file_path)

    @classmethod
    def load(cls, path):
        file_path = path / f"{cls.__name__}.pt"
        return torch.load(file_path)


class FFN(AbstractNetworkPolicy):
    def __init__(
        self,
        hidden_layers: List[int],
        hidden_act: Optional[nn.Module] = None,
        output_act: Optional[nn.Module] = None,
        bias: bool = True,
    ):
        super().__init__()

        layers = create_ffn_layers([2, *hidden_layers, 1], hidden_act, bias=bias)
        if output_act is not None:
            layers.append(output_act())
        self.seq = nn.Sequential(*layers)

    def act(self, state):
        state = torch.tensor([state["step"], state["loss"].mean()])
        return self.seq(state).item()

    def reset(self, _):
        pass


class RNN(AbstractNetworkPolicy):
    def __init__(
        self,
        lstm_hidden_size: int,
        lstm_num_layers: int = 1,
        linear_hidden_layers: List[int] = [],
        hidden_act: Optional[nn.Module] = None,
        output_act: Optional[nn.Module] = None,
        bias: bool = True,
    ):
        super().__init__()

        self.hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(2, self.hidden_size, num_layers=lstm_num_layers, bias=bias)
        layers = create_ffn_layers(
            [self.hidden_size, *linear_hidden_layers, 1], hidden_act, bias=bias
        )
        if output_act is not None:
            layers.append(output_act())
        self.seq = nn.Sequential(*layers)

    def act(self, state):
        state = torch.tensor([[[state["step"], state["loss"].mean()]]])
        out, self.hidden = self.lstm(state, self.hidden)
        return self.seq(out).item()

    def reset(self, _):
        self.hidden = (
            torch.zeros(1, 1, self.hidden_size),
            torch.zeros(1, 1, self.hidden_size),
        )
