import torch
import torch.nn as nn


class RevON(nn.Module):
    def __init__(
        self, num_features: int, eps=1e-5, affine=True, subtract_last=False
    ):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevON, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, mask=None):
        if mode == "norm":
            self._get_statistics(x, mask)
            x = self._normalize(x, mask)
        elif mode == "norm-fore":
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, mask=None):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
            self.stdev = torch.sqrt(
                torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)
                + self.eps
            ).detach()
        else:
            if mask is not None:
                valid_sum = x.sum(dim=dim2reduce, keepdim=True)
                valid_count = mask.sum(dim=dim2reduce, keepdim=True)
                self.mean = valid_sum / (valid_count + self.eps)
                self.mean = self.mean.detach()

                squared_deviation = ((x - self.mean) ** 2) * mask
                variance = squared_deviation.sum(
                    dim=dim2reduce, keepdim=True
                ) / (valid_count + self.eps)
                self.stdev = torch.sqrt(variance + self.eps).detach()
            else:
                self.mean = torch.mean(
                    x, dim=dim2reduce, keepdim=True
                ).detach()
                self.stdev = torch.sqrt(
                    torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)
                    + self.eps
                ).detach()

    def _normalize(self, x, mask=None):
        if mask is not None:
            x = x - self.mean
            x = x / self.stdev
            if self.affine:
                x = x * self.affine_weight
                x = x + self.affine_bias
            return x * mask
        else:
            if self.subtract_last:
                x = x - self.last
            else:
                x = x - self.mean
            x = x / self.stdev
            if self.affine:
                x = x * self.affine_weight
                x = x + self.affine_bias
            return x

    def _denormalize(self, x, mask=None):
        if mask is not None:
            if self.affine:
                x = x - self.affine_bias
                x = x / (self.affine_weight + self.eps * self.eps)
            x = x * self.stdev
            if self.subtract_last:
                x = x + self.last
            else:
                x = x + self.mean
            return x * mask
        else:
            if self.affine:
                x = x - self.affine_bias
                x = x / (self.affine_weight + self.eps * self.eps)
            x = x * self.stdev
            if self.subtract_last:
                x = x + self.last
            else:
                x = x + self.mean
            return x
