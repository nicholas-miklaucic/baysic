"""PyTorch equivalent for scipy's linear interpolation."""

import torch

class LinearSpline(torch.nn.Module):
    """A linear spline."""
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        super().__init__()
        x_sort = torch.argsort(x)
        self.x = x[x_sort]
        self.y = y[x_sort]
        self.x = torch.cat([self.x[..., [0]], self.x, self.x[..., [-1]]], dim=-1)
        self.y = torch.cat([self.y[..., [0]], self.y, self.y[..., [-1]]], dim=-1)

    def forward(self, x_new: torch.Tensor) -> torch.Tensor:
        i = torch.searchsorted(self.x[..., 1:-1], x_new, right=True)

        x_left, x_right = self.x[i], self.x[i+1]
        y_left, y_right = self.y[i], self.y[i+1]

        weight = torch.where(x_left == x_right, 0, (x_new - x_left) / (x_right - x_left))
        return torch.lerp(y_left, y_right, weight)
    


if __name__ == '__main__':
    from scipy.interpolate import interp1d
    import numpy as np

    x = torch.linspace(0, 1, 100)
    y = torch.randn(100)
    y[0] = 0
    y[1] = 1
    new_data = torch.rand(100)

    spl = LinearSpline(x, y)(new_data)
    sci = interp1d(x.numpy(), y.numpy())(new_data.numpy())

    assert np.all(np.abs(spl.numpy() - sci) <= 1e-3)
    print('Works!')