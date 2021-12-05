#


#
import torch

#


#
def loss_202012var_primitive(pl, var, std):
    lost = -pl + torch.exp((1 / std) * (-pl + var)) - 1
    return lost


def loss_202012var_solo(weights, yields):

    # step 1: portfolio daily P&L

    pls = torch.sum(weights * yields, dim=1)

    # step 3: portfolio daily VaR

    alpha = 0.01
    var_daily = torch.quantile(pls, q=alpha)

    # step 5: compute the loss

    std = torch.std(pls)
    lost = loss_202012var_primitive(pls, var_daily, std)
    lost = torch.sum(lost)

    #
    return lost


def loss_202012var_party(weights, yields):

    # step 1: portfolio daily P&L

    pls = torch.sum(weights * yields, dim=1)

    # step 2: portfolio overall P&L

    pls_ = torch.ones(size=pls.shape) + pls
    pl = torch.prod(pls_, dim=0)

    # step 3: portfolio daily VaR

    alpha = 0.01
    var_daily = torch.quantile(pls, q=alpha)

    # step 4: portfolio overall VaR

    var_overall = var_daily * torch.sqrt(torch.tensor(weights.shape[0], dtype=torch.float))

    # step 5: compute the loss

    std = torch.std(pls)
    lost = loss_202012var_primitive(pl, var_overall, std)

    #
    return lost
