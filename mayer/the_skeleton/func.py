#


#
import numpy
import pandas
from scipy import stats

from sklearn.linear_model import LinearRegression

from torch import nn


#


#
def standard_layer_multiplier(j):
    if j == 0:
        return 1
    elif j == 1:
        return 2
    elif j == 2:
        return 5
    elif j == 3:
        return 12
    else:
        raise Exception("Make Params: Layer Multiplier: Standard: too big 'j'!")


def tidal_wave_layer_multiplier(j):
    if j == 0:
        return 128
    elif j == 1:
        return 64
    elif j == 2:
        return 32
    elif j == 3:
        return 16
    else:
        raise Exception("Make Params: Layer Multiplier: Tidal Wave: too big 'j'!")


def c_less(array, c):
    q = numpy.quantile(array, q=c)
    return array[array <= q]


def v_less(array, v):
    return array[array <= v]


def make_params(layer_type, n_multiplier, verse, depth, act, drops, optima, lr, loss, eps):
    nn_kwargs = {'layers': [layer_type] * depth + [nn.Linear]}

    if verse == 'inc':
        nn_kwargs['layers_dimensions'] = [standard_layer_multiplier(j) * n_multiplier
                                          for j in range(depth)] + [2]
    if verse == 'dec':
        nn_kwargs['layers_dimensions'] = [tidal_wave_layer_multiplier(j) * (2 ** (n_multiplier - 1))
                                          for j in range(depth)] + [2]
    if verse == 'sta':
        nn_kwargs['layers_dimensions'] = [10 * n_multiplier] * depth + [2]

    nn_kwargs['layers_kwargs'] = [{}, {}, {}, {}]
    nn_kwargs['activators'] = [act] * (depth + 1)

    if isinstance(drops, float) or isinstance(drops, int):
        nn_kwargs['interdrops'] = [drops] * depth + [None]
    else:
        nn_kwargs['interdrops'] = drops + [None]

    nn_kwargs['optimiser'] = optima
    nn_kwargs['optimiser_kwargs'] = {'lr': lr}
    nn_kwargs['loss_function'] = loss
    nn_kwargs['epochs'] = eps

    return nn_kwargs


def simple_search_report(hero_cum_left, c0_cum_left, c1_cum_left, c2_cum_left, synth_cum_left,
                         hero_yields_left, static_yields_left, component1_yields_left, component2_yields_left,
                         syn_yields_left, bench_yields_left,
                         hero_cum_right, c0_cum_right, c1_cum_right, c2_cum_right, synth_cum_right,
                         hero_yields_right, static_yields_right, component1_yields_right, component2_yields_right,
                         syn_yields_right, bench_yields_right
                         ):
    perf_left = pandas.DataFrame(
        data={'Annual Yield': [numpy.power(hero_cum_left[-1], (12 / hero_yields_left.shape[0])) - 1,
                               numpy.power(c0_cum_left[-1], (12 / hero_yields_left.shape[0])) - 1,
                               numpy.power(c1_cum_left[-1], (12 / hero_yields_left.shape[0])) - 1,
                               numpy.power(c2_cum_left[-1], (12 / hero_yields_left.shape[0])) - 1,
                               numpy.power(synth_cum_left[-1], (12 / hero_yields_left.shape[0])) - 1],
              'Monthly Yield': [numpy.power(hero_cum_left[-1], (1 / hero_yields_left.shape[0])) - 1,
                                numpy.power(c0_cum_left[-1], (1 / hero_yields_left.shape[0])) - 1,
                                numpy.power(c1_cum_left[-1], (1 / hero_yields_left.shape[0])) - 1,
                                numpy.power(c2_cum_left[-1], (1 / hero_yields_left.shape[0])) - 1,
                                numpy.power(synth_cum_left[-1], (1 / hero_yields_left.shape[0])) - 1],
              'St Dev': [numpy.std(hero_yields_left),
                         numpy.std(static_yields_left),
                         numpy.std(component1_yields_left),
                         numpy.std(component2_yields_left),
                         numpy.std(syn_yields_left)],
              'VaR 99': [numpy.quantile(hero_yields_left, q=0.01),
                         numpy.quantile(static_yields_left, q=0.01),
                         numpy.quantile(component1_yields_left, q=0.01),
                         numpy.quantile(component2_yields_left, q=0.01),
                         numpy.quantile(syn_yields_left, q=0.01)]},
        index=['Hero', 'Static', 'C1', 'C2', 'Synth'])

    perf_right = pandas.DataFrame(
        data={'Annual Yield': [numpy.power(hero_cum_right[-1], (12 / hero_yields_right.shape[0])) - 1,
                               numpy.power(c0_cum_right[-1], (12 / hero_yields_right.shape[0])) - 1,
                               numpy.power(c1_cum_right[-1], (12 / hero_yields_right.shape[0])) - 1,
                               numpy.power(c2_cum_right[-1], (12 / hero_yields_right.shape[0])) - 1,
                               numpy.power(synth_cum_right[-1], (12 / hero_yields_right.shape[0])) - 1],
              'Monthly Yield': [numpy.power(hero_cum_right[-1], (1 / hero_yields_right.shape[0])) - 1,
                                numpy.power(c0_cum_right[-1], (1 / hero_yields_right.shape[0])) - 1,
                                numpy.power(c1_cum_right[-1], (1 / hero_yields_right.shape[0])) - 1,
                                numpy.power(c2_cum_right[-1], (1 / hero_yields_right.shape[0])) - 1,
                                numpy.power(synth_cum_right[-1], (1 / hero_yields_right.shape[0])) - 1],
              'St Dev': [numpy.std(hero_yields_right),
                         numpy.std(static_yields_right),
                         numpy.std(component1_yields_right),
                         numpy.std(component2_yields_right),
                         numpy.std(syn_yields_right)],
              'VaR 99': [numpy.quantile(hero_yields_right, q=0.01),
                         numpy.quantile(static_yields_right, q=0.01),
                         numpy.quantile(component1_yields_right, q=0.01),
                         numpy.quantile(component2_yields_right, q=0.01),
                         numpy.quantile(syn_yields_right, q=0.01)]},
        index=['Hero', 'Static', 'C1', 'C2', 'Synth'])

    return perf_left, perf_right


def extended_risk_report(hero_cum_left, c0_cum_left, c1_cum_left, c2_cum_left, synth_cum_left,
                         hero_yields_left, static_yields_left, component1_yields_left, component2_yields_left,
                         syn_yields_left, bench_yields_left,
                         hero_cum_right, c0_cum_right, c1_cum_right, c2_cum_right, synth_cum_right,
                         hero_yields_right, static_yields_right, component1_yields_right, component2_yields_right,
                         syn_yields_right, bench_yields_right
                         ):
    perf_left = pandas.DataFrame(
        data={'Annual Yield': [numpy.power(hero_cum_left[-1], (12 / hero_yields_left.shape[0])) - 1,
                               numpy.power(c0_cum_left[-1], (12 / hero_yields_left.shape[0])) - 1,
                               numpy.power(c1_cum_left[-1], (12 / hero_yields_left.shape[0])) - 1,
                               numpy.power(c2_cum_left[-1], (12 / hero_yields_left.shape[0])) - 1,
                               numpy.power(synth_cum_left[-1], (12 / hero_yields_left.shape[0])) - 1],
              'St Dev': [numpy.std(hero_yields_left),
                         numpy.std(static_yields_left),
                         numpy.std(component1_yields_left),
                         numpy.std(component2_yields_left),
                         numpy.std(syn_yields_left)],
              'VaR 95': [numpy.quantile(hero_yields_left, q=0.05),
                         numpy.quantile(static_yields_left, q=0.05),
                         numpy.quantile(component1_yields_left, q=0.05),
                         numpy.quantile(component2_yields_left, q=0.05),
                         numpy.quantile(syn_yields_left, q=0.05)],
              'VaR 99': [numpy.quantile(hero_yields_left, q=0.01),
                         numpy.quantile(static_yields_left, q=0.01),
                         numpy.quantile(component1_yields_left, q=0.01),
                         numpy.quantile(component2_yields_left, q=0.01),
                         numpy.quantile(syn_yields_left, q=0.01)],
              'C-Mean 95': [c_less(hero_yields_left, c=0.05).mean(),
                            c_less(static_yields_left, c=0.05).mean(),
                            c_less(component1_yields_left, c=0.05).mean(),
                            c_less(component2_yields_left, c=0.05).mean(),
                            c_less(syn_yields_left, c=0.05).mean()],
              'C-Mean 99': [c_less(hero_yields_left, c=0.01).mean(),
                            c_less(static_yields_left, c=0.01).mean(),
                            c_less(component1_yields_left, c=0.01).mean(),
                            c_less(component2_yields_left, c=0.01).mean(),
                            c_less(syn_yields_left, c=0.01).mean()],
              'C-Median 95': [numpy.median(c_less(hero_yields_left, c=0.05)),
                              numpy.median(c_less(static_yields_left, c=0.05)),
                              numpy.median(c_less(component1_yields_left, c=0.05)),
                              numpy.median(c_less(component2_yields_left, c=0.05)),
                              numpy.median(c_less(syn_yields_left, c=0.05))],
              'C-Median 99': [numpy.median(c_less(hero_yields_left, c=0.01)),
                              numpy.median(c_less(static_yields_left, c=0.01)),
                              numpy.median(c_less(component1_yields_left, c=0.01)),
                              numpy.median(c_less(component2_yields_left, c=0.01)),
                              numpy.median(c_less(syn_yields_left, c=0.01))],
              'C-Mode 95': [numpy.nan,  # stats.mode(c_less(hero_yields_left, c=0.05), axis=None),
                            numpy.nan,  # stats.mode(c_less(bench_yields_left, c=0.05), axis=None),
                            numpy.nan,  # stats.mode(c_less(component1_yields_left, c=0.05), axis=None),
                            numpy.nan,  # stats.mode(c_less(component2_yields_left, c=0.05), axis=None),
                            numpy.nan],  # stats.mode(c_less(syn_yields_left, c=0.05), axis=None)],
              'C-Mode 99': [numpy.nan,  # stats.mode(c_less(hero_yields_left, c=0.01), axis=None),
                            numpy.nan,  # numpy.nan,  # stats.mode(c_less(bench_yields_left, c=0.01), axis=None),
                            numpy.nan,  # stats.mode(c_less(component1_yields_left, c=0.01), axis=None),
                            numpy.nan,  # stats.mode(c_less(component2_yields_left, c=0.01), axis=None),
                            numpy.nan],  # stats.mode(c_less(syn_yields_left, c=0.01), axis=None)],
              },
        index=['Hero', 'Static', 'C1', 'C2', 'Synth'])
    perf_right = pandas.DataFrame(
        data={'Annual Yield': [numpy.power(hero_cum_right[-1], (12 / hero_yields_right.shape[0])) - 1,
                               numpy.power(c0_cum_right[-1], (12 / hero_yields_right.shape[0])) - 1,
                               numpy.power(c1_cum_right[-1], (12 / hero_yields_right.shape[0])) - 1,
                               numpy.power(c2_cum_right[-1], (12 / hero_yields_right.shape[0])) - 1,
                               numpy.power(synth_cum_right[-1], (12 / hero_yields_right.shape[0])) - 1],
              'St Dev': [numpy.std(hero_yields_right),
                         numpy.std(static_yields_right),
                         numpy.std(component1_yields_right),
                         numpy.std(component2_yields_right),
                         numpy.std(syn_yields_right)],
              'VaR 95': [numpy.quantile(hero_yields_right, q=0.05),
                         numpy.quantile(static_yields_right, q=0.05),
                         numpy.quantile(component1_yields_right, q=0.05),
                         numpy.quantile(component2_yields_right, q=0.05),
                         numpy.quantile(syn_yields_right, q=0.05)],
              'VaR 99': [numpy.quantile(hero_yields_right, q=0.01),
                         numpy.quantile(static_yields_right, q=0.01),
                         numpy.quantile(component1_yields_right, q=0.01),
                         numpy.quantile(component2_yields_right, q=0.01),
                         numpy.quantile(syn_yields_right, q=0.01)],
              'C-Mean 95': [c_less(hero_yields_right, c=0.05).mean(),
                            c_less(static_yields_right, c=0.05).mean(),
                            c_less(component1_yields_right, c=0.05).mean(),
                            c_less(component2_yields_right, c=0.05).mean(),
                            c_less(syn_yields_right, c=0.05).mean()],
              'C-Mean 99': [c_less(hero_yields_right, c=0.01).mean(),
                            c_less(static_yields_right, c=0.01).mean(),
                            c_less(component1_yields_right, c=0.01).mean(),
                            c_less(component2_yields_right, c=0.01).mean(),
                            c_less(syn_yields_right, c=0.01).mean()],
              'C-Median 95': [numpy.median(c_less(hero_yields_right, c=0.05)),
                              numpy.median(c_less(static_yields_right, c=0.05)),
                              numpy.median(c_less(component1_yields_right, c=0.05)),
                              numpy.median(c_less(component2_yields_right, c=0.05)),
                              numpy.median(c_less(syn_yields_right, c=0.05))],
              'C-Median 99': [numpy.median(c_less(hero_yields_right, c=0.01)),
                              numpy.median(c_less(static_yields_right, c=0.01)),
                              numpy.median(c_less(component1_yields_right, c=0.01)),
                              numpy.median(c_less(component2_yields_right, c=0.01)),
                              numpy.median(c_less(syn_yields_right, c=0.01))],
              'C-Mode 95': [numpy.nan,  # stats.mode(c_less(hero_yields_right, c=0.05), axis=None),
                            numpy.nan,  # stats.mode(c_less(bench_yields_right, c=0.05), axis=None),
                            numpy.nan,  # stats.mode(c_less(component1_yields_right, c=0.05), axis=None),
                            numpy.nan,  # stats.mode(c_less(component2_yields_right, c=0.05), axis=None),
                            numpy.nan],  # stats.mode(c_less(syn_yields_right, c=0.05), axis=None)],
              'C-Mode 99': [numpy.nan,  # stats.mode(c_less(hero_yields_right, c=0.01), axis=None),
                            numpy.nan,  # stats.mode(c_less(bench_yields_right, c=0.01), axis=None),
                            numpy.nan,  # stats.mode(c_less(component1_yields_right, c=0.01), axis=None),
                            numpy.nan,  # stats.mode(c_less(component2_yields_right, c=0.01), axis=None),
                            numpy.nan],  # stats.mode(c_less(syn_yields_right, c=0.01), axis=None)],
              },
        index=['Hero', 'Static', 'C1', 'C2', 'Synth'])

    return perf_left, perf_right


def sharpe_ratio(hero, bench):
    return (hero - bench).mean() / (hero - bench).std()


def downside_semivariance(array):
    mean = array.mean()
    return numpy.array([numpy.power(x, 2) for x in v_less(array, mean) - mean]).sum() / (array.shape[0] - 1)


def sortino_ratio(hero, bench):
    return (hero - bench).mean() / numpy.power(downside_semivariance(hero - bench), 0.5)


def alpha(hero, bench):
    y = hero.ravel()
    x = bench.reshape(-1, 1)

    print(y.shape)
    print(x.shape)

    model = LinearRegression()
    model.fit(X=x, y=y)

    return model.intercept_


def beta(hero, bench):
    y = hero.ravel()
    x = bench.reshape(-1, 1)

    print(y.shape)
    print(x.shape)

    model = LinearRegression()
    model.fit(X=x, y=y)

    return model.coef_[0]


def max_drawdown_value(array):
    array_ = (1 + array).cumprod()
    draws = []
    for j in range(array_.shape[0] - 1):
        mores = []
        i = 1
        if (j + i) < array_.shape[0]:
            if array_[j + i] < array_[j]:
                mores.append(array_[j + i])
                i += 1
                rush = True
                while (j + i) < array_.shape[0] and rush:
                    if array_[j + i] < array_[j]:
                        mores.append(array_[j + i])
                        i += 1
                    else:
                        rush = False
        if len(mores) != 0:
            minimal = numpy.min(mores)
            draw = (minimal / array_[j]) - 1
            draws.append(draw)
    if len(draws) != 0:
        return numpy.min(draws)
    else:
        return numpy.nan


def max_drawdown_length(array):
    array_ = (1 + array).cumprod()
    draws = []
    for j in range(array_.shape[0] - 1):
        draw = 0
        i = 1
        if (j + i) < array_.shape[0]:
            if array_[j + i] < array_[j]:
                draw = i
                i += 1
                rush = True
                while (j + i) < array_.shape[0] and rush:
                    if array_[j + i] < array_[j]:
                        draw = i
                        i += 1
                    else:
                        rush = False
        draws.append(draw)
    return numpy.max(draws)


def positive_deals_rate(array):
    return (array > 0).sum() / array.shape[0]


def hedge_management_report(hero_cum_left, c0_cum_left, c1_cum_left, c2_cum_left, synth_cum_left,
                            hero_yields_left, static_yields_left, component1_yields_left, component2_yields_left,
                            syn_yields_left, bench_yields_left,
                            hero_cum_right, c0_cum_right, c1_cum_right, c2_cum_right, synth_cum_right,
                            hero_yields_right, static_yields_right, component1_yields_right, component2_yields_right,
                            syn_yields_right, bench_yields_right
                            ):
    perf_left = pandas.DataFrame(
        data={'Annual Yield': [numpy.power(hero_cum_left[-1], (12 / hero_yields_left.shape[0])) - 1,
                               numpy.power(c0_cum_left[-1], (12 / hero_yields_left.shape[0])) - 1,
                               numpy.power(c1_cum_left[-1], (12 / hero_yields_left.shape[0])) - 1,
                               numpy.power(c2_cum_left[-1], (12 / hero_yields_left.shape[0])) - 1,
                               numpy.power(synth_cum_left[-1], (12 / hero_yields_left.shape[0])) - 1],
              'Monthly Yield': [numpy.power(hero_cum_left[-1], (1 / hero_yields_left.shape[0])) - 1,
                                numpy.power(c0_cum_left[-1], (1 / hero_yields_left.shape[0])) - 1,
                                numpy.power(c1_cum_left[-1], (1 / hero_yields_left.shape[0])) - 1,
                                numpy.power(c2_cum_left[-1], (1 / hero_yields_left.shape[0])) - 1,
                                numpy.power(synth_cum_left[-1], (1 / hero_yields_left.shape[0])) - 1],
              'Max Drawdown Value': [max_drawdown_value(hero_yields_left),
                                     max_drawdown_value(static_yields_left),
                                     max_drawdown_value(component1_yields_left),
                                     max_drawdown_value(component2_yields_left),
                                     max_drawdown_value(syn_yields_left)],
              'Max Drawdown Length': [max_drawdown_length(hero_yields_left),
                                      max_drawdown_length(static_yields_left),
                                      max_drawdown_length(component1_yields_left),
                                      max_drawdown_length(component2_yields_left),
                                      max_drawdown_length(syn_yields_left)],
              'Positive Deals Rate': [positive_deals_rate(hero_yields_left),
                                      positive_deals_rate(static_yields_left),
                                      positive_deals_rate(component1_yields_left),
                                      positive_deals_rate(component2_yields_left),
                                      positive_deals_rate(syn_yields_left)],
              'St Dev': [numpy.std(hero_yields_left),
                         numpy.std(static_yields_left),
                         numpy.std(component1_yields_left),
                         numpy.std(component2_yields_left),
                         numpy.std(syn_yields_left)],
              'Downside Semivariance': [downside_semivariance(hero_yields_left),
                                        downside_semivariance(static_yields_left),
                                        downside_semivariance(component1_yields_left),
                                        downside_semivariance(component2_yields_left),
                                        downside_semivariance(syn_yields_left)],
              'Skew': [stats.skew(hero_yields_left.ravel()),
                       stats.skew(static_yields_left),
                       stats.skew(component1_yields_left),
                       stats.skew(component2_yields_left),
                       stats.skew(syn_yields_left)],
              'Kurtosis': [stats.kurtosis(hero_yields_left.ravel()),
                           stats.kurtosis(static_yields_left),
                           stats.kurtosis(component1_yields_left),
                           stats.kurtosis(component2_yields_left),
                           stats.kurtosis(syn_yields_left)],
              'VaR 99': [numpy.quantile(hero_yields_left, q=0.01),
                         numpy.quantile(static_yields_left, q=0.01),
                         numpy.quantile(component1_yields_left, q=0.01),
                         numpy.quantile(component2_yields_left, q=0.01),
                         numpy.quantile(syn_yields_left, q=0.01)],
              'Alpha': [alpha(hero_yields_left, bench_yields_left),
                        alpha(static_yields_left, bench_yields_left),
                        alpha(component1_yields_left, bench_yields_left),
                        alpha(component2_yields_left, bench_yields_left),
                        alpha(syn_yields_left, bench_yields_left)],
              'Beta': [beta(hero_yields_left, bench_yields_left),
                       beta(static_yields_left, bench_yields_left),
                       beta(component1_yields_left, bench_yields_left),
                       beta(component2_yields_left, bench_yields_left),
                       beta(syn_yields_left, bench_yields_left)],
              'Sharpe Ratio': [sharpe_ratio(hero_yields_left, bench_yields_left),
                               sharpe_ratio(static_yields_left, bench_yields_left),
                               sharpe_ratio(component1_yields_left, bench_yields_left),
                               sharpe_ratio(component2_yields_left, bench_yields_left),
                               sharpe_ratio(syn_yields_left, bench_yields_left)],
              'Sortino Ratio': [sortino_ratio(hero_yields_left, bench_yields_left),
                                sortino_ratio(static_yields_left, bench_yields_left),
                                sortino_ratio(component1_yields_left, bench_yields_left),
                                sortino_ratio(component2_yields_left, bench_yields_left),
                                sortino_ratio(syn_yields_left, bench_yields_left)]},
        index=['Hero', 'Static', 'C1', 'C2', 'Synth'])

    perf_right = pandas.DataFrame(
        data={'Annual Yield': [numpy.power(hero_cum_right[-1], (12 / hero_yields_right.shape[0])) - 1,
                               numpy.power(c0_cum_right[-1], (12 / hero_yields_right.shape[0])) - 1,
                               numpy.power(c1_cum_right[-1], (12 / hero_yields_right.shape[0])) - 1,
                               numpy.power(c2_cum_right[-1], (12 / hero_yields_right.shape[0])) - 1,
                               numpy.power(synth_cum_right[-1], (12 / hero_yields_right.shape[0])) - 1],
              'Monthly Yield': [numpy.power(hero_cum_right[-1], (1 / hero_yields_right.shape[0])) - 1,
                                numpy.power(c0_cum_right[-1], (1 / hero_yields_right.shape[0])) - 1,
                                numpy.power(c1_cum_right[-1], (1 / hero_yields_right.shape[0])) - 1,
                                numpy.power(c2_cum_right[-1], (1 / hero_yields_right.shape[0])) - 1,
                                numpy.power(synth_cum_right[-1], (1 / hero_yields_right.shape[0])) - 1],
              'Max Drawdown Value': [max_drawdown_value(hero_yields_right),
                                     max_drawdown_value(static_yields_right),
                                     max_drawdown_value(component1_yields_right),
                                     max_drawdown_value(component2_yields_right),
                                     max_drawdown_value(syn_yields_right)],
              'Max Drawdown Length': [max_drawdown_length(hero_yields_right),
                                      max_drawdown_length(static_yields_right),
                                      max_drawdown_length(component1_yields_right),
                                      max_drawdown_length(component2_yields_right),
                                      max_drawdown_length(syn_yields_right)],
              'Positive Deals Rate': [positive_deals_rate(hero_yields_right),
                                      positive_deals_rate(static_yields_right),
                                      positive_deals_rate(component1_yields_right),
                                      positive_deals_rate(component2_yields_right),
                                      positive_deals_rate(syn_yields_right)],
              'St Dev': [numpy.std(hero_yields_right),
                         numpy.std(static_yields_right),
                         numpy.std(component1_yields_right),
                         numpy.std(component2_yields_right),
                         numpy.std(syn_yields_right)],
              'Downside Semivariance': [downside_semivariance(hero_yields_right),
                                        downside_semivariance(static_yields_right),
                                        downside_semivariance(component1_yields_right),
                                        downside_semivariance(component2_yields_right),
                                        downside_semivariance(syn_yields_right)],
              'Skew': [stats.skew(hero_yields_right.ravel()),
                       stats.skew(static_yields_right),
                       stats.skew(component1_yields_right),
                       stats.skew(component2_yields_right),
                       stats.skew(syn_yields_right)],
              'Kurtosis': [stats.kurtosis(hero_yields_right.ravel()),
                           stats.kurtosis(static_yields_right),
                           stats.kurtosis(component1_yields_right),
                           stats.kurtosis(component2_yields_right),
                           stats.kurtosis(syn_yields_right)],
              'VaR 99': [numpy.quantile(hero_yields_right, q=0.01),
                         numpy.quantile(static_yields_right, q=0.01),
                         numpy.quantile(component1_yields_right, q=0.01),
                         numpy.quantile(component2_yields_right, q=0.01),
                         numpy.quantile(syn_yields_right, q=0.01)],
              'Alpha': [alpha(hero_yields_right, bench_yields_right),
                        alpha(static_yields_right, bench_yields_right),
                        alpha(component1_yields_right, bench_yields_right),
                        alpha(component2_yields_right, bench_yields_right),
                        alpha(syn_yields_right, bench_yields_right)],
              'Beta': [beta(hero_yields_right, bench_yields_right),
                       beta(static_yields_right, bench_yields_right),
                       beta(component1_yields_right, bench_yields_right),
                       beta(component2_yields_right, bench_yields_right),
                       beta(syn_yields_right, bench_yields_right)],
              'Sharpe Ratio': [sharpe_ratio(hero_yields_right, bench_yields_right),
                               sharpe_ratio(static_yields_right, bench_yields_right),
                               sharpe_ratio(component1_yields_right, bench_yields_right),
                               sharpe_ratio(component2_yields_right, bench_yields_right),
                               sharpe_ratio(syn_yields_right, bench_yields_right)],
              'Sortino Ratio': [sortino_ratio(hero_yields_right, bench_yields_right),
                                sortino_ratio(static_yields_right, bench_yields_right),
                                sortino_ratio(component1_yields_right, bench_yields_right),
                                sortino_ratio(component2_yields_right, bench_yields_right),
                                sortino_ratio(syn_yields_right, bench_yields_right)]},
        index=['Hero', 'Static', 'C1', 'C2', 'Synth'])

    return perf_left, perf_right


def daily_hedge_report(hero_cum_left, c0_cum_left, c1_cum_left, c2_cum_left, synth_cum_left,
                       hero_yields_left, static_yields_left, component1_yields_left, component2_yields_left,
                       syn_yields_left, bench_yields_left,
                       hero_cum_right, c0_cum_right, c1_cum_right, c2_cum_right, synth_cum_right,
                       hero_yields_right, static_yields_right, component1_yields_right, component2_yields_right,
                       syn_yields_right, bench_yields_right
                       ):
    perf_left = pandas.DataFrame(
        data={'20 Obs Yield': [numpy.power(hero_cum_left[-1], (20 / hero_yields_left.shape[0])) - 1,
                               numpy.power(c0_cum_left[-1], (20 / hero_yields_left.shape[0])) - 1,
                               numpy.power(c1_cum_left[-1], (20 / hero_yields_left.shape[0])) - 1,
                               numpy.power(c2_cum_left[-1], (20 / hero_yields_left.shape[0])) - 1,
                               numpy.power(synth_cum_left[-1], (20 / hero_yields_left.shape[0])) - 1],
              'Daily Yield': [numpy.power(hero_cum_left[-1], (1 / hero_yields_left.shape[0])) - 1,
                              numpy.power(c0_cum_left[-1], (1 / hero_yields_left.shape[0])) - 1,
                              numpy.power(c1_cum_left[-1], (1 / hero_yields_left.shape[0])) - 1,
                              numpy.power(c2_cum_left[-1], (1 / hero_yields_left.shape[0])) - 1,
                              numpy.power(synth_cum_left[-1], (1 / hero_yields_left.shape[0])) - 1],
              'Max Drawdown Value': [max_drawdown_value(hero_yields_left),
                                     max_drawdown_value(static_yields_left),
                                     max_drawdown_value(component1_yields_left),
                                     max_drawdown_value(component2_yields_left),
                                     max_drawdown_value(syn_yields_left)],
              'Max Drawdown Length': [max_drawdown_length(hero_yields_left),
                                      max_drawdown_length(static_yields_left),
                                      max_drawdown_length(component1_yields_left),
                                      max_drawdown_length(component2_yields_left),
                                      max_drawdown_length(syn_yields_left)],
              'Positive Deals Rate': [positive_deals_rate(hero_yields_left),
                                      positive_deals_rate(static_yields_left),
                                      positive_deals_rate(component1_yields_left),
                                      positive_deals_rate(component2_yields_left),
                                      positive_deals_rate(syn_yields_left)],
              'St Dev': [numpy.std(hero_yields_left),
                         numpy.std(static_yields_left),
                         numpy.std(component1_yields_left),
                         numpy.std(component2_yields_left),
                         numpy.std(syn_yields_left)],
              'Downside Semivariance': [downside_semivariance(hero_yields_left),
                                        downside_semivariance(static_yields_left),
                                        downside_semivariance(component1_yields_left),
                                        downside_semivariance(component2_yields_left),
                                        downside_semivariance(syn_yields_left)],
              'Skew': [stats.skew(hero_yields_left.ravel()),
                       stats.skew(static_yields_left),
                       stats.skew(component1_yields_left),
                       stats.skew(component2_yields_left),
                       stats.skew(syn_yields_left)],
              'Kurtosis': [stats.kurtosis(hero_yields_left.ravel()),
                           stats.kurtosis(static_yields_left),
                           stats.kurtosis(component1_yields_left),
                           stats.kurtosis(component2_yields_left),
                           stats.kurtosis(syn_yields_left)],
              'VaR 99': [numpy.quantile(hero_yields_left, q=0.01),
                         numpy.quantile(static_yields_left, q=0.01),
                         numpy.quantile(component1_yields_left, q=0.01),
                         numpy.quantile(component2_yields_left, q=0.01),
                         numpy.quantile(syn_yields_left, q=0.01)],
              'Alpha': [alpha(hero_yields_left, bench_yields_left),
                        alpha(static_yields_left, bench_yields_left),
                        alpha(component1_yields_left, bench_yields_left),
                        alpha(component2_yields_left, bench_yields_left),
                        alpha(syn_yields_left, bench_yields_left)],
              'Beta': [beta(hero_yields_left, bench_yields_left),
                       beta(static_yields_left, bench_yields_left),
                       beta(component1_yields_left, bench_yields_left),
                       beta(component2_yields_left, bench_yields_left),
                       beta(syn_yields_left, bench_yields_left)],
              'Sharpe Ratio': [sharpe_ratio(hero_yields_left, bench_yields_left),
                               sharpe_ratio(static_yields_left, bench_yields_left),
                               sharpe_ratio(component1_yields_left, bench_yields_left),
                               sharpe_ratio(component2_yields_left, bench_yields_left),
                               sharpe_ratio(syn_yields_left, bench_yields_left)],
              'Sortino Ratio': [sortino_ratio(hero_yields_left, bench_yields_left),
                                sortino_ratio(static_yields_left, bench_yields_left),
                                sortino_ratio(component1_yields_left, bench_yields_left),
                                sortino_ratio(component2_yields_left, bench_yields_left),
                                sortino_ratio(syn_yields_left, bench_yields_left)]},
        index=['Hero', 'Static', 'C1', 'C2', 'Synth'])

    perf_right = pandas.DataFrame(
        data={'20 Obs Yield': [numpy.power(hero_cum_right[-1], (20 / hero_yields_right.shape[0])) - 1,
                               numpy.power(c0_cum_right[-1], (20 / hero_yields_right.shape[0])) - 1,
                               numpy.power(c1_cum_right[-1], (20 / hero_yields_right.shape[0])) - 1,
                               numpy.power(c2_cum_right[-1], (20 / hero_yields_right.shape[0])) - 1,
                               numpy.power(synth_cum_right[-1], (20 / hero_yields_right.shape[0])) - 1],
              'Daily Yield': [numpy.power(hero_cum_right[-1], (1 / hero_yields_right.shape[0])) - 1,
                              numpy.power(c0_cum_right[-1], (1 / hero_yields_right.shape[0])) - 1,
                              numpy.power(c1_cum_right[-1], (1 / hero_yields_right.shape[0])) - 1,
                              numpy.power(c2_cum_right[-1], (1 / hero_yields_right.shape[0])) - 1,
                              numpy.power(synth_cum_right[-1], (1 / hero_yields_right.shape[0])) - 1],
              'Max Drawdown Value': [max_drawdown_value(hero_yields_right),
                                     max_drawdown_value(static_yields_right),
                                     max_drawdown_value(component1_yields_right),
                                     max_drawdown_value(component2_yields_right),
                                     max_drawdown_value(syn_yields_right)],
              'Max Drawdown Length': [max_drawdown_length(hero_yields_right),
                                      max_drawdown_length(static_yields_right),
                                      max_drawdown_length(component1_yields_right),
                                      max_drawdown_length(component2_yields_right),
                                      max_drawdown_length(syn_yields_right)],
              'Positive Deals Rate': [positive_deals_rate(hero_yields_right),
                                      positive_deals_rate(static_yields_right),
                                      positive_deals_rate(component1_yields_right),
                                      positive_deals_rate(component2_yields_right),
                                      positive_deals_rate(syn_yields_right)],
              'St Dev': [numpy.std(hero_yields_right),
                         numpy.std(static_yields_right),
                         numpy.std(component1_yields_right),
                         numpy.std(component2_yields_right),
                         numpy.std(syn_yields_right)],
              'Downside Semivariance': [downside_semivariance(hero_yields_right),
                                        downside_semivariance(static_yields_right),
                                        downside_semivariance(component1_yields_right),
                                        downside_semivariance(component2_yields_right),
                                        downside_semivariance(syn_yields_right)],
              'Skew': [stats.skew(hero_yields_right.ravel()),
                       stats.skew(static_yields_right),
                       stats.skew(component1_yields_right),
                       stats.skew(component2_yields_right),
                       stats.skew(syn_yields_right)],
              'Kurtosis': [stats.kurtosis(hero_yields_right.ravel()),
                           stats.kurtosis(static_yields_right),
                           stats.kurtosis(component1_yields_right),
                           stats.kurtosis(component2_yields_right),
                           stats.kurtosis(syn_yields_right)],
              'VaR 99': [numpy.quantile(hero_yields_right, q=0.01),
                         numpy.quantile(static_yields_right, q=0.01),
                         numpy.quantile(component1_yields_right, q=0.01),
                         numpy.quantile(component2_yields_right, q=0.01),
                         numpy.quantile(syn_yields_right, q=0.01)],
              'Alpha': [alpha(hero_yields_right, bench_yields_right),
                        alpha(static_yields_right, bench_yields_right),
                        alpha(component1_yields_right, bench_yields_right),
                        alpha(component2_yields_right, bench_yields_right),
                        alpha(syn_yields_right, bench_yields_right)],
              'Beta': [beta(hero_yields_right, bench_yields_right),
                       beta(static_yields_right, bench_yields_right),
                       beta(component1_yields_right, bench_yields_right),
                       beta(component2_yields_right, bench_yields_right),
                       beta(syn_yields_right, bench_yields_right)],
              'Sharpe Ratio': [sharpe_ratio(hero_yields_right, bench_yields_right),
                               sharpe_ratio(static_yields_right, bench_yields_right),
                               sharpe_ratio(component1_yields_right, bench_yields_right),
                               sharpe_ratio(component2_yields_right, bench_yields_right),
                               sharpe_ratio(syn_yields_right, bench_yields_right)],
              'Sortino Ratio': [sortino_ratio(hero_yields_right, bench_yields_right),
                                sortino_ratio(static_yields_right, bench_yields_right),
                                sortino_ratio(component1_yields_right, bench_yields_right),
                                sortino_ratio(component2_yields_right, bench_yields_right),
                                sortino_ratio(syn_yields_right, bench_yields_right)]},
        index=['Hero', 'Static', 'C1', 'C2', 'Synth'])

    return perf_left, perf_right
