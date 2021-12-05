#


#
import numpy
import pandas
# from scipy import stats
from matplotlib import pyplot


#


#
class DiStill:
    def __init__(self, nn_model, nn_kwargs, distill_model, distill_kwargs, commi):
        self.nn_model = nn_model
        self.nn_kwargs = nn_kwargs
        self.nn_model_fit = None
        self.distill_model = distill_model
        self.distill_kwargs = distill_kwargs
        self.distill_model_fit = None
        self.commi = commi

    def _fit_nn(self, X_train, Y_train, X_val, Y_val):
        self.nn_model_fit = self.nn_model(**self.nn_kwargs)
        self.nn_model_fit.fit(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val)

    def _fit_still(self, X, Y):
        self.distill_model_fit = self.distill_model(**self.distill_kwargs)
        self.distill_model_fit.fit(X=X, y=Y)

    def still(self, X_train, Y_train, X_val, Y_val):
        self._fit_nn(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val)
        predicted = self.nn_model_fit.predict(X=X_train)
        self._fit_still(X=X_train, Y=predicted[:, 0])

    def _portfolio(self, predicted, Y, cum):
        distribute_dynamics = (predicted * Y.numpy()).sum(axis=1).reshape(-1, 1)
        if cum:
            cumulative_dynamics = ((distribute_dynamics + 1) * (1 - self.commi)).prod()
            return cumulative_dynamics
        else:
            return distribute_dynamics

    def nn_signals(self, X):
        predicted = self.nn_model_fit.predict(X=X)
        return predicted

    def nn_portfolio(self, X, Y, cum):
        predicted = self.nn_signals(X=X)
        cumulative_dynamics = self._portfolio(predicted=predicted, Y=Y, cum=cum)
        return cumulative_dynamics

    def distill_signals(self, X):
        predicted = self.distill_model_fit.predict(X=X).reshape(-1, 1)
        predicted[predicted < 0] = 0
        predicted[predicted > 1] = 1
        predicted = numpy.concatenate((predicted, 1 - predicted), axis=1)
        return predicted

    def distill_portfolio(self, X, Y, cum):
        predicted = self.distill_signals(X=X)
        cumulative_dynamics = self._portfolio(predicted=predicted, Y=Y, cum=cum)
        return cumulative_dynamics

    def _plot(self, hero_yields_left, component1_yields_left, component2_yields_left, tt_left, bench_yields_left,
              hero_yields_right, component1_yields_right, component2_yields_right, tt_right, bench_yields_right,
              synth, do_plot, report):

        if do_plot:
            fig, ax = pyplot.subplots(6, 2)

        # tt_left = numpy.array(numpy.arange(hero_yields_left.shape[0]))

        hero_cum_left = (1 + hero_yields_left).cumprod()
        static_yields_left = 0.5 * component1_yields_left + 0.5 * component2_yields_left
        c0_cum_left = (1 + static_yields_left).cumprod()
        c1_cum_left = (1 + component1_yields_left).cumprod()
        c2_cum_left = (1 + component2_yields_left).cumprod()
        syn_yields_left = synth * static_yields_left + (1 - synth) * hero_yields_left.ravel()
        synth_cum_left = (1 + syn_yields_left).cumprod()

        # tt_right = numpy.array(numpy.arange(hero_yields_right.shape[0]))

        hero_cum_right = (1 + hero_yields_right).cumprod()
        static_yield_right = 0.5 * component1_yields_right + 0.5 * component2_yields_right
        c0_cum_right = (1 + static_yield_right).cumprod()
        c1_cum_right = (1 + component1_yields_right).cumprod()
        c2_cum_right = (1 + component2_yields_right).cumprod()
        syn_yield_right = synth * static_yield_right + (1 - synth) * hero_yields_right.ravel()
        synth_cum_right = (1 + syn_yield_right).cumprod()

        perf_left, perf_right = report(hero_cum_left, c0_cum_left, c1_cum_left, c2_cum_left, synth_cum_left,
                          hero_yields_left, static_yields_left, component1_yields_left, component2_yields_left,
                          syn_yields_left, bench_yields_left,
                          hero_cum_right, c0_cum_right, c1_cum_right, c2_cum_right, synth_cum_right,
                          hero_yields_right, static_yield_right, component1_yields_right, component2_yields_right,
                          syn_yield_right, bench_yields_right)

        print(perf_left)
        print(perf_right)

        if do_plot:
            ax[0, 0].plot(tt_left, hero_cum_left, 'black', tt_left, c0_cum_left, 'gray', tt_left, c1_cum_left, 'blue',
                          tt_left, c2_cum_left,
                          'orange', tt_left,
                          synth_cum_left, '#960018')
            ax[1, 0].plot(tt_left, hero_yields_left, 'black')
            ax[2, 0].plot(tt_left, static_yields_left, 'gray')
            ax[3, 0].plot(tt_left, component1_yields_left, 'blue')
            ax[4, 0].plot(tt_left, component2_yields_left, 'orange')
            ax[5, 0].plot(tt_left, syn_yields_left, '#960018')

        if do_plot:
            ax[0, 1].plot(tt_right, hero_cum_right, 'black', tt_right, c0_cum_right, 'gray', tt_right, c1_cum_right,
                          'blue', tt_right,
                          c2_cum_right, 'orange', tt_right,
                          synth_cum_right, '#960018')
            ax[1, 1].plot(tt_right, hero_yields_right, 'black')
            ax[2, 1].plot(tt_right, static_yield_right, 'gray')
            ax[3, 1].plot(tt_right, component1_yields_right, 'blue')
            ax[4, 1].plot(tt_right, component2_yields_right, 'orange')
            ax[5, 1].plot(tt_right, syn_yield_right, '#960018')

        return perf_left, perf_right

    def plot(self, X_train, Y_train, tt_train, bench_train, X_test, Y_test, tt_test, bench_test, on, report, synth=0.8, do_plot=True):

        if on == 'nn':
            ge = self.nn_portfolio
        else:
            ge = self.distill_portfolio

        dynamics_train = ge(X=X_train, Y=Y_train, cum=False)
        dynamics_test = ge(X=X_test, Y=Y_test, cum=False)
        print('huh')
        return self._plot(hero_yields_left=dynamics_train, component1_yields_left=Y_train[:, 0].numpy(),
                          component2_yields_left=Y_train[:, 1].numpy(), tt_left=tt_train, bench_yields_left=bench_train,
                          hero_yields_right=dynamics_test, component1_yields_right=Y_test[:, 0].numpy(),
                          component2_yields_right=Y_test[:, 1].numpy(), tt_right=tt_test, bench_yields_right=bench_test,
                          synth=synth, do_plot=do_plot, report=report)
