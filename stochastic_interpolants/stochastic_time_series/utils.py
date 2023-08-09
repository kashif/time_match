import matplotlib.pyplot as plt
import pandas as pd
from si import SIEstimator
from si_epsilon import SIEPSEstimator
from ddpm import DDPMEstimator
from sgm import SGMEstimator
from t2t_sb import T2TSBEstimator
from si_regularized import SIREGEstimator
from si_epsilon_regularized import SIEPSREGEstimator
from fm import FMEstimator

def get_estimator(args, dataset):
    if 'wiki' in args.dataset:
        max_target_dim = 2000
    else:
        max_target_dim = int(dataset.metadata.feat_static_cat[0].cardinality)
    
    if 'wiki' in args.dataset or 'taxi' in args.dataset:
        lags_seq = [1]
    else:
        lags_seq = None

    if args.estimator == 'ddpm':
        estimator = DDPMEstimator(
            input_size=max_target_dim,
            hidden_size=64,
            num_layers=2,
            dropout_rate=0.1,
            linear_end=0.1,
            lags_seq=lags_seq,
            n_timestep=args.steps,
            prediction_length=dataset.metadata.prediction_length,
            context_length=dataset.metadata.prediction_length,
            freq=dataset.metadata.freq,
            scaling="mean",
            trainer_kwargs=dict(max_epochs=100, accelerator="gpu", devices=[args.device]),
        )
    elif args.estimator == 'sgm':
        estimator = SGMEstimator(
            input_size=max_target_dim,
            hidden_size=64,
            num_layers=2,
            dropout_rate=0.1,
            lags_seq=lags_seq,
            linear_start=0.001,
            linear_end=10.0,
            n_timestep=args.steps,
            prediction_length=dataset.metadata.prediction_length,
            context_length=dataset.metadata.prediction_length,
            freq=dataset.metadata.freq,
            scaling="mean",
            trainer_kwargs=dict(max_epochs=100, accelerator="gpu", devices=[args.device]),
        )
    elif args.estimator == 'i2sb':
        estimator = T2TSBEstimator(
            input_size=int(max_target_dim),
            hidden_size=64,
            num_layers=2,
            dropout_rate=0.1,
            lags_seq=lags_seq,
            linear_start=0.0001,
            linear_end=0.01,
            n_timestep=args.steps,
            start_noise=args.start_noise,
            prediction_length=dataset.metadata.prediction_length,
            context_length=dataset.metadata.prediction_length,
            freq=dataset.metadata.freq,
            scaling="mean",
            trainer_kwargs=dict(max_epochs=100, accelerator="gpu", devices=[args.device]),
        )
    elif args.estimator == 'si':
        estimator = SIEstimator(
            input_size=int(max_target_dim),
            hidden_size=64,
            num_layers=2,
            dropout_rate=0.1,
            lags_seq=lags_seq,
            interpolant=args.interpolant,
            gamma=args.gamma,
            start_noise=args.start_noise,
            n_timestep=args.steps,
            epsilon=args.epsilon,
            prediction_length=dataset.metadata.prediction_length,
            context_length=dataset.metadata.prediction_length,
            freq=dataset.metadata.freq,
            scaling="mean",
            trainer_kwargs=dict(max_epochs=100, accelerator="gpu", devices=[args.device]),
        )
    elif args.estimator == 'si_epsilon':
        estimator = SIEPSEstimator(
            input_size=int(max_target_dim),
            hidden_size=64,
            num_layers=2,
            dropout_rate=0.1,
            lags_seq=lags_seq,
            interpolant=args.interpolant,
            gamma=args.gamma,
            start_noise=args.start_noise,
            n_timestep=args.steps,
            epsilon=args.epsilon,
            prediction_length=dataset.metadata.prediction_length,
            context_length=dataset.metadata.prediction_length,
            freq=dataset.metadata.freq,
            scaling="mean",
            trainer_kwargs=dict(max_epochs=100, accelerator="gpu", devices=[args.device]),
        )
    elif args.estimator == 'si_reg':
        estimator = SIREGEstimator(
            input_size=int(max_target_dim),
            hidden_size=64,
            num_layers=2,
            dropout_rate=0.1,
            lags_seq=lags_seq,
            interpolant=args.interpolant,
            gamma=args.gamma,
            n_timestep=args.steps,
            epsilon=args.epsilon,
            prediction_length=dataset.metadata.prediction_length,
            context_length=dataset.metadata.prediction_length,
            freq=dataset.metadata.freq,
            scaling="mean",
            trainer_kwargs=dict(max_epochs=100, accelerator="gpu", devices=[args.device]),
        )
    elif args.estimator == 'si_epsilon_reg':
        estimator = SIEPSREGEstimator(
            input_size=int(max_target_dim),
            hidden_size=64,
            num_layers=2,
            dropout_rate=0.1,
            lags_seq=lags_seq,
            interpolant=args.interpolant,
            gamma=args.gamma,
            n_timestep=args.steps,
            epsilon=args.epsilon,
            prediction_length=dataset.metadata.prediction_length,
            context_length=dataset.metadata.prediction_length,
            freq=dataset.metadata.freq,
            scaling="mean",
            trainer_kwargs=dict(max_epochs=100, accelerator="gpu", devices=[args.device]),
        )
    elif args.estimator == 'fm':
        estimator = FMEstimator(
            input_size=int(max_target_dim),
            hidden_size=64,
            num_layers=2,
            dropout_rate=0.1,
            lags_seq=lags_seq,
            n_timestep=args.steps,
            sigma_min=0.01,
            prediction_length=dataset.metadata.prediction_length,
            context_length=dataset.metadata.prediction_length,
            freq=dataset.metadata.freq,
            scaling="mean",
            trainer_kwargs=dict(max_epochs=100, accelerator="gpu", devices=[args.device]),
        )
    
    return estimator

def get_log(
    agg_metric
):
    log = "CRPS: {}\n".format(agg_metric["mean_wQuantileLoss"])
    log += "ND: {}\n".format(agg_metric["ND"])
    log += "NRMSE: {}\n".format(agg_metric["NRMSE"])
    log += "MSE: {}\n\n".format(agg_metric["MSE"])
    log += "CRPS: {}\n".format(agg_metric["m_sum_mean_wQuantileLoss"])
    log += "ND: {}\n".format(agg_metric["m_sum_ND"])
    log += "NRMSE: {}\n".format(agg_metric["m_sum_NRMSE"])
    log += "MSE: {}".format(agg_metric["m_sum_MSE"])
    return log

def plot(
    target,
    forecast,
    prediction_length,
    prediction_intervals=(50.0, 90.0),
    color="g",
    fname=None,
):
    label_prefix = ""
    rows = 4
    cols = 4
    fig, axs = plt.subplots(rows, cols, figsize=(24, 24))
    axx = axs.ravel()
    seq_len, target_dim = target.shape

    ps = [50.0] + [
        50.0 + f * c / 2.0 for c in prediction_intervals for f in [-1.0, +1.0]
    ]

    percentiles_sorted = sorted(set(ps))

    def alpha_for_percentile(p):
        return (p / 100.0) ** 0.3

    for dim in range(0, min(rows * cols, target_dim)):
        ax = axx[dim]

        target[-2 * prediction_length :][dim].plot(ax=ax)

        ps_data = [forecast.quantile(p / 100.0)[:, dim] for p in percentiles_sorted]
        i_p50 = len(percentiles_sorted) // 2

        p50_data = ps_data[i_p50]
        p50_series = pd.Series(data=p50_data, index=forecast.index)
        p50_series.plot(color=color, ls="-", label=f"{label_prefix}median", ax=ax)

        for i in range(len(percentiles_sorted) // 2):
            ptile = percentiles_sorted[i]
            alpha = alpha_for_percentile(ptile)
            ax.fill_between(
                forecast.index,
                ps_data[i],
                ps_data[-i - 1],
                facecolor=color,
                alpha=alpha,
                interpolate=True,
            )
            # Hack to create labels for the error intervals.
            # Doesn't actually plot anything, because we only pass a single data point
            pd.Series(data=p50_data[:1], index=forecast.index[:1]).plot(
                color=color,
                alpha=alpha,
                linewidth=10,
                label=f"{label_prefix}{100 - ptile * 2}%",
                ax=ax,
            )

    legend = ["observations", "median prediction"] + [
        f"{k}% prediction interval" for k in prediction_intervals
    ][::-1]
    axx[0].legend(legend, loc="upper left")

    if fname is not None:
        plt.savefig(fname, bbox_inches="tight", pad_inches=0.05)