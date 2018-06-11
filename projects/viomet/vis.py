'''
Plots for my first-year (and beyond) figurative violence in the media project.

Author: Matthew Turner <maturner01@gmail.com>

Date: April 01, 2017
'''
import matplotlib.pyplot as plt
import matplotlib.dates as pltdates
import pandas as pd
# import seaborn as sns

from datetime import date, datetime, timedelta

from .analysis import relative_likelihood

from projects.common.analysis import (
    daily_frequency, daily_metaphor_counts
)


# CUR_PAL = sns.color_palette()


# for 8.5x11 paper
DEFAULT_FIGSIZE = (7.5, 5)


def by_network_frequency_figure(
            frequency_df,
            date_range=pd.date_range(
                '2016-09-01', '2016-11-30', freq='D'
            ),
            iatv_corpus_name=None,
            freq=True,
            partition_infos=None,
            font_scale=1.15,
            save_path=None):

    # sns.axes_style("darkgrid")
    # sns.set(font_scale=font_scale)

    # CUR_PAL = sns.color_palette()

    df = frequency_df

    # fits are not being shown for this condition
    if (partition_infos is None):

        if freq:

            network_freq = daily_frequency(
                df, date_range, iatv_corpus_name, by=['network']
            )

            network_freq.plot(style='o')

        else:

            full_df = daily_metaphor_counts(
                df, ['network'], date_range
            )[['MSNBCW', 'CNNW', 'FOXNEWSW']]

            full_df.plot(style='o')

    # show fits TODO Include more arguments so that fits don't have to be
    # generated just to plot. Generate fits outside and pass fits in.
    else:

        if freq:

            # put networks in desired order, left to right
            networks = ['MSNBCW', 'CNNW', 'FOXNEWSW']
            line_styles = [':', '--', '-']
            markers = ['s', 'o', '^']
            # markers = ['bs', 'go', 'r^']

            network_freq = daily_frequency(
                df, date_range, iatv_corpus_name, by=['network']
            )

            ax = network_freq[networks].plot(
                style=markers, mew=1, mfc='white', ms=6, alpha=0.9, legend=False,
                figsize=DEFAULT_FIGSIZE, mec='lightgrey'
            )

            for net_idx, network in enumerate(networks):

                pinfo = partition_infos[network]

                day_td = timedelta(seconds=60)

                d0 = date_range[0]
                d1 = pinfo.partition_date_1 - day_td

                d2 = pinfo.partition_date_1
                d3 = pinfo.partition_date_2

                d4 = pinfo.partition_date_2 + day_td
                d5 = date_range[-1]

                fg = pinfo.f_ground
                fe = pinfo.f_excited

                dates = pd.DatetimeIndex([d0, d1, d2, d3, d4, d5])
                datas = [fg, fg, fe, fe, fg, fg]

                network_formatted = ['MSNBC', 'CNN', 'Fox News']

                pd.Series(
                    index=dates, data=datas
                ).plot(
                    lw=3, ax=ax, color='k',
                    ls=line_styles[net_idx],
                    # legend=True, label=network_formatted[net_idx]
                )

            ax.xaxis.set_minor_formatter(pltdates.DateFormatter('%-d'))
            ax.xaxis.set_minor_locator(pltdates.DayLocator(bymonthday=(1, 15)))

            yheight = 0.1
            zo = 10
            textargs = dict(
                size=13,
                ha='right',
                bbox=dict(alpha=0.6, color='white')
            )
            if '2016' in iatv_corpus_name:
                ax.axvline(
                    datetime(2016, 9, 26), ymax=yheight, color='k', zorder=zo
                )
                ax.axvline(
                    datetime(2016, 10, 9), ymax=yheight, color='k', zorder=zo
                )
                ax.axvline(
                    datetime(2016, 10, 19), ymax=yheight, color='k', zorder=zo
                )
                # ax.axhline(
                #     yheight, color='k', xmin=0.278, xmax=0.535)
                # , '2016-9-26', '2016-10-19', zorder=1
                # )
                ax.text('2016-9-25', 0.2, "Debate #1", **textargs)
                ax.text('2016-10-8', 0.2, "#2", **textargs)
                ax.text('2016-10-18', 0.2, "#3", **textargs)
                ax.set_xlim(datetime(2016, 8, 31), datetime(2016, 11, 29))

            if '2012' in iatv_corpus_name:
                ax.axvline(
                    datetime(2012, 10, 3), ymax=yheight, color='k', zorder=zo
                )
                ax.axvline(
                    datetime(2012, 10, 16), ymax=yheight, color='k', zorder=zo
                )
                ax.axvline(
                    datetime(2012, 10, 22), ymax=yheight, color='k', zorder=zo
                )
                # ax.axhline(
                #     yheight, color='k', xmin=0.278, xmax=0.535)
                # , '2016-9-26', '2016-10-19', zorder=1
                # )
                ax.text('2012-10-2', 0.2, "Debate #1", **textargs)
                ax.text('2012-10-15', 0.2, "#2", **textargs)
                ax.text('2012-10-21', 0.2, "#3", **textargs)
                ax.set_xlim(datetime(2012, 8, 31), datetime(2012, 11, 29))

            ax.grid(False)

            ax.set_xlabel('Date')
            ax.set_ylabel('Frequency of usage')
            ax.set_title(
                'Metaphorical violence usage data and dynamic model'
            )

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            plt.tight_layout()

            ax.set_ylim(-.25, 7.25)

            import matplotlib.lines as mlines
            net_lines = []

            # Manually create the legend with line and marker styles.
            for net_idx, network in enumerate(network_formatted):
                net_lines.append(
                    mlines.Line2D(
                        [], [],
                        mfc='white', color='black', marker=markers[net_idx],
                        markersize=7.5, ls=line_styles[net_idx],
                        label=network, mec='gray'
                    )
                )
            # 3.8 handlelength to only have whole dashes.
            plt.legend(handles=net_lines, handlelength=3.8)

            if save_path is not None:
                fig = ax.get_figure()
                fig.savefig(save_path)
                plt.close()


# FIGURE 1
def plot_daily_usage(df, ma_period=7, lw=3, marker='o', ms=10, xlab='Date',
                     ylab='Figurative uses',
                     title='Figurative uses of violent words'):
    '''
    Plot daily and `ma_period`-day moving average from dataframe with index
    of every date in observation period (currently Sep 1 - Nov 30)
    '''
    import seaborn as sns
    sns.set(font_scale=1.75)

    # calculate the moving average over ma_period
    ma = df.rolling(ma_period).mean().fillna(0)

    ax = ma.plot(lw=lw, figsize=(15, 12))

    df.plot(marker='o', ms=ms, lw=0, ax=ax, color=CUR_PAL)

    p, _ = ax.get_legend_handles_labels()
    leg = ax.legend(p, ['attack ({}-day MA)'.format(ma_period), 'beat', 'hit',
                        'attack (daily)', 'beat', 'hit'], frameon=True)

    leg.get_frame().set_facecolor('white')

    # XXX magic number gotten from ipython session
    plt.xlim([17044, plt.xlim()[1]])

    plt.ylim([0, 17])

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)

    ax.axvline(date(2016, 9, 26), color='k', lw=3, ls='--')
    ax.axvline(date(2016, 10, 9), color='k', lw=3, ls='--')
    ax.axvline(date(2016, 10, 19), color='k', lw=3, ls='--')
    ax.axvline(date(2016, 11, 8), color='k', lw=3, ls='--')

    plt.tight_layout()


def plot_total_daily_usage(series, ma_period=5, lw=3, marker='o', ms=6,
                           xlab='Date', ylab='Frequency of uses',
                           plot_ma=False,
                           show_debates=False, show_election=False,
                           show_means=False,
                           save_path=None,
                           title='Figurative violence usage during debate season'):
    sns.set(font_scale=1.5)

    fig = plt.figure(figsize=(16, 9))
    ax = series.plot(style='ok', markerfacecolor="None", markeredgecolor="black",
                     markeredgewidth=5, lw=0, ms=ms)

    if plot_ma:
        ma = series.rolling(ma_period).mean().fillna(0)

        ax = ma.plot(style='k-', lw=lw, ax=ax, figsize=(15, 12))

        p, _ = ax.get_legend_handles_labels()
        leg = ax.legend(p, ['frequency of usage ({}-day) MA'.format(ma_period),
                            'frequency of usage per day'
                            ]
                        )
    # else:
        # p, _ = ax.get_legend_handles_labels()
        # leg = ax.legend(p, ['frequency of usage per day'])

    # leg.get_frame().set_facecolor('white')

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)

    if show_debates:
        ax.axvline(date(2016, 9, 26), color='#cc0000', lw=3, ls='--')
        ax.axvline(date(2016, 10, 9), color='#cc0000', lw=3, ls='--')
        ax.axvline(date(2016, 10, 19), color='#cc0000', lw=3, ls='--')

        plt.text(date(2016, 9, 25), 6, 'First Debate\nSeptember 26',
                 horizontalalignment='right', color='#cc0000')
        plt.text(date(2016, 10, 8), 6.5, 'Second Debate\nOctober 9',
                 horizontalalignment='right', color='#cc0000')
        plt.text(date(2016, 10, 20), 6, 'Third Debate\nOctober 19',
                 horizontalalignment='left', color='#cc0000')
    if show_election:
        ax.axvline(date(2016, 11, 8), color='#cc0000', lw=3, ls='--')

    if show_means:
        ax.plot(
            [date(2016, 9, 1), date(2016, 9, 24)], [1, 1],
            lw=5, color='#cc0000'
        )
        ax.plot(
            [date(2016, 9, 25), date(2016, 10, 26)], [2.7, 2.7],
            lw=5, color='#cc0000'
        )
        ax.plot(
            [date(2016, 10, 27), date(2016, 11, 30)], [1, 1],
            lw=5, color='#cc0000'
        )

    ax.xaxis.set_major_formatter(pltdates.DateFormatter('%m/%d/%Y'))
    ax.xaxis.set_major_locator(pltdates.DayLocator(bymonthday=(1, 15)))

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        plt.close()

    return ax


# FIGURE 2
def aic_heatmap(df, relative=False, lw=2, annot=True, lim_ticklabels=False,
                title='', save_path=None):
    '''
    heatmap demonstrating relative likelihood of each model minimizing
    information loss

    Arguements:
        df (pandas.DataFrame): data frame with the first and second
            partition dates and an AIC column if relative=False and a
            rl column if relative=True
        relative (bool): see above

    Returns:
        (matplotlib.pyplot.Axes): Axes plotted to
    '''
    sns.set(font_scale=1.2)

    cbar_kws = dict(label='Relative Likelihood, $\mathcal{L}_i$', size=16)

    fig = plt.figure(figsize=(16, 9))

    if relative:
        val_col = 'rl'
        fmt = '.2f'

        df_rel = df.copy()
        # min_aic = df_rel.AIC.min()

        # df_rel = df_rel.rename(columns={'AIC': val_col})
        # df_rel[val_col] = relative_likelihood(min_aic, df_rel[val_col])

        if lim_ticklabels:
            ax = sns.heatmap(
                df_rel.pivot('first_date', 'last_date', val_col),
                annot=annot, fmt=fmt, linewidths=lw, xticklabels=5,
                yticklabels=5
            )

        else:
            ax = sns.heatmap(
                df_rel.pivot('first_date', 'last_date', val_col),
                annot=annot, fmt=fmt, linewidths=lw
            )

        cbar = plt.gcf().axes[-1]  # .colorbar(df_rel[val_col].values)
        cbar.tick_params(labelsize=14)
        cbar.set_title('Relative Likelihood, $\mathcal{L}_i$\n',
                       size=18, loc='left')

    else:
        fmt = '1.0f'

        if lim_ticklabels:
            ax = sns.heatmap(
                df_rel.pivot('first_date', 'last_date', val_col),
                annot=annot, fmt=fmt, linewidths=lw, xticklabels=5,
                yticklabels=5, cbar_kws=cbar_kws
            )

        else:
            ax = sns.heatmap(
                df_rel.pivot('first_date', 'last_date', val_col),
                annot=annot, fmt=fmt, linewidths=lw, cbar_kws=cbar_kws
            )

    ax.set_ylabel('Date frequency increased', size=20)
    ax.set_xlabel('Date frequency returned to normal', size=20)

    yt_labels = df_rel.first_date.dt.strftime('%m-%d').iloc[
        [int(yt) for yt in ax.get_yticks()]
    ]
    ax.set_yticklabels(yt_labels, rotation=0, size=15)

    dates = df_rel.last_date.unique()[[int(xt) for xt in ax.get_xticks()]]
    import ipdb
    ipdb.set_trace()
    xt_labels = [
        dt.astype(datetime).strftime('%m-%d') for dt in dates
    ]
    import ipdb
    ipdb.set_trace()
    ax.set_xticklabels(xt_labels, rotation=-30, ha='left', size=15)

    ax.invert_yaxis()
    plt.title(title, size=22)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        plt.close()

    return ax
