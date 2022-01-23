import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fitter
import VarGamma as vg

## this function that makes log increments of market cap data and fits all three distributions and calculates the sum-square errors for each ##
def get_errors(df, plot=False):
    ## preparing shifting one column so that we the increment values be in the same rows ##
    MarketCap = df.columns[0]
    df['MarketCapPrev'] = df[MarketCap].shift()
    df['increment'] = df[MarketCap]/df['MarketCapPrev']

    ## log of the increment ##
    df['log_inc'] = np.log(df['increment'])
    data = df['log_inc'].dropna()

    ## dropping lower than 0.01 quantile and higher than 0.99 quantiles data (the outliers) ##
    data = data[data.between(data.quantile(.01), data.quantile(.99))].apply(float)

    ## Rice's rule for number of bins ##
    NBR_BINS = np.int16(2*(data.size)**(1/3)

    ## fit the distributions ##
    f = fitter.Fitter(data, timeout = 100, bins = NBR_BINS), distributions = ['vg', 'norm', 'norminvgauss'])# np.int16(2*(data.size)**(1/3))  Rice's rule
    f.fit()
    if plot == True:
        plt.rcParams["axes.labelpad"] = 12
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['legend.loc'] = 'upper right'
        plt.rc('legend', fontsize=13)
        plt.rc('axes', axisbelow=True)
        plt.rc('grid', linestyle="--", color='grey', alpha=0.4, lw=0.5)
        plt.xlabel('$\log{\\frac{E_t}{E_{t-1}}}$')
        plt.ylabel('count')
        plt.show()
        ## or use these 2 lines below to save the plot ##
        # plt.savefig(save_name + '.pdf', bbox_inches = 'tight')
        # plt.close()

    ## get the sum-square errors and save them as a row in a dataframe error_data by the index of corresponding ric ##
    row_of_errors = [[f.summary()['sumsquare_error']['vg'], f.summary()['sumsquare_error']['norminvgauss'], f.summary()['sumsquare_error']['norm']]]
    vg_parameters = f.fitted_param['vg']
    nig_parameters = f.fitted_param['norminvgauss']
    norm_parameters = f.fitted_param['norm']
    error_data = pd.DataFrame(row_of_errors, columns=['VG', 'NIG', 'NORM'], index=[ric])
    return error_data, vg_parameters, nig_parameters, norm_parameters