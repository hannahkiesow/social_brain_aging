import os
import numpy as np
import pandas as pd
import joblib
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from matplotlib import pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


OUT_DIR = os.path.abspath('/Users/hannah/UKBB_socialbrain_aging')
DECONF = True
DO_SAVEFIGS = True
DO_PARTIALVOLUMES = False
try:
    os.mkdir(OUT_DIR)
except:
    print('Output directory already exists!')
    pass


COLS_NAMES = []
for fname in ['UKBB_socialbrain_aging/ukbbids_aging.txt']:
    with open(fname) as f:
        lines = f.readlines()
        f.close()
        for line in lines:
            COLS_NAMES.append(line.split('\t'))
COLS_NAMES = np.array(COLS_NAMES)

if 'ukbb' not in locals():
    ukbb = pd.read_csv('UKBB_socialbrain_aging/ukb_add1_holmes_merge_brain.csv')
else:
    print('Database is already in memory!')



T1_subnames, DMN_vols, rois = joblib.load('UKBB_socialbrain_aging/dump_sMRI_socialbrain_sym_r2.5_s5')
rois = np.array(rois)
T1_subnames_int = np.array([np.int(nr) for nr in T1_subnames], dtype=np.int64)
roi_names = np.array(rois)


head_size = StandardScaler().fit_transform(np.nan_to_num(ukbb['25006-2.0'].values[:, None]))  # Volume of grey matter
body_mass = StandardScaler().fit_transform(np.nan_to_num(ukbb['21001-0.0'].values[:, None]))  # BMI
conf_mat = np.hstack([
    np.atleast_2d(head_size), np.atleast_2d(body_mass)])



# load discovery and replication data 
DMN_discovery, DMN_replication, T1_discovery, T1_replication = joblib.load('UKBB_socialbrain_aging/DMN_T1_discovery_replication_traintestsplit_dump_hk')

# inds_disc, inds_mri_disc, b_inds_ukbb_disc, X_disc, ukbb_tar_disc = joblib.load('/Users/hannah/UKBB_socialbrain_aging/discovery_data_dump_hk')
inds_repl, inds_mri_repl, b_inds_ukbb_repl, X_repl, ukbb_tar_repl = joblib.load('UKBB_socialbrain_aging/replication_data_dump_hk')


# conf_mat = conf_mat[b_inds_ukbb_disc]
conf_mat = conf_mat[b_inds_ukbb_repl]


# we need to update the ukbb_tar_disc dataframe from the initial discovery analysis
# OLD_ukbb_tar_disc = ukbb_tar_disc
OLD_ukbb_tar_repl = ukbb_tar_repl



# TAR_COLS_disc = COLS_NAMES[:, 0]
# ukbb_tar_disc = ukbb[TAR_COLS_disc][b_inds_ukbb_disc]

TAR_COLS_repl = COLS_NAMES[:, 0]
ukbb_tar_repl = ukbb[TAR_COLS_repl][b_inds_ukbb_repl]


STOPLOADING









#############
# swap brain region volume to the output variable y
#############



np.random.seed(0)
def my_impute(arr):
    print('Replacing %i NaN values!' % np.sum(np.isnan(arr)))
    arr = np.array(arr)
    b_nan = np.isnan(arr)
    b_negative = arr < 0
    b_bad = b_nan | b_negative

    arr[b_bad] = np.random.choice(arr[~b_bad], np.sum(b_bad))
    return arr




# deconfound DMN volumes for head size and BMI

# SCALED_disc = StandardScaler().fit_transform(DMN_discovery)
SCALED_repl = StandardScaler().fit_transform(DMN_replication)


if DECONF == True:
    from nilearn.signal import clean

    print('Deconfounding BMI & grey-matter space!')
    SCALED_repl = clean(SCALED_repl, confounds=conf_mat, detrend=False, standardize=False)

# sb_disc = pd.DataFrame(SCALED_disc, columns=rois)
sb_repl = pd.DataFrame(SCALED_repl, columns=rois)



# construct input variables X with various social traits
# age_disc = ukbb_tar_disc['21022-0.0'].values.astype(np.int)  # age at recruit.
age_repl = ukbb_tar_repl['21022-0.0'].values.astype(np.int)  # age at recruit.



left = ['FG_L', 'pSTS_L', 'MTV5_L', 'AM_L', 'HC_L', 'NAC_L', 'AI_L', 'SMG_L', 'SMA_L', 'IFG_L', 'Cereb_L', 'TPJ_L', 'MTG_L', 'TP_L']
right = ['FG_R', 'pSTS_R', 'MTV5_R','AM_R', 'HC_R', 'NAC_R', 'AI_R', 'SMG_R', 'SMA_R', 'IFG_R', 'Cereb_R', 'TPJ_R', 'MTG_R', 'TP_R']
middle = ['vmPFC', 'rACC', 'aMCC', 'FP', 'dmPFC',  'PCC', 'Prec', 'pMCC']


# for r in rois[:2]: # sanity check
for r in left:    
    TAR_ROI_repl = r
    print('Current roi is: {}'.format(TAR_ROI_repl))
    y_repl = np.squeeze(SCALED_repl[:, rois == r])
    print(y_repl)

    cur_gender_repl = ukbb_tar_repl['31-0.0'].values.astype(np.int)


    X_repl = pd.DataFrame()

    # 12 SOCIAL INDICATORS (1 = SOCIAL, 0 = UNSOCIAL)

    cur_meta_cat = ukbb_tar_repl['22617-0.0']  # job IDs
    cur_meta_cat = my_impute(cur_meta_cat)
    # top10jobs = cur_meta_cat.value_counts().head(10).index
    cur_meta_cat = np.where(
        np.in1d(cur_meta_cat, [2314., 2315., 7111., 3211., 4123.]), 1, 0)
    X_repl['socialjob'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['4570-0.0'].values # friendship satisfaction 
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat <= 2, dtype=np.int)
    X_repl['highfriendshipsatisfaction'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['4559-0.0'].values # family satisfaction
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat <= 2, dtype=np.int)
    X_repl['highfamilysatisfaction'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['1031-0.0'].values
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat <= 2, dtype=np.int)  # higher, less visits
    X_repl['manyfamilyvisits'] = cur_meta_cat

    cur_meta_cat = np.array(ukbb_tar_repl['709-0.0'].values, dtype=np.int)  # number of ppl in household
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat != 1, dtype=np.int)  # True=living more social since other ppl present
    X_repl['notaloneinhousehold'] = cur_meta_cat

    cur_meta_cat = np.array(ukbb_tar_repl['709-0.0'].values, dtype=np.int)  # number of ppl in household
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat >= 4, dtype=np.int)  # True=living more social since other ppl present
    X_repl['manyinhousehold'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['5057-0.0'].values
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat != 0, dtype=np.int)  # True=living other ppl in same generation
    X_repl['hassiblings'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['2149-0.0'].values # lifetime sex partners (one vs. more)
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat != 1, dtype=np.int)
    X_repl['sexpartners'] = cur_meta_cat

    cur_meta_cat = my_impute(ukbb_tar_repl['2110-0.0'].values)
    cur_meta_cat = np.array(cur_meta_cat == 5, dtype=np.int)  # soc. support
    X_repl['highsocialsupport'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['6160-0.0'].values # sports club
    cur_meta_cat[cur_meta_cat == -7] = 7 # -7 = people with no activity   
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat == 1, dtype=np.int)
    X_repl['sportsclub'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['6160-0.0'].values # weekly social activity
    cur_meta_cat[cur_meta_cat == -7] = 7 
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat != 7, dtype=np.int)
    X_repl['weeklysocialactivity'] = cur_meta_cat

    cur_meta_cat = np.array(ukbb_tar_repl['2020-0.0'].values, dtype=np.int)  # lonely: yes=1, no=0
    cur_meta_cat = my_impute(cur_meta_cat)
    X_repl['loneliness'] = cur_meta_cat


    # 13 DEMOGRAPHIC INDICATORS

    cur_meta_cat = ukbb_tar_repl['845-0.0'].values # age completed full time education
    cur_meta_cat[cur_meta_cat == -2] = 2 # -2 = never went to school 
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat >= 17, dtype=np.int) # 17+ means higher education
    X_repl['agecompletededucation'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['728-0.0'].values # many vehicles (3 or more cars)
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat >= 3, dtype=np.int)
    X_repl['manyvehicles'] = cur_meta_cat

    cur_meta_cat = np.array(ukbb_tar_repl['738-0.0'].values, dtype=np.int)  # income
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat >= 4, dtype=np.int)
    X_repl['highincome'] = cur_meta_cat

    cur_meta_cat = np.array(ukbb_tar_repl['4537-0.0'].values, dtype=np.int) # work/job satisfaction
    cur_meta_cat = ukbb_tar_repl['4537-0.0'].values
    cur_meta_cat[cur_meta_cat == 7] = -7 # kick out 7 (not employed)
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat <= 2, dtype=np.int)
    X_repl['highjobsatisfaction'] = cur_meta_cat

    cur_meta_cat = np.array(ukbb_tar_repl['4548-0.0'].values, dtype=np.int) # health satisfaction
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat <= 2, dtype=np.int)
    X_repl['highhealthsatisfaction'] = cur_meta_cat

    cur_meta_cat = np.array(ukbb_tar_repl['4581-0.0'].values, dtype=np.int) # financial satisfaction
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat <= 2, dtype=np.int)
    X_repl['highfinancialsatisfaction'] = cur_meta_cat

    cur_meta_cat = np.array(ukbb_tar_repl['767-0.0'].values, dtype=np.int) # Length of working week for main job 
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat > 40, dtype=np.int)
    X_repl['manyworkinghours'] = cur_meta_cat

    cur_meta_cat = np.array(ukbb_tar_repl['796-0.0'].values, dtype=np.int) # Distance between home and job workplace 
    cur_meta_cat[cur_meta_cat == -10] = 10 # -10 represents 'less than 1 mile'
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat >= 11, dtype=np.int)
    X_repl['fardistanceworkhome'] = cur_meta_cat

    cur_meta_cat = np.array(ukbb_tar_repl['806-0.0'].values, dtype=np.int) # Job involves mainly walking or standing 
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat != 1, dtype=np.int)
    X_repl['walkingstandingjob'] = cur_meta_cat

    cur_meta_cat = np.array(ukbb_tar_repl['816-0.0'].values, dtype=np.int) # Job involves heavy manual or physical work 
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat == 1, dtype=np.int)
    X_repl['manualjob'] = cur_meta_cat

    cur_meta_cat = np.array(ukbb_tar_repl['1677-0.0'].values, dtype=np.int) # Breastfed
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat == 1, dtype=np.int)
    X_repl['breastfed'] = cur_meta_cat

    cur_meta_cat = np.array(ukbb_tar_repl['4674-2.0'].values, dtype=np.int) # private health care
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat != 4, dtype=np.int)
    X_repl['privatehealthcare'] = cur_meta_cat

    cur_meta_cat = np.array(ukbb_tar_repl['20016-0.0'].values, dtype=np.int) # fluid IQ score
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat >= 7, dtype=np.int)
    X_repl['highIQ'] = cur_meta_cat


    # 15 PERSONALITY INDICATORS (Note: 1=yes they have the trait - applies also to loneliness)

    cur_meta_cat = ukbb_tar_repl['1180-0.0'].values # morning, evening person
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat <= 2, dtype=np.int) # 1 = morning person
    X_repl['morningevening'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['1920-0.0'].values # mood swings
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat == 1, dtype=np.int)
    X_repl['moodswings'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['1930-0.0'].values # miserableness
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat == 1, dtype=np.int)
    X_repl['miserableness'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['1940-0.0'].values # Irritability
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat == 1, dtype=np.int)
    X_repl['irritability'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['1950-0.0'].values # Sensitivity / hurt feelings
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat == 1, dtype=np.int)
    X_repl['sensitivity'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['1960-0.0'].values # Fed-up feelings
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat == 1, dtype=np.int)
    X_repl['fedup'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['1970-0.0'].values # Nervous feelings
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat == 1, dtype=np.int)
    X_repl['nervous'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['1980-0.0'].values # worrier / anxious feelings
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat == 1, dtype=np.int)
    X_repl['worrier'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['1990-0.0'].values # tense / 'highly strung'
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat == 1, dtype=np.int)
    X_repl['tense'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['2000-0.0'].values # worry too long after embarrasment
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat == 1, dtype=np.int)
    X_repl['embarrasment'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['2010-0.0'].values # suffer from 'nerves'
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat == 1, dtype=np.int)
    X_repl['sufferfromnerves'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['2030-0.0'].values # guilty feelings
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat == 1, dtype=np.int)
    X_repl['guilty'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['2040-0.0'].values # risk taking
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat == 1, dtype=np.int)
    X_repl['risktaking'] = cur_meta_cat

    cur_meta_cat = ukbb_tar_repl['20127-0.0'].values # neuroticism score
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat > 4, dtype=np.int)
    X_repl['highneuroticism'] = cur_meta_cat

    cur_meta_cat = np.array(ukbb_tar_repl['4526-0.0'].values, dtype=np.int) # happiness
    cur_meta_cat = my_impute(cur_meta_cat)
    cur_meta_cat = np.array(cur_meta_cat <= 2, dtype=np.int)
    X_repl['happymood'] = cur_meta_cat

    X_repl['age'] = StandardScaler().fit_transform(age_repl[:, None])[:, 0]


    # outlier detection
    inds_inlier_repl = (y_repl >= -2.5) & (y_repl <= +2.5)
    X_repl = X_repl[inds_inlier_repl]
    y_repl = y_repl[inds_inlier_repl]
    cur_gender_repl = cur_gender_repl[inds_inlier_repl]
    female = cur_gender_repl == 0
    male = cur_gender_repl == 1

    # X_disc = X_disc[:90]  # for sanity checks
    # y_disc = y_disc[:90]  # for sanity checks
    # cur_gender_disc = cur_gender_disc[:90] # for sanity checks

    lr = LinearRegression(fit_intercept=False)
    lr.fit(X_repl, y_repl)
    r2 = lr.score(X_repl, y_repl)
    print('Baseline accuracy as in-sample R^2: %1.4f' % (r2))


    import pymc3 as pm 

    pm_varnames = []
    n_meta_cat = 2
    with pm.Model() as hierarchical_model:

        hyper_mu = pm.Normal('mu_sex', mu=0., sd=1, shape=n_meta_cat)
        hyper_sigma_b = pm.HalfCauchy('sigma_sex', 1, shape=n_meta_cat)
        pm_varnames.append('mu_sex')
        pm_varnames.append('sigma_sex')

        beh_est = 0
        for i_beh, behav_name in enumerate(X_repl.columns):
            pm_varnames.append(behav_name)
            cur_beta_param = pm.Normal(behav_name, mu=hyper_mu, sd=hyper_sigma_b,
                shape=n_meta_cat)

            beh_est = beh_est + cur_beta_param[cur_gender_repl] * X_repl[behav_name]
            

        eps = pm.HalfCauchy('eps', 5)  # Model error
        group_like = pm.Normal('beh_like', mu=beh_est, sd=eps, observed=y_repl)


    with hierarchical_model:
        hierarchical_trace = pm.sample(draws=5000, n_init=1000, init='advi',
            chains=1, cores=1, progressbar=True,
            random_seed=[123])  # one per chain needed

    output_name = TAR_ROI_repl
    subgroup_labels = ['female', 'male']
    for cur_trait in pm_varnames:
        from matplotlib.lines import Line2D
        import arviz as az
        plt.close('all')
        THRESH = 0.5
        n_last_chains = 1000
        #try:
        fig = pm.plot_posterior(hierarchical_trace[-n_last_chains:], varnames=[cur_trait], kind='hist', credible_interval=0.95, round_to=4)
            # fig = pm.plot_posterior(hierarchical_trace[-n_last_chains:], varnames=[cur_roi])
            # try:
            #     for i_higher_cat in range(n_meta_cat):
            #         fig[i_higher_cat].set_xlim(-THRESH, THRESH)  # make plots more comparable
            # except:
            #     pass
        plt.tight_layout()
        plt.savefig('%s/%s_%s_posterior_repl2.png' % (OUT_DIR, output_name, cur_trait), dpi=150)
        plt.savefig('%s/%s_%s_posterior_repl2.pdf' % (OUT_DIR, output_name, cur_trait), dpi=150)
        plt.close() 

        fig = az.plot_trace(hierarchical_trace[-n_last_chains:], compact=True, var_names=[cur_trait])
        fig[0][0].get_lines()[0].set_color('magenta')
        fig[0][1].get_lines()[0].set_color('magenta')
        fig[0][0].get_lines()[1].set_color('blue')
        fig[0][1].get_lines()[1].set_color('blue') 
        post_lines = fig[0][0].get_lines()
        custom_lines = [Line2D([0], [0], color=l.get_c(), lw=4) for l in post_lines]
            # fig = pm.traceplot(hierarchical_trace[-n_last_chains:], varnames=[cur_roi])
            # max_abs_mode = np.max(np.abs(hierarchical_trace[-n_last_chains:][cur_roi].mean(0)))
            # try:
            #     if max_abs_mode < THRESH and not 'nuisance' in cur_roi:
            #         fig[0][0].set_xlim(-THRESH, THRESH)  # make plots more comparable
            # except:
            #     pass
            # post_lines = fig[0][0].get_lines()
            # custom_lines = [Line2D([0], [0], color=l.get_c(), lw=4) for l in post_lines]
        if subgroup_labels is None:
            subgroup_labels = ['subgroup %i' % i for i in range(len(custom_lines))]
        fig[0][0].legend(custom_lines, subgroup_labels, loc='upper left', prop={'size': 7.5})
        max_abs_mode = np.max(np.abs(hierarchical_trace[-n_last_chains:][cur_trait].mean(0)))
        if max_abs_mode < THRESH and not 'nuisance' in cur_trait:
            fig[0][0].set_xlim(-THRESH, THRESH)  # make plots more comparable
        #except:
        #    pass
            plt.savefig('%s/%s_%s_trace_partial_corr_repl2.png' % (OUT_DIR, output_name, cur_trait), dpi=150)
            plt.savefig('%s/%s_%s_trace_partial_corr_repl2.pdf' % (OUT_DIR, output_name, cur_trait), dpi=150)
            plt.close()
        #except:
        #    pass

    t = hierarchical_trace

    from sklearn.metrics import r2_score
    Y_ppc_insample = pm.sample_ppc(hierarchical_trace[-n_last_chains:], 500, hierarchical_model, random_seed=123)['beh_like']
    y_pred_insample = Y_ppc_insample.mean(axis=0)
    ppc_insample = r2_score(y_repl, y_pred_insample)
    out_str = 'PPC in sample R^2: %2.6f' % (ppc_insample)
    print(out_str)
    plt.figure(figsize=(7, 8))
    sns.regplot(x=y_repl, y=y_pred_insample, fit_reg=True, ci=95,
        line_kws={'color':'black', 'linewidth':4})
    plt.xlabel('real output variable')
    plt.ylabel('predicted output variable')
    plt.title(out_str + ' (%i samples)' % len(X_repl))
    plt.savefig('%s/%s_r2scatter_repl2.png' % (OUT_DIR, output_name), dpi=150)
    plt.savefig('%s/%s_r2scatter_repl2.pdf' % (OUT_DIR, output_name), dpi=150)

    plt.figure()
    plt.hist([it.mean() for it in Y_ppc_insample.T], bins=19, alpha=0.35,
        label='predicted output')
    plt.hist(y_repl, bins=19, alpha=0.5, label='original output')
    plt.legend(loc='upper right')
    plt.title('Posterior predictive check: predictive distribution', fontsize=10)
    plt.savefig('%s/%s_ppc_repl2.png' % (OUT_DIR, output_name), dpi=150)
    plt.savefig('%s/%s_ppc_repl2.pdf' % (OUT_DIR, output_name), dpi=150)


    loo_res = pm.loo(hierarchical_trace, hierarchical_model, progressbar=True, pointwise=True)
    print('LOO point-wise deviance: mean=%.2f+/-%.2f' % (np.mean(loo_res[4]), np.std(loo_res[4])))

    pd.DataFrame(Y_ppc_insample).to_csv('%s/%s_Y_ppc_insample_repl2.csv' % (OUT_DIR, output_name))
    pd.DataFrame(y_pred_insample).to_csv('%s/%s_y_pred_insample_repl2.csv' % (OUT_DIR, output_name))       
    pd.DataFrame(loo_res).to_csv('%s/%s_loo_res_repl2.csv' % (OUT_DIR, output_name))
    joblib.dump([ppc_insample], os.path.join(OUT_DIR, output_name + '_ppc_insample_repl2_dump'), compress=9)


    # female ppc

    female_Y_ppc_insample = Y_ppc_insample.T[female]
    female_y_pred_insample = female_Y_ppc_insample.mean(axis=1)
    ppc_insample = r2_score(y_repl[female], female_y_pred_insample)
    out_str = 'PPC in sample R^2: %2.6f' % (ppc_insample)
    print(out_str)
    plt.figure(figsize=(7, 8))
    sns.regplot(x=y_repl[female], y=female_y_pred_insample, fit_reg=True, ci=95,
        line_kws={'color':'black', 'linewidth':4})
    plt.xlabel('real output variable')
    plt.ylabel('predicted output variable')
    plt.title(out_str + ' (%i female samples)' % len(X_repl[female]))
    plt.savefig('%s/%s_r2scatter_FEMALE_repl2.png' % (OUT_DIR, output_name), dpi=150)
    plt.savefig('%s/%s_r2scatter_FEMALE_repl2.pdf' % (OUT_DIR, output_name), dpi=150)

    plt.figure()
    plt.hist([it.mean() for it in female_Y_ppc_insample], bins=19, alpha=0.35,
        label='predicted output')
    plt.hist(y_repl, bins=19, alpha=0.5, label='original output')
    plt.legend(loc='upper right')
    plt.title('Posterior predictive check: predictive distribution for females', fontsize=10)
    plt.savefig('%s/%s_ppc_FEMALE_repl2.png' % (OUT_DIR, output_name), dpi=150)
    plt.savefig('%s/%s_ppc_FEMALE_repl2.pdf' % (OUT_DIR, output_name), dpi=150)

    female_loo_res_mean = pd.Series(np.mean(loo_res[4][female]), name='female_loo_res_mean')
    female_loo_res_std = pd.Series(np.std(loo_res[4][female]), name='female_loo_res_std')
    print('LOO point-wise deviance for females: mean=%.2f+/-%.2f' % (female_loo_res_mean, female_loo_res_mean))

    pd.DataFrame(Y_ppc_insample).to_csv('%s/%s_FEMALE_Y_ppc_insample_repl2.csv' % (OUT_DIR, output_name))
    pd.DataFrame(y_pred_insample).to_csv('%s/%s_FEMALE_y_pred_insample_repl2.csv' % (OUT_DIR, output_name))       
    joblib.dump([ppc_insample], os.path.join(OUT_DIR, output_name + '_FEMALE_ppc_insample_repl2_dump'), compress=9)
    pd.concat([female_loo_res_mean, female_loo_res_std]).to_csv('%s/%s_FEMALE_loo_res_repl2.csv' % (OUT_DIR, TAR_ROI_repl))

    # male ppc

    male_Y_ppc_insample = Y_ppc_insample.T[male]
    male_y_pred_insample = male_Y_ppc_insample.mean(axis=1)
    ppc_insample = r2_score(y_repl[male], male_y_pred_insample)
    out_str = 'PPC in sample R^2: %2.6f' % (ppc_insample)
    print(out_str)
    plt.figure(figsize=(7, 8))
    sns.regplot(x=y_repl[male], y=male_y_pred_insample, fit_reg=True, ci=95,
        line_kws={'color':'black', 'linewidth':4})
    plt.xlabel('real output variable')
    plt.ylabel('predicted output variable')
    plt.title(out_str + ' (%i male samples)' % len(X_repl[male]))
    plt.savefig('%s/%s_r2scatter_MALE_repl2.png' % (OUT_DIR, output_name), dpi=150)
    plt.savefig('%s/%s_r2scatter_MALE_repl2.pdf' % (OUT_DIR, output_name), dpi=150)

    plt.figure()
    plt.hist([it.mean() for it in male_Y_ppc_insample], bins=19, alpha=0.35,
        label='predicted output')
    plt.hist(y_repl, bins=19, alpha=0.5, label='original output')
    plt.legend(loc='upper right')
    plt.title('Posterior predictive check: predictive distribution for males', fontsize=10)
    plt.savefig('%s/%s_ppc_MALE_repl2.png' % (OUT_DIR, output_name), dpi=150)
    plt.savefig('%s/%s_ppc_MALE_repl2.pdf' % (OUT_DIR, output_name), dpi=150)

    male_loo_res_mean = pd.Series(np.mean(loo_res[4][male]), name='male_loo_res_mean')
    male_loo_res_std = pd.Series(np.std(loo_res[4][male]), name='male_loo_res_std')
    print('LOO point-wise deviance for males: mean=%.2f+/-%.2f' % (male_loo_res_mean, male_loo_res_mean))

    pd.DataFrame(Y_ppc_insample).to_csv('%s/%s_MALE_Y_ppc_insample_repl2.csv' % (OUT_DIR, output_name))
    pd.DataFrame(y_pred_insample).to_csv('%s/%s_MALE_y_pred_insample_repl2.csv' % (OUT_DIR, output_name))       
    pd.DataFrame(loo_res).to_csv('%s/%s_MALE_loo_res_repl2.csv' % (OUT_DIR, output_name))    
    joblib.dump([ppc_insample], os.path.join(OUT_DIR, output_name + '_MALE_ppc_insample_repl2_dump'), compress=9)
    pd.concat([male_loo_res_mean, male_loo_res_std]).to_csv('%s/%s_MALE_loo_repl2.csv' % (OUT_DIR, TAR_ROI_repl))

    joblib.dump([hierarchical_trace, hierarchical_model], os.path.join(OUT_DIR, output_name + '_repl2_dump'), compress=9)
    pd.DataFrame(X_repl).to_csv('%s/%s_indicators_repl2.csv' % (OUT_DIR, output_name))
    pd.DataFrame(y_repl).to_csv('%s/%s_output_repl2.csv' % (OUT_DIR, output_name))



