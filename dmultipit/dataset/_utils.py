import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PowerTransformer

RAD_JOB_TAG = "filtered-radiomics"


def select_outlier_stable_features(df, cutoff):
    """
    Identify outliers in each feature (i.e., with absolute z-score above a predefined cutoff) and filter out features
    with outliers

    Parameters
    ----------
    df: pandas dataframe
        Radiomic data

    cutoff: float > 0
        Minimum absolute z-score to consider samples as outliers

    Returns
    -------
    outlier_counts: pandas dataframe of shape (n_features, 1)
        Number of outliers for each feature

    selected_features: pandas Index
        Selected features without outliers
    """
    df = df.reset_index()
    df = df[df["job_tag"] == RAD_JOB_TAG]

    x = df.copy(deep=True).drop(columns=["main_index", "job_tag", "site", "index"], errors="ignore")
    x[(np.abs(zscore(x.values.astype(float))) > cutoff)] = np.nan
    outlier_counts = x.isna().sum()

    selected_features = outlier_counts[outlier_counts == 0].index
    return outlier_counts, selected_features


def select_by_interlesion_variance(df, cutoff=0.1):
    """
    Compute robustness score for each feature (i.e., ratio of average inter average inter-lesion variance across the
    ten perturbations and the feature intra-lesion variance average across the entire dataset) and  select features
    with a score lower than a predefined cutoff.

    Parameters
    ----------
    df: pandas dataframe
        Radiomic data

    cutoff: float in [0, 1]
        Minimum robustness score to consider a feature not robust

    Returns
    -------
    robustness_score: pandas dataframe of shape (n_features, 1)
        Robustness score for each feature

    robust_features: pandas Index
        Selected robust features
    """
    df = df.reset_index().set_index(["main_index", "lesion_index"])
    df = df[df["job_tag"] == "pertubation-radiomics"]

    x = df.copy(deep=True).drop(columns=["job_tag", "site"], errors="ignore")
    x = x.loc[:, x.nunique() > 1]  # remove constant features

    robustness_score = (
        x.groupby(level=[0, 1]).agg(np.std).mean()
        / x.groupby(level=[0, 1]).agg(np.mean).std()
    )

    robust_features = robustness_score[robustness_score < cutoff].index

    # variance_ranks = interlesion_variance.rank(method='min').rename("interlesion_variance_rank")

    return robustness_score, robust_features


def select_radiomics_features_elastic(train_df, outcomes_df, l1_C=0.1, outlier_cutoff=6, robustness_cutoff=0.15):
    """
    Select radiomic features with elasticnet logistic regression (i.e., non-zero coefficients), looking only at robust
    features, without outliers.

    Parameters
    ----------
    train_df: pandas dataframe
        Radiomic data

    outcomes_df: pandas dataframe
        Outcome/target label for each sample

    l1_C: float
        Inverse of regularization strenght for elasticnet penalty in logistic regression
        (see sklearn.linear_model.LogisticRegression).

    outlier_cutoff: float > 0
        Minimum absolute z-score to consider samples as outliers

    robustness_cutoff: float in [0, 1]
        Minimum robustness score to consider a feature not robust

    Returns
    -------
    selected_features: pandas Index
        Selected radiomic features

    References
    ----------
    1. Zwanenburg, A. et al. Assessing robustness of radiomic features by image perturbation. Sci. Rep. 9, 1–10 (2019).
    (https://doi.org/10.1038/s41598-018-36938-4)

    2. Vanguri, R.S. et al. Multimodal integration of radiology, pathology and genomics for prediction of response to
    PD-(L)1 blockade in patients with non-small cell lung cancer. Nat Cancer 3, 1151–1164 (2022).
    (https://doi.org/10.1038/s43018-022-00416-8)
    """
    # 1. Select robust features without outliers
    _, sel_fx_outlier = select_outlier_stable_features(train_df, cutoff=outlier_cutoff)
    _, sel_fx_robust = select_by_interlesion_variance(train_df, cutoff=robustness_cutoff
                                                      )
    fx_to_use = sorted(set(sel_fx_outlier).intersection(set(sel_fx_robust)))

    # 2. Select remaining features with elastic net logistic coefficients (i.e., non-zero coefficient)
    train_df = train_df[train_df["job_tag"] == RAD_JOB_TAG].drop(columns=["job_tag", "site"])[fx_to_use]
    Y_train = outcomes_df.loc[train_df.index]

    # 2.1 Pre-process data with power transform to deal with skewed data
    scaler = PowerTransformer()
    X_train = scaler.fit_transform(train_df)

    # 2.2 Fit logistic regression with elastic net penalty
    clf_lr = LogisticRegression(
        penalty="elasticnet",
        random_state=0,
        max_iter=2500,
        solver="saga",
        l1_ratio=0.5,
        C=l1_C,
        class_weight="balanced",
    )
    clf_lr.fit(X_train, Y_train)

    # 2.3 Keep features with non-zero coefficients
    selected_features = train_df.columns[np.where(clf_lr.coef_[0])[0]]

    # df_coef = pd.DataFrame(clf_lr.coef_[0], index=train_df.columns, columns=['coef']).abs().sort_values(by='coef',
    #            ascending=False)

    return selected_features  # , df_coef
