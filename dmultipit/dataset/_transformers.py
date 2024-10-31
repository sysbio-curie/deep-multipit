import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.impute._base import _BaseImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, PowerTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from dmultipit.dataset._utils import select_radiomics_features_elastic


class CustomOmicsImputer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for imputing missing values and encoding categorical features in omics data.

    Parameters
    ----------
    site_feature : int
        Index of the site feature to be imputed and encoded.

    min_frequency : float, default=0.1
        Minimum frequency threshold for encoding infrequent categories.

    Attributes
    ----------
    imputer_ : KNNImputer
        Fitted KNNImputer for imputing missing values.

    encoder_ : OneHotEncoder
        Fitted OneHotEncoder for categorical encoding.

    len_encoding_ : int
        Length of the encoding after transformation.

    len_features_ : int
        Number of omics features after transformation.
    """

    def __init__(self, site_feature, min_frequency=0.1):
        self.site_feature = site_feature
        self.min_frequency = min_frequency
        self.imputer_ = None
        self.encoder_ = None
        self.len_encoding_ = None
        self.len_features_ = None

    def fit(self, X, y=None):
        """
        Fit the CustomOmicsImputer to the provided data.

        Parameters
        ----------
        X : 2D array of shape (n_samples, n_features)
            Training data.

        y : Ignored

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.imputer_ = KNNImputer(n_neighbors=1)
        X[:, self.site_feature] = self.imputer_.fit_transform(X)[:, self.site_feature]
        self.encoder_ = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=self.min_frequency,
                                      sparse_output=False).fit(X[:, self.site_feature].reshape(-1, 1))
        temp = self.encoder_.transform(X[:, self.site_feature].reshape(-1, 1))
        self.len_features_ = (X.shape[1] - 1) + temp.shape[1]
        return self

    def transform(self, X):
        """
        Transform the input data by imputing missing values and encoding categorical features.

        Parameters
        ----------
        X : 2D array of shape (n_samples, n_features)
            Data to be transformed.

        Returns
        -------
        2D array of shape (n_samples, (n_features-1) + len_encoding_)
            Transformed data after imputation and encoding.
        """
        X[:, self.site_feature] = self.imputer_.transform(X)[:, self.site_feature]
        b = self.encoder_.transform(X[:, self.site_feature].reshape(-1, 1))
        self.len_encoding_ = b.shape[1]
        a = np.delete(X, self.site_feature, 1)
        return np.hstack((a, b))


class CustomImputer(_BaseImputer):
    """
    Custom imputer for missing values which deals with categorical variables with most frequent imputation and with
    numerical variables with median imputation

    Parameters
    ----------
    categoricals: list of integers.
        List of indexes associated to the categorical columns with missing values. If None, no categorical column is
        considered.

    numericals: list of integers.
        List of indexes associated to the numerical columns with missing values. If None, no numerical column is
        considered.

    Attributes
    ----------
    mask_cat_: 1D array of booleans.
        Boolean mask indicating the categorical columns with missing values.

    mask_num_: 1D array of booleans.
        Boolean mask indicating the numerical columns with missing values.
    """

    def __init__(self, categoricals, numericals):
        super(CustomImputer, self).__init__()
        self.categoricals = categoricals
        self.numericals = numericals
        if self.categoricals is not None:
            self.imputer_cat = SimpleImputer(strategy="most_frequent")
        if self.numericals is not None:
            self.imputer_num = SimpleImputer(strategy="median")
        assert (self.categoricals is not None) | (self.numericals is not None), ""

    def fit(self, X, y=None):
        """
        Fit the custom imputer.
        Parameters
        ----------
        X: 2D array of shape (n_samples, n_features)

        y: Ignored

        Returns
        -------
        self: object
            Fitted estimator.
        """
        if self.categoricals is not None:
            self.mask_cat_ = np.zeros(X.shape[1], bool)
            self.mask_cat_[self.categoricals] = True
            self.imputer_cat.fit(X[:, self.mask_cat_])
        if self.numericals is not None:
            self.mask_num_ = np.zeros(X.shape[1], bool)
            self.mask_num_[self.numericals] = True
            self.imputer_num.fit(X[:, self.mask_num_])
        return self

    def transform(self, X):
        """
        Impute missing values in X.

        Parameters
        ----------
        X: 2D array of shape (n_samples, n_features)

        Returns
        -------
        X_imputed: 2D array of shape (n_samples, n_features)
            X with imputed values
        """
        X_imputed = np.copy(X)
        if self.categoricals is not None:
            X_imputed[:, self.mask_cat_] = self.imputer_cat.transform(
                X[:, self.mask_cat_]
            )
        if self.numericals is not None:
            X_imputed[:, self.mask_num_] = self.imputer_num.transform(
                X[:, self.mask_num_]
            )
        return np.float32(X_imputed)


class CustomSelection(BaseEstimator, TransformerMixin):
    """
    Custom univariate selection for classification (based on AUC) tasks.

    Parameters
    ----------
    threshold: float in [O.5, 1].
        Threshold for the metric (i.e., AUC). Features associated to a metric lower than this threshold will
        not be selected. If None, no threshold is applied. The default is None.

    max_corr: float in [0, 1]
        This parameter sets the threshold for the Pearson correlation. When analyzing feature performance, starting from
        the top-performing feature, all features with a Pearson correlation above this threshold are excluded.
        Subsequently, the algorithm considers the second-best performing feature among those that were not filtered out,
        and continues this process iteratively. If max_corr=1, no threshold is applied. The default is 0.8.

    max_number: int.
        Maximum number of selected features. If the number of remaining features after the different filtering steps is
        lower than max_number or if max_number is None all the remaining features are kept. The default is None.

    Attributes
    ----------
    features_: list of integers.
        List of indexes corresponding to the selected features.

    n_select_modalities_: list of integers.
        Number of selected features for each modality.
    """

    def __init__(self, threshold=None, max_corr=0.8, max_number=None):
        self.threshold = threshold
        self.max_corr = max_corr
        self.max_number = max_number
        self.n_select_modalities_ = None

    def fit(self, X, y, modalities=None):
        """
        Fit the custom selection.

        Parameters
        ----------
        X: 2D array of shape (n_samples, n_features).

        y: 1D array of shape (n_samples).
            Binary outcome for the classification task.

        modalities: 1D array of shape (n_features).
            This parameter deals with scenarios where features from different modalities are concatenated. It comprises
            integer labels indicating the membership of each feature to a specific modality. If `max_number` is not
            None, and the remaining features outnumber `max_number`, the algorithm selects the top
            `max_number/n_modalities` performing features within each modality. If None the different modalities are
            ignored. The default is None.

        Returns
        -------
        self: object
            Fitted estimator.
        """
        # Xmasked = X[np.sum(np.isnan(X), axis=1) == 0, :]
        # ymasked = y[np.sum(np.isnan(X), axis=1) == 0]
        # self.features_ = np.arange(Xmasked.shape[1])
        self.features_ = np.arange(X.shape[1])
        scores = np.zeros(X.shape[1])

        for i in range(X.shape[1]):
            mask = np.isnan(y) | np.isnan(X[:, i])
            # auc = roc_auc_score(ymasked, Xmasked[:, i])
            auc = roc_auc_score(y[~mask], X[~mask][:, i])
            scores[i] = max(auc, 1 - auc)

        if self.threshold is not None:
            self.features_ = self.features_[scores >= self.threshold]
            assert len(self.features_) > 0
            scores = scores[scores >= self.threshold]
        self.features_ = self.features_[np.argsort(scores)[::-1]]

        # corr = np.abs(np.corrcoef(Xmasked[:, self.features_], rowvar=False))
        # corr = np.abs(_pearsonccs(Xmasked[:, self.features_], rowvar=False))
        if self.max_corr < 1:
            corr = np.abs(pd.DataFrame(X[:, self.features_]).corr()).values
            delete = []
            for i in range(len(self.features_) - 1):
                if i not in delete:
                    delete += list((i + 1) + np.where(corr[i, i + 1:] > self.max_corr)[0])
            delete = np.unique(delete)
            if len(delete) > 0:
                self.features_ = np.delete(self.features_, delete)

        if modalities is not None:
            if (self.max_number is not None) and (len(self.features_) > self.max_number):
                n_modalities = len(np.unique(modalities))
                n_select_modalities = self.max_number // n_modalities
                modalities_ordered = modalities[self.features_]
                self.n_select_modalities_ = []
                list_features = []
                for m in np.unique(modalities):
                    temp = self.features_[modalities_ordered == m][:n_select_modalities]
                    self.n_select_modalities_.append(len(temp))
                    list_features += list(temp)
                self.features_ = np.array(list_features)
            else:
                modalities_ordered = modalities[self.features_]
                self.n_select_modalities_ = []
                list_features = []
                for m in np.unique(modalities):
                    temp = self.features_[modalities_ordered == m]
                    self.n_select_modalities_.append(len(temp))
                    list_features += list(temp)
                self.features_ = np.array(list_features)
        else:
            if self.max_number is not None and len(self.features_) > self.max_number:
                self.features_ = self.features_[:self.max_number]
        return self

    def transform(self, X):
        """
        Select features

        Parameters
        ----------
        X: 2D array of shape (n_samples, n_features)

        Returns
        -------
            2D array of shape (n_samples, n_selected_features)
        """
        return X[:, self.features_]

    def transform_multimodal(self, X):
        """
        Select features and return one data set per modality.

        Parameters
        ----------
        X: 2D array of shape (n_samples, n_features)

        Returns
        -------
            List of 2D arrays of shape (n_samples, n_selected_features (for that modality))
        """
        if self.n_select_modalities_ is None:
            raise ValueError("The modality each feature belongs to shoud have been specified with the modalities "
                             "argument of the fit method.")
        X_all_selected = X[:, self.features_]
        output = []
        n = 0
        for size in self.n_select_modalities_:
            output.append(X_all_selected[:, n:n + size])
            n += size
        return output


class CustomScaler(BaseEstimator, TransformerMixin):
    """
    A custom data scaler that allows for different scaling strategies and can be applied on a subset of features.

    Parameters
    ----------
    features: 1D array of shape (n_features,)
        Indices or labels of the features to be scaled. If None, all features are considered for scaling. The default is
        None.

    strategy: {'standardize', 'robust', 'minmax'}
        The strategy used for scaling. The default is 'standardize'.

    Attributes
    ----------
    scaler_: object
        Fitted scaler based on the specified strategy.
    """

    def __init__(self, features=None, strategy='standardize'):
        self.features = features
        self.strategy = strategy

    def fit(self, X, y=None):
        """
        Fit the custom scaler.

        Parameters
        ----------
        X : 2D array of shape (n_samples, n_features)
            Training data.

        y : Ignored

        Returns
        -------
        self : object
            Fitted estimator.
        """

        if self.strategy == 'standardize':
            self.scaler_ = StandardScaler()
        elif self.strategy == 'robust':
            self.scaler_ = RobustScaler()
        elif self.strategy == 'minmax':
            self.scaler_ = MinMaxScaler()
        else:
            raise ValueError(
                "Only 'standardize', 'robust', or 'minmax' are available for the scaling strategy"
            )

        # deal with cases where X is empty ?
        if X.shape[1] == 0:
            self.scaler_ = None
        else:
            if self.features is None:
                self.scaler_.fit(X)
            else:
                self.scaler_.fit(X[:, self.features])
        return self

    def transform(self, X):
        """
        Transform the input data using the fitted scaler.

        Parameters
        ----------
        X : 2D array of shape (n_samples, n_features)
            Data to be transformed.

        Returns
        -------
        Xnew : 2D array of shape (n_samples, n_features)
            Transformed data.
        """

        # deal with cases where X is empty ?
        if self.scaler_ is None:
            Xnew = np.copy(X)
        else:
            if self.features is None:
                Xnew = self.scaler_.transform(X)
            else:
                Xnew = np.copy(X)
                Xnew[:, self.features] = self.scaler_.transform(Xnew[:, self.features])
        return Xnew


class CustomLogTransform(BaseEstimator, TransformerMixin):
    """
    A custom transformer for applying a logarithmic transformation to specified features.

    Parameters
    ----------
    features : 2D array of shape (n_features,)
        Indices or labels of the features to be transformed. If None, logarithmic transformation is applied to all
        features. The default is None.

    Attributes
    ----------
    fitted_ : bool
        Indicates whether the transformer has been fitted.
    """

    def __init__(self, features=None):
        self.features = features

    def fit(self, X, y=None):
        """
        Fit the transformer to the provided data.

        Parameters
        ----------
        X : 2D array of shape (n_samples, n_features)
            Training data.

        y : Ignored.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Apply a logarithmic transformation to the input data.

        Parameters
        ----------
        X : 2D array of shape (n_samples, n_features)
            Data to be transformed.

        Returns
        -------
        Xnew : 2D array of shape (n_samples, n_features)
            Transformed data after applying the logarithmic transformation.
        """

        if self.features is None:
            Xnew = np.log(X + 1)
        else:
            Xnew = np.copy(X)
            Xnew[:, self.features] = np.log(Xnew[:, self.features] + 1)
        return Xnew


class CustomPCA(BaseEstimator, TransformerMixin):
    """
    A custom transformer applying PCA on input data (dealing with nan values).

    Parameters
    ----------
    n_components : int or None
        Number of components to keep. If `None`, all components are kept.

    whiten : bool
        When True, the components are whitened. The default is False

    Attributes
    ----------
    pca_ : PCA
        Fitted PCA object based on the provided parameters.
    """

    def __init__(self, n_components, whiten):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X, y=None):
        """
        Fit the PCA transformer.

        Parameters
        ----------
        X : 2D array of shape (n_samples, n_features)
            Training data.

        y : Ignored.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        self.pca_ = PCA(n_components=self.n_components, whiten=self.whiten)
        # Missing values are disregarded in fit
        self.pca_.fit(X[np.sum(np.isnan(X), axis=1) == 0, :])
        return self

    def transform(self, X):
        """
        Apply PCA transformation to the input data.

        Parameters
        ----------
        X : 2D array of shape (n_samples, n_features)
            Data to be transformed.

        Returns
        -------
        X_pca : 2D array of shape (n_samples, n_components)
            Transformed data after applying PCA.
        """

        return self.pca_.transform(X)


class CustomVIF(BaseEstimator, TransformerMixin):
    """
    Select features with Variance Inflation Factor (VIF) (i.e., measure of multicolinearity between features), removing
    features that contribute the most to multicolinearity (i.e., high VIF values)

    Parameters
    ----------
    cutoff: int > 1
        Maximum VIF value to consider. Features with a VIF value lower than this cutoff will be selected.

    power_transform: bool
        If True, applies power transformation before VIF analysis (deal with skewed data).

    Attributes
    ----------
    features_: list of int
        Indexes of selected features.

    """
    def __init__(self, cutoff=5, power_transform=False):
        self.cutoff = cutoff
        self.power_transform = power_transform

    def fit(self, X, y=None):
        """
        Select features with low VIF values

        Parameters:
        X: 2D array of shape (n_samples, n_features)
            Training data.

        y: ignored

        Results
        -------
        self: object
            Fitted estimator.
        """
        if self.power_transform:
            X_temp = PowerTransformer().fit_transform(np.copy(X))
        else:
            X_temp = np.copy(X)

        self.features_ = np.arange(X.shape[1])
        while True:
            vit_values = _run_vif_analysis(X_temp)
            max_feature = np.argmax(vit_values[1:])
            if vit_values[max_feature] < self.cutoff:
                break
            self.features_ = np.delete(self.features_, max_feature)
            X_temp = np.delete(X_temp, max_feature, axis=1)
        return self

    def transform(self, X):
        """
        Select features

        Parameters
        ---------
        X: 2D array of shape (n_samples, n_features)
            Data to transform

        Returns
        -------
            2D array of shape (n_samples, n_selected_features)
        """
        return X[:, self.features_]


def _run_vif_analysis(X):
    """
    Compute VIF for each feature (i.e., each column of X). Fit a linear regression for each column using the remaining
    columns as variables.

    Parameters
    ----------
    X: 2D array of shape (n_samples, n_features)

    Returns
    -------
        1D array of shape (n_features,)
            VIF values for each feature.
    """
    X_new = add_constant(np.copy(X))
    return np.array([variance_inflation_factor(X_new, i) for i in range(X_new.shape[1])])


class MSKCCRadiomicsTransform(BaseEstimator, TransformerMixin):
    """
    Transformer to reproduce the experiments from Vanguri et al. (https://doi.org/10.1038/s43018-022-00416-8)

    Select radiomic features with three steps:
        * Remove features with outliers
        * Remove features which vary too much across perturbations of the initial segmentations (see [1])
        * Select features with elasticnet logistic regression, predicting the binary target of interest (i.e., features
        with non-zero coefficient)

    Parameters
    ----------
    lesion_type: list of strings or string in ['PC', 'LN', 'PL']
        Lesion type to consider. If list, features are selected for each type separately.

    robustness_cutoff: float in [0, 1]
        Minimum value to consider unrobust features. Features with a robustness ratio (i.e., average inter-lesion
        variance across the 10 perturbations / variance across all lesions) greater than this cutoff will be considered
        not robust and therefore not selected.

    outlier_cutoff: float >= 0
        Minimum absolute Z-score to define outliers. Samples with a Z-score greater that outlier_cutoff are considered
        outliers for the feature of interest. Only features with no outliers are ultimately selected.

    l1_C: float
        Inverse of regularization strenght for elasticnet penalty in logistic regression
        (see sklearn.linear_model.LogisticRegression).

    aggregation: string in ['mean', 'largest']
        Aggregation strategy when there are several target lesions of the same type (i.e., 'PC', 'LN', 'PL') for the
        same sample
        * 'mean' average feature values across the different target lesions
        * 'largest' take the largest lesion

    Attributes
    ----------
    selected_features_: dictionary
        Keys correspond to the type of target lesion (i.e., 'PC', 'LN', or 'PL') and values correspond to the list of
        indexes associated with the selected radiomic features for this specific type of lesion.

    References
    ----------
    1. Zwanenburg, A. et al. Assessing robustness of radiomic features by image perturbation. Sci. Rep. 9, 1–10 (2019).
    (https://doi.org/10.1038/s41598-018-36938-4)

    2. Vanguri, R.S. et al. Multimodal integration of radiology, pathology and genomics for prediction of response to
    PD-(L)1 blockade in patients with non-small cell lung cancer. Nat Cancer 3, 1151–1164 (2022).
    (https://doi.org/10.1038/s43018-022-00416-8)
    """

    # aggregation can be either by average or taking the first lesion sorted by index
    def __init__(self, lesion_type, robustness_cutoff, outlier_cutoff, l1_C, aggregation):
        self.lesion_type = lesion_type
        self.robustness_cutoff = robustness_cutoff
        self.outlier_cutoff = outlier_cutoff
        self.l1_C = l1_C
        assert aggregation in ['mean', 'largest'], "aggregation should either be 'mean' or 'largest'"
        self.aggregation = aggregation

    def fit(self, X, y):
        """
        Select features with elasticnet logistic regression algorithm for each lesion type separately (after removing
        features with outliers and un-robust features under perturbation of the original segmentation).

        Parameters
        ----------
        X: tuple of pandas dataframe and pandas Index
            * dataframe containing radiomics features extracted target lesions and their perturbed segmentations (10
            times)
            * Index containing the indexes of all the samples

        y: pandas serie or pandas dataframe of shape (n_samples,)
            Label for each sample

        Returns
        -------
        self: object
            Fitted estimator
        """
        assert isinstance(X, tuple), "X should be a tuple with pandas dataframe and indexes"
        data, indexes = X
        assert isinstance(data, pd.DataFrame), "X should be a pandas dataframe"
        assert isinstance(y, (pd.Series, pd.DataFrame)), "y should be a pandas object"

        # Select radiomic features for each lesion type of interest
        self.selected_features_ = {}
        if isinstance(self.lesion_type, list):
            for site in self.lesion_type:
                self.selected_features_[site] = select_radiomics_features_elastic(data[data['site'] == site],
                                                                                  y,
                                                                                  self.l1_C,
                                                                                  self.outlier_cutoff,
                                                                                  self.robustness_cutoff)
        elif isinstance(self.lesion_type, str):
            self.selected_features_[self.lesion_type] = select_radiomics_features_elastic(data[data['site'] ==
                                                                                               self.lesion_type],
                                                                                          y,
                                                                                          self.l1_C,
                                                                                          self.outlier_cutoff,
                                                                                          self.robustness_cutoff)
        return self

    def transform(self, X):
        """
        Select features and aggregate target lesions of the same type across samples (for each lesion type separately)

        Parameters
        ----------
        X: tuple of pandas dataframe and pandas Index
            * dataframe containing radiomics features extracted target lesions and their perturbed segmentations (10
            times)
            * Index containing the indexes of all the samples

        Returns
        -------
            List of pandas dataframes of shape (n_samples, n_selected_features_lesion_type_1),
            (n_samples, n_selected_features_lesion_type_2), ...
            Radiomic data with selected features and aggregated across the different target lesions for the different
            lesion type specified in self.lesion_type
        """
        assert isinstance(X, tuple), "X should be a tuple with pandas dataframe and indexes"
        data, indexes = X
        output = tuple()
        if self.aggregation == 'mean':
            data = data.reset_index()
            data = data[data["job_tag"] == 'filtered-radiomics'].drop(columns=['job_tag', 'lesion_index']) \
                .set_index('main_index')
            for lesion_type, selected_features in self.selected_features_.items():
                transformed_data = data[list(selected_features) + ['site']].groupby(level=0) \
                    .apply(_agg_average, site=lesion_type).drop('index', errors='ignore').reindex(indexes).values
                output = output + (transformed_data,)
        elif self.aggregation == 'largest':
            data = data.reset_index()
            data = data[data["job_tag"] == 'filtered-radiomics'].drop(columns='job_tag').set_index('main_index')
            for lesion_type, selected_features in self.selected_features_.items():
                transformed_data = data[list(selected_features) + ['site', 'lesion_index']].groupby(level=0) \
                    .apply(_agg_largest, site=lesion_type).droplevel(1).reindex(indexes).values
                output = output + (transformed_data,)
        return output


def _agg_average(g, site):
    """
    Compute the average value across different target lesions of the same type for each sample and each feature
    """
    g = g[g['site'] == site].drop(columns='site')
    return g.mean(axis=0)


def _agg_largest(g, site):
    """
    Return the values associated with the largest lesion when there are multiple lesions of the same type for one sample
    """
    g = g[g['site'] == site].drop(columns='site').sort_values('lesion_index', ascending=True) \
        .drop(columns='lesion_index').reset_index()
    return g.drop_duplicates(subset=['main_index']).drop(columns="main_index")
