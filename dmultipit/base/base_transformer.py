from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class UnimodalTransformer(TransformerMixin, BaseEstimator):

    def get_dimension(self):
        check_is_fitted(self, "is_fitted_")
        return self._get_transformed_dimension()

    def _get_transformed_dimension(self):
        raise NotImplementedError


class MultimodalTransformer(TransformerMixin, BaseEstimator):

    def get_dimension(self):
        check_is_fitted(self, "is_fitted_")
        return self._get_transformed_dimension()

    def get_multimodal_dimension(self):
        check_is_fitted(self)
        return self._get_transformed_multi_dimension()

    def transform_multimodal(self, X):
        raise NotImplementedError

    def _get_transformed_dimension(self):
        raise NotImplementedError

    def _get_transformed_multi_dimension(self):
        raise NotImplementedError
