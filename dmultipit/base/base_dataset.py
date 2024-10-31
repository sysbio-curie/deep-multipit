import inspect

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import NotFittedError
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    """
    Base class for multimodal datasets

    Parameters
    ---------
    list_raw_data: list of numpy arrays or pandas dataframes of shape (n_samples, n_features_1),
        (n_samples, n_features_2)... The rows of the different datasets should be ordered in the same way (i.e., same
        order of samples each time).

    labels: 1D numpy array or pandas serie (n_samples,)
        Label for each sample. It should be ordered in the same way as the raw data sets provied in *list_raw_data*.

    list_unimodal_processings: list of dictionaries, sklearn.base.TransformerMixin and None
        List of processing operations to apply to each modality separately.
        * If None no operation is performed for the corresponding modality
        * If dictionary it should define a processing strategy to be fitted on the data
         (e.g., {'scaling': {'with_std': True}, 'PCA': {'n_components': 0.9}} could define a standard scaling
         operations followed by a PCA (to be defined in the fit_process method !))
        * If sklearn.base.TransformerMixin it should correspond to a fitted transformer !

    multimodal_processing: dictionary, sklearn.base.TransformerMixin and None
        Processing operations to apply to the multimodal data set.
        * If None no operation is performed
        * If dictionary it should define a processing strategy to be fitted on the data (to be defined in the
         fit_multimodal_process method !)
        * If sklearn.base.TransformerMixin it should correspond to a fitted transformer !

    transform: callable or None
            Transformation to apply to each sample in the batch (e.g., data augmentation). If None, no transformation is
            performed. The default is None.
    """

    def __init__(
            self,
            list_raw_data,
            labels,
            list_unimodal_processings,
            multimodal_processing,
            transform=None,
    ):
        self.list_raw_data = tuple()
        self.sample_names = None

        for data in list_raw_data:
            assert (len(self.list_raw_data) == 0) or data.shape[0] == self.list_raw_data[0].shape[0], ("all data sets "
                                                                                                       "should have "
                                                                                                       "the same "
                                                                                                       "number of "
                                                                                                       "samples ")
            if isinstance(data, pd.DataFrame):
                self.list_raw_data = self.list_raw_data + (data.values,)
                assert (self.sample_names is None) or (len(set(data.index) ^ set(self.sample_names)) == 0), ("all "
                                                                                                        "dataframes"
                                                                                                        " should have"
                                                                                                        " the same"
                                                                                                        " indexes ")
                self.sample_names = data.index
            elif isinstance(data, np.ndarray):
                self.list_raw_data = self.list_raw_data + (data.copy(),)
            else:
                raise ValueError("Data sets should be either pd.DataFrame or np.ndarray")

        if self.sample_names is None:
            self.sample_names = ["sample " + str(i) for i in range(self.list_raw_data[0].shape[0])]

        self.labels = labels.values if isinstance(labels, (pd.Series, pd.DataFrame)) else labels

        # Remove unlabelled data points
        nan_labels = np.isnan(self.labels)
        if np.sum(nan_labels) > 0:
            self.labels = self.labels[~nan_labels]
            filtered_data = tuple()
            for data in self.list_raw_data:
                filtered_data = filtered_data + (data[~nan_labels, :],)
            self.list_raw_data = filtered_data

        # compute weights for batch sampling (only for binary classification and labelled data)
        if type_of_target(self.labels[~np.isnan(self.labels)]) == "binary":
            counts_0, counts_1 = np.nansum(self.labels == 0), np.nansum(self.labels == 1)
            self.sample_weights = []
            for label in self.labels:
                if not np.isnan(label):
                    self.sample_weights.append(1/counts_0) if label == 0 else self.sample_weights.append(1/counts_1)
                else:
                    self.sample_weights.append(np.nan)
        else:
            self.sample_weights = None

        # Create mask indicating for each sample whether a modality is missing (i.e., only NaN values)
        self.masks = self._create_masks(self.list_raw_data)

        # Apply processing transformations to each individual modality (either pre-fitted operations or fit it to the
        # whole data).
        self.list_unimodal_processings = self.init_unimodal_processing(list_unimodal_processings)
        self.list_transformed_data = []
        for i, (processor, raw_data) in enumerate(zip(self.list_unimodal_processings, self.list_raw_data)):
            mask_moda = self.masks[:, i] == 1
            if processor is not None:
                temp = processor.transform(raw_data[mask_moda])
                data = np.full((raw_data.shape[0], temp.shape[1]), np.nan)
                data[mask_moda] = temp
            else:
                data = np.copy(raw_data)
            self.list_transformed_data.append(data)

        # Apply processing transformations to all the modalities
        self.multimodal_processing = self.init_multimodal_processing(multimodal_processing)
        if self.multimodal_processing is not None:
            self.list_transformed_data = (self.multimodal_processing
                                              .transform_multimodal(np.hstack(self.list_transformed_data))
                                          )

        # Replace NaN values from missing modalities by 0 values
        final_list = []
        for i, data in enumerate(self.list_transformed_data):
            final_list.append(np.where((self.masks[:, i] == 1).reshape(-1, 1), data, 0))
        self.list_transformed_data = final_list

        # Define transformation to apply to each sample
        self.transform = transform

    @staticmethod
    def _create_masks(list_data):
        """Define boolean mask indicating whether a modality is missing (i.e., only NaN values) for each sample.

        Parameters
        ----------
        list_data: list of numpy arrays of shape (n_samples, n_features_1), (n_samples, n_features_2)...

        Returns
        -------
            boolean numpy array of shape (n_samples, n_modalities)
        """
        modality_masks = []
        for data in list_data:
            modality_masks.append((np.sum(~np.isnan(data), axis=1) > 0).reshape(-1, 1))
        return np.hstack(modality_masks)

    def init_unimodal_processing(self, list_unimodal_processings):
        """Initialize unimodal processing operations. Check that transformers are already fitted or define and
        fit a new transformer from a dictionary (as implemented in the fit_process method !)

        Parameters
        ----------
        list_unimodal_processings: list of dictionaries, sklearn.base.TransformerMixin and None
            List of processing operations to apply to each modality separately.
            * If None no operation is performed for the corresponding modality
            * If dictionary it should define a processing strategy to be fitted on the data
             (e.g., {'scaling': {'with_std': True}, 'PCA': {'n_components': 0.9}} could define a standard scaling
             operations followed by a PCA (to be defined in the fit_process method !))
            * If sklearn.base.TransformerMixin it should correspond to a fitted transformer !

        Returns
        -------
            list of fitted sklearn.base.TransformerMixin and None
        """
        output = []
        for i, processing in enumerate(list_unimodal_processings):
            # 1. No pre-processing
            if processing is None:
                output.append(None)
            # 2. Fitted transformer
            elif _check_transform(processing):
                try:
                    _check_is_fitted(processing)
                except NotFittedError:
                    print("processing is not fitted")
                    raise
                else:
                    output.append(processing)
            # 3. New pre-processing to be fitted on raw_data (remove samples with missing modalities)
            elif isinstance(processing, dict):
                mask = np.sum(~np.isnan(self.list_raw_data[i]), axis=1) > 0
                processing = self.fit_process(
                    X=self.list_raw_data[i][mask],
                    y=self.labels[mask],
                    params=processing,
                )
                if _check_transform(processing):
                    output.append(processing)
                else:
                    print("when processing is a dictionary it should refer to a transformer class which inherits from"
                          " sklearn.base.TransformerMixin.")
                    raise
            # 4. Raise an error if processing is none of the above
            else:
                raise ValueError(
                    "processing argument can only be None, a dictionnary specifying the processing strategy"
                    " or a fitted transformer which inherits from sklearn.base.TransformerMixin."
                )
        return output

    def init_multimodal_processing(self, multimodal_processing):
        """Initialize multimodal processing operations. Check that transformers are already fitted or define and
        fit a new transformer from a dictionary (as implemented in the fit_process method !)

        Parameters
        ----------
        multimodal_processing: dictionary, sklearn.base.TransformerMixin and None
            Processing operations to apply to the multimodal data set.
            * If None no operation is performed
            * If dictionary it should define a processing strategy to be fitted on the data (to be defined in the
             fit_multimodal_process method !)
            * If sklearn.base.TransformerMixin it should correspond to a fitted transformer !

        Returns
        -------
            fitted sklearn.base.TransformerMixin or None
        """
        # 1. No pre-processing
        if multimodal_processing is None:
            output = None
        # 2. Fitted transformer
        elif _check_transform(multimodal_processing):
            try:
                _check_is_fitted(multimodal_processing)
            except NotFittedError:
                print("processing is not fitted")
                raise
            else:
                if hasattr(multimodal_processing, "transform_multimodal"):
                    output = multimodal_processing
                else:
                    print("Processing must have a transform_multimodal method")
                    raise
        # 3. New pre-processing to be fitted on raw_data (remove samples with missing modalities)
        elif isinstance(multimodal_processing, dict):
            full_data = np.hstack(self.list_transformed_data)
            mask = np.sum(~np.isnan(full_data), axis=1) > 0
            modalities = []
            for m, data in enumerate(self.list_transformed_data):
                modalities += [m] * data.shape[1]
            modalities = np.array(modalities)
            multimodal_processing = self.fit_multimodal_process(
                X=full_data,
                y=self.labels[mask],
                params=multimodal_processing,
                modalities=modalities,
            )
            if hasattr(multimodal_processing, "transform_multimodal"):
                output = multimodal_processing
            else:
                print(
                    "when processing is a dictionary it should refer to a class with a 'transform_multimodal' method"
                )
                raise
        # 4. Raise an error if processing is none of the above
        else:
            raise ValueError(
                "processing argument can only be None, a dictionnary specifying the processing strategy"
                " or a fitted transformer which inherits from sklearn.base.TransformerMixin."
            )
        return output

    def fit_multimodal_process(self, X, y, params, modalities):
        raise NotImplementedError

    def fit_process(self, X, y, params):
        raise NotImplementedError

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mask = self.masks[idx, :].reshape(1, -1) if isinstance(idx, int) else self.masks[idx, :]
        item = ()
        for i, transformed_data in enumerate(self.list_transformed_data):
            data = transformed_data[idx, :].reshape(1, -1) if isinstance(idx, int) else transformed_data[idx, :]
            item = item + (torch.squeeze(torch.from_numpy(data).to(dtype=torch.float), 0),)
        item = item + (torch.tensor(mask).squeeze(dim=0), torch.tensor(self.labels[idx]),)

        if self.transform is not None:
            item = self.transform(item)
        return item


def _check_is_fitted(test_object):
    """Check whether the object is fitted (or whether each element is fitted if the object is a pipeline)."""
    classes = inspect.getmro(test_object.__class__)
    if classes[0].__name__ == "Pipeline":
        check_is_fitted(test_object[-1])
    else:
        check_is_fitted(test_object)
    return


def _check_transform(test_object):
    """Check whether the object belongs to the sklearn.base.TransformerMixin class (or a pipeline composed of
    transformers).
    """
    check = False
    classes = inspect.getmro(test_object.__class__)
    if classes[0].__name__ == "Pipeline":
        check = all([_check_transform(estim) for estim in test_object])
    else:
        for parentClass in classes[1:]:
            if parentClass.__name__ == "TransformerMixin":
                check = True
                break
    return check
