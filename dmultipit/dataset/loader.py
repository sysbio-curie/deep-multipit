import pandas as pd
import numpy as np
from sksurv.util import Surv


def load_TIPIT_multimoda(
    clinical_file,
    radiomics_file,
    pathomics_file,
    rna_file,
    clinical_features,
    radiomic_features,
    pathomics_features,
    rna_features,
    order,
    outcome="OS",
    keep_unlabelled=False,
    return_survival=None,
):
    """
    Loader for raw TIPIT data.

    Parameters
    ----------
    clinical_file: path to clinical data

    radiomics_file: path to radiomic data or None
        If None, no radiomic data are loaded

    pathomics_file: path to pathomic data or None
        If None, no pathomic data are loaded

    rna_file: path to transcriptomic data or None
        If None, no RNA data are loaded

    clinical_features: list, pandas Index, or None
        Names of clinical features to consider (the other will be filtered out). If None all the clinical features in
        the original clinical dataset are kept.

    radiomic_features: list, pandas Index, or None
        Names of radiomic features to consider (the other will be filtered out). If None all the radiomic features in
        the original radiomic dataset are kept.

    pathomics_features: list, pandas Index, or None
        Names of pathomic features to consider (the other will be filtered out). If None all the pathomic features in
        the original pathomic dataset are kept.

    rna_features: list, pandas Index, or None
        Names of RNA features to consider (the other will be filtered out). If None all the RNA features in
        the original RNA dataset are kept.

    order: list of strings
         Order in which to store the loaded data sets (e.g., ['clinicals', 'radiomics', 'pathomics', 'RNA']). The
         modalities in that list should correspond to the lodaed modalities (e.g., ['clinicals', 'RNA'] if
         radiomics_file and pathomics_file are set to None)

    outcome: string in ['OS', 'PFS', 'RECIST']
        Specify which binary target to load.
        * If 'OS', binary label will be whether the patient died before 1 year. Patients censored before 1 year will not
        be considered.
        * If 'PFS', binary label will be whether the patient progressed before 6 months. Patients censored before
        6 months will not be considered.
        * If 'RECIST', binary label will be whether patient response belongs to the 'partial' or 'complete' categories.

        The default is 'OS'.

    keep_unlabelled: boolean
        If True, patients with no label (e.g. missing value or censored before threshold) will be kept in the analysis
        and their label will be set to NaN value. Otherwise, they are discarded. The default is False.

    return_survival: string in ['OS', 'PFS'] or None
        Load survival data (i.e., time to event + censorship status) in addition to the binary target. If None no
        additional survival data is loaded. The default is None.

    Returns
    -------
        Tuple of pandas dataframes and sksruv.utils.Surv
            Loaded modalities (ordered according to *order* parameter), binary target, and survival data (if
            *return_survival* is not None).
    """
    # 1. Load raw data and concatenate them
    assert clinical_file is not None, "clinical data should always be provided"
    df_clinicals = pd.read_csv(clinical_file, index_col=0, sep=";")
    df_radiomics = (
        pd.read_csv(radiomics_file, index_col=0, sep=";")
        if radiomics_file is not None
        else None
    )
    df_pathomics = (
        pd.read_csv(pathomics_file, index_col=0, sep=";")
        if pathomics_file is not None
        else None
    )
    df_RNA = (
        pd.read_csv(rna_file, index_col=0, sep=";") if rna_file is not None else None
    )

    # Encode 'Biopsy site' feature for RNA data
    d = {}
    for site in df_RNA["Biopsy site"].unique():
        if pd.isnull(site) | (site == "Non disponible"):
            d[site] = np.nan
        elif site in ["PRIMITIF", "META_PULM", "META_PULM_HL", "META_PULM_CL"]:
            d[site] = 0
        elif site in ["META_PLEVRE", "META_PLEVRE_HL", "META_PLEVRE_CL"]:
            d[site] = 1
        elif site.split("_")[0] == "ADP":
            d[site] = 2
        elif site == "META_OS":
            d[site] = 3
        elif site == "META_FOIE":
            d[site] = 4
        elif site == "META_SURRENALE":
            d[site] = 5
        elif site == "META_BRAIN":
            d[site] = 6
        else:
            d[site] = 7
    df_RNA.replace({"Biopsy site": d}, inplace=True)

    list_data = [df for df in [df_clinicals, df_pathomics, df_RNA, df_radiomics] if df is not None]
    df_total = pd.concat(list_data, axis=1, join="outer") if len(list_data) > 1 else list_data[0].copy()

    # 2. Collect outcome/target (either OS, PFS or Best Response)
    if outcome == "OS":
        bool_mask = (df_total["OS"].isnull()) | ((df_total["OS"] <= 365) & (df_total["Statut Vital"] == "Vivant"))
        if keep_unlabelled:
            target = (1 * (df_total["OS"] <= 365)).where(~bool_mask, other=np.nan)
        else:
            df_total = df_total[~bool_mask]
            target = 1 * (df_total["OS"] <= 365)

    elif outcome == "PFS":
        bool_mask = (df_total["PFS"].isnull()) | ((df_total["PFS"] <= 180) & (df_total["Progression"] == "No"))
        if keep_unlabelled:
            target = (1 * (df_total["PFS"] <= 180)).where(~bool_mask, other=np.nan)
        else:
            df_total = df_total[~bool_mask]
            target = 1 * (df_total["PFS"] <= 180)

    elif outcome == "RECIST":
        bool_mask = df_total["Best response"].isnull()
        if keep_unlabelled:
            # target = 1*((df_total['Best response'] == 'Stable') | (df_total['Best response'] == 'Progression'))
            target = 1 * ((df_total["Best response"] == "Partielle") | (df_total["Best response"] == "Complete"))
            target = target.where(~bool_mask, other=np.nan)
        else:
            df_total = df_total[~bool_mask]
            # target = 1*((df_total['Best response'] == 'Stable') | (df_total['Best response'] == 'Progression'))
            target = 1 * ((df_total["Best response"] == "Partielle") | (df_total["Best response"] == "Complete"))

    else:
        raise ValueError("outcome can only be 'OS','PFS' or 'RECIST'")

    # 3. Select specific features for each modality
    datasets = {key: None for key in ["clinicals", "radiomics", "pathomics", "RNA"]}

    if return_survival == "OS":
        target_survival = Surv().from_arrays(event=(1 * (df_total["Statut Vital"] == "Decede")).values,
                                             time=df_total["OS"].values,
                                             )
    elif return_survival == "PFS":
        target_survival = Surv().from_arrays(event=(1 * (df_total["Progression"] == "Yes")).values,
                                             time=df_total["PFS"].values,
                                             )

    if clinical_features is not None:
        datasets["clinicals"] = df_total[clinical_features]
    else:
        datasets["clinicals"] = df_total[df_clinicals.columns].drop(
            columns=["OS", "PFS", "Statut Vital", "Progression", "Best response"],
            errors="ignore",
        )

    if df_radiomics is not None:
        datasets["radiomics"] = df_total[radiomic_features] if radiomic_features is not None \
            else df_total[df_radiomics.columns]
    if df_pathomics is not None:
        datasets["pathomics"] = df_total[pathomics_features] if pathomics_features is not None \
            else df_total[df_pathomics.columns]

    if df_RNA is not None:
        datasets["RNA"] = df_total[rna_features] if rna_features is not None else df_total[df_RNA.columns]

    # 4. Return each dataset and the target in the right order
    output = tuple()
    for modality in order:
        assert datasets[modality] is not None, (
            "order specifies a modality but the input file for loading the raw "
            "data is not given "
        )
        output = output + (datasets[modality],)

    if return_survival is not None:
        return output + (target, target_survival)
    else:
        return output + (target,)


def load_MSKCC_multimoda(
    clinical_file,
    radiomics_file,
    pathomics_file,
    omics_file,
    pdl1_file,
    clinical_features,
    radiomic_features,
    pathomics_features,
    omics_features,
    order,
    outcome,
    keep_unlabelled,
):
    """
    Loader for MSKCC data to reproduce experiments from in Vanguri et al. (https://doi.org/10.1038/s43018-022-00416-8)

    Parameters
    ----------
    clinical_file: path to clinical data

    radiomics_file: path to radiomic data or None
        If None, no radiomic data are loaded

    pathomics_file: path to pathomic data or None
        If None, no pathomic data are loaded

    omics_file: path to omic data or None
        If None, no omic data are loaded

    pdl1_file: path to pdl1 data or None
        If None, no pdl1 data are loaded

    clinical_features: list, pandas Index, or None
        Names of clinical features to consider (the other will be filtered out). If None all the clinical features in
        the original clinical dataset are kept.

    radiomic_features: list, pandas Index, or None
        Names of radiomic features to consider (the other will be filtered out). If None all the raiomic features in
        the original radiomic dataset are kept.

    pathomics_features: list, pandas Index, or None
        Names of pathomic features to consider (the other will be filtered out). If None all the pathomic features in
        the original pathomic dataset are kept.

    omics_features: list, pandas Index, or None
        Names of omics features to consider (the other will be filtered out). If None all the omic features in
        the original RNA dataset are kept.

    order: list of strings
         Order in which to store the loaded data sets (e.g., ['clinicals', 'radiomics', 'pathomics', 'RNA']). The
         modalities in that list should correspond to the lodaed modalities (e.g., ['clinicals', 'RNA'] if
         radiomics_file and pathomics_file are set to None)

    outcome: string in ['OS', 'PFS', 'RECIST']
        Specify which binary target to load.
        * If 'OS', binary label will be whether the patient died before 1 year. Patients censored before 1 year will not
        be considered.
        * If 'PFS', binary label will be whether the patient progressed before 6 months. Patients censored before
        6 months will not be considered.
        * If 'RECIST', binary label will be whether patient response belongs to the 'partial' or 'complete' categories.

        The default is 'OS'.

    keep_unlabelled: boolean
        If True, patients with no label (e.g. missing value or censored before threshold) will be kept in the analysis
        and their label will be set to NaN value. Otherwise, they are discarded. The default is False.

    Returns
    -------
        Tuple of pandas dataframes
            Loaded modalities (ordered according to *order* parameter) and binary target

    References
    ----------
    1. Vanguri, R.S. et al. Multimodal integration of radiology, pathology and genomics for prediction of response to
    PD-(L)1 blockade in patients with non-small cell lung cancer. Nat Cancer 3, 1151–1164 (2022).
    (https://doi.org/10.1038/s43018-022-00416-8)

    Notes
    -----
    Data are available at: https://www.synapse.org/#!Synapse:syn26642505
    """

    assert clinical_file is not None, "clinical data should always be provided"
    df_clinicals = pd.read_csv(clinical_file, index_col=0, sep=";")
    df_radiomics = pd.read_csv(radiomics_file, index_col=0) if radiomics_file is not None else None
    df_pathomics = pd.read_csv(pathomics_file, index_col=0) if pathomics_file is not None else None
    df_omics = pd.read_csv(omics_file, index_col=0) if omics_file is not None else None
    df_pdl1 = pd.read_csv(pdl1_file, index_col=0) if pdl1_file is not None else None

    # concatenate datasets but without radiomic file
    list_data = [df for df in [df_clinicals, df_pathomics, df_omics, df_pdl1] if df is not None]
    df_total = (
        pd.concat(list_data, axis=1, join="outer")
        if len(list_data) > 1
        else list_data[0].copy()
    )

    # 2. Collect outcome/target (either OS, PFS or Best Response)
    if outcome == "RECIST":
        target = df_total["label"]
    elif outcome == "OS":
        df_total[df_total["pfs_censor"] == 0]["os_int"] = df_total[df_total["pfs_censor"] == 0]["pfs"]
        bool_mask = (df_total["os_int"].isnull()) | ((df_total["os_int"] <= 12) & (df_total["pfs_censor"] == 0))
        if keep_unlabelled:
            target = (1 * (df_total["os_int"] <= 12)).where(~bool_mask, other=np.nan)
        else:
            df_total = df_total[~bool_mask]
            target = 1 * (df_total["os_int"] <= 12)
    elif outcome == "PFS":
        bool_mask = (df_total["pfs"] <= 6) & (df_total["pfs_censor"] == 0)
        if keep_unlabelled:
            target = (1 * (df_total["pfs"] <= 6)).where(~bool_mask, other=np.nan)
        else:
            df_total = df_total[~bool_mask]
            target = 1 * (df_total["pfs"] <= 6)
    else:
        raise ValueError("outcome can only be 'OS','PFS' or 'RECIST'")

    # 3. Select specific features for each modality
    datasets = {key: None for key in ["clinicals", "radiomics", "pathomics", "omics", "pdl1"]}

    if clinical_features is not None:
        datasets["clinicals"] = df_total[clinical_features]
    else:
        datasets["clinicals"] = df_total[df_clinicals.columns].drop(
            columns=["os_int", "pfs", "label", "pfs_censor"], errors="ignore"
        )

    # radiomics data set consists in a tuple of radiomics data + index from total data set
    if df_radiomics is not None:
        datasets["radiomics"] = (df_radiomics[radiomic_features], df_total.index) if radiomic_features is not None \
                                else (df_radiomics, df_total.index)
    if df_pathomics is not None:
        datasets["pathomics"] = df_total[pathomics_features] if pathomics_features is not None \
                                else df_total[df_pathomics.columns]
    if df_omics is not None:
        datasets["omics"] = df_total[omics_features] if omics_features is not None else df_total[df_omics.columns]

    if df_pdl1 is not None:
        datasets["pdl1"] = df_total[df_pdl1.columns]

    # 4. Return each dataset and the target in the right order
    output = tuple()
    rad = False
    for modality in order:
        if modality.split("_")[0] == "radiomics":
            if not rad:
                assert datasets["radiomics"] is not None, (
                    "order specifies a modality but the input file for loading "
                    "the raw data is not given "
                )
                output = output + (datasets["radiomics"],)
                rad = True
        else:
            assert datasets[modality] is not None, (
                "order specifies a modality but the input file for loading the raw "
                "data is not given "
            )
            output = output + (datasets[modality],)

    return output + (target,)
