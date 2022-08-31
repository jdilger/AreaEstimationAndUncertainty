from collections import namedtuple
from typing import Tuple, Union
import numpy as np
import pandas as pd


AreaEstimate = namedtuple("AreaEstimate", "category_area, category_area_95_ci")
Uncertainty = namedtuple("Uncertainty", "variance, stdError, ci_95, moe")


def load_csv(file: str, index_col: Union[int, None] = None) -> pd.DataFrame:
    df = pd.read_csv(file, index_col=index_col)
    return df


def prep_analysis(
    confusion_matrix: pd.DataFrame, areas: pd.DataFrame, area_field: str
) -> Tuple[pd.DataFrame, int, int]:
    shape = confusion_matrix.shape
    assert (
        shape[0] == shape[1]
    ), f"confusion maxtrix should have equal columns and rows: {shape}"

    sum_ref_samples = np.sum(confusion_matrix, 0).array
    sum_map_samples = np.sum(confusion_matrix, 1).array

    areas["sum_strata_samples"] = sum_map_samples.reshape(shape[0], 1)

    total_area = np.sum(areas[area_field])
    total_samples = np.sum(sum_ref_samples)
    areas["weight"] = areas[area_field] / total_area
    return areas, total_area, total_samples


def area_proportions(
    prep_areas: pd.DataFrame, confusion_matrix: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    proportional = confusion_matrix.apply(
        lambda i: i
        * prep_areas["weight"].array
        / prep_areas["sum_strata_samples"].array
    )
    area_proportion = np.sum(proportional, axis=0)
    return proportional, area_proportion


def uncertainty_estimation(
    proportional: pd.DataFrame,
    area_proportion: pd.DataFrame,
    prep_areas: pd.DataFrame,
) -> Uncertainty:
    v = np.sum(
        proportional.apply(
            lambda i: (prep_areas["weight"].array * i - i**2)
            / (prep_areas["sum_strata_samples"].array - 1)
        )
    )
    se = np.sqrt(v)
    ci_95 = se * 1.96
    moe = ci_95 / area_proportion
    return Uncertainty(v, se, ci_95, moe)


def area_estimation(
    total_area: int,
    area_proportion: pd.DataFrame,
    confidence_interval: pd.DataFrame,
    pixel_resolution: Union[int, None] = None,
) -> AreaEstimate:
    if pixel_resolution:
        total_m2 = pixel_resolution * pixel_resolution * total_area
    else:
        total_m2 = total_area

    total_hectacres = total_m2 * 0.0001
    category_area = total_hectacres * area_proportion
    category_area_95_ci = total_hectacres * confidence_interval
    return AreaEstimate(category_area, category_area_95_ci)


def main():
    input_areas_path = "countsReadable_S2_2021_LandCover_Zambezi.csv"
    input_confusion_matrix_path = "confusion.csv"
    area_field = "count"

    areas = load_csv(input_areas_path)
    conf = load_csv(input_confusion_matrix_path, index_col=0)

    prep_areas, total_area, total_samples = prep_analysis(
        conf, areas, area_field
    )
    proportional, area_proportion = area_proportions(prep_areas, conf)
    uncertainty = uncertainty_estimation(
        proportional, area_proportion, prep_areas
    )

    results = area_estimation(total_area, area_proportion, uncertainty.ci_95)

    print(results)


if __name__ == "__main__":
    main()
