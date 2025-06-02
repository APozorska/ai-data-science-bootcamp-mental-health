from sklearn.pipeline import Pipeline

from depression_classification.preprocessing.custom_transformers.imputation import RelationalImputer
from depression_classification.preprocessing.custom_transformers.category_transformers import FeatureCategoryMapper, RareCategoryCombiner
from depression_classification.preprocessing.custom_transformers.numerical_binning import UniformBinner
from depression_classification.preprocessing.custom_transformers.numerical_combination import FeatureCombiner
from depression_classification.preprocessing.custom_transformers.constants import mappings, normalize_category


def get_custom_cat_pipe():
    return Pipeline([
        ('cat_imputer_students', RelationalImputer(
            cols_to_impute=['profession'],
            condition_col='occupation_status',
            condition_value='Student',
            fill_value='Student'
        )),
        ('cat_rare_combiner', RareCategoryCombiner(
            columns=['profession', 'city'],
            min_count=10,
            other_label='other'
        )),
        ('cat_mapper', FeatureCategoryMapper(
            mappings=mappings,
            other_value='other',
            normalize_func=normalize_category
        )),
    ])


def get_custom_num_pipe():
    return Pipeline([
        ('num_imputer_students_const', RelationalImputer(
            cols_to_impute=['work_pressure', 'job_satisfaction'],
            condition_col='occupation_status',
            condition_value='Student',
            strategy='constant',
            fill_value=-1,
        )),
        ('num_imputer_workers_const', RelationalImputer(
            cols_to_impute=['academic_pressure', 'study_satisfaction', 'cgpa'],
            condition_col='occupation_status',
            condition_value='Working Professional',
            strategy='constant',
            fill_value=-1,
        )),
        ('num_imputer_students_median', RelationalImputer(
            cols_to_impute=['academic_pressure', 'study_satisfaction', 'cgpa'],
            condition_col='occupation_status',
            condition_value='Student',
            strategy='median'
        )),
        ('num_imputer_workers_median', RelationalImputer(
            cols_to_impute=['work_pressure', 'job_satisfaction'],
            condition_col='occupation_status',
            condition_value='Working Professional',
            strategy='median'
        )),
    ])


def get_feature_combiner_pipe():
    return Pipeline([
        ('num_feature_combiner_satisfaction', FeatureCombiner(
            col1='job_satisfaction', col2='study_satisfaction',
            strategy='max',
            new_col_name='job_study_satisfaction'
        )),
        ('num_feature_combiner_pressure', FeatureCombiner(
            col1='work_pressure', col2='academic_pressure',
            strategy='max',
            new_col_name='work_academic_stress'
        )),
    ])


def get_custom_preprocessor():
    transformer = Pipeline([
        ('custom_cat', get_custom_cat_pipe()),
        ('custom_num', get_custom_num_pipe()),
        ('feature_combiner', get_feature_combiner_pipe())
    ])
    return "custom_preprocessor", transformer
