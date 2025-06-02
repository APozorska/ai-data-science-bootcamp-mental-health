import pandas as pd


# Normalize function
def normalize_category(value):
    if pd.isna(value):
        return value
    if isinstance(value, str):
        return value.replace('.', '').replace(' ', '').replace('_', '').lower()
    return str(value).lower()


# Categories and mappings
dietary_mapping = {
    'healthy': ['healthy'],
    'unhealthy': ['unhealthy'],
    'moderate': ['moderate']
}

sleep_mapping = {
    '<5': ['lessthan5hours', '1-2hours', '1-3hours', '1-6hours', '2-3hours', '3-4hours', '3-6hours', '4-5hours'],
    '5-7': ['5-6hours', '4-6hours', '6-7hours'],
    '7-8': ['7-8hours', '6-8hours', '8hours'],
    '>8': ['morethan8hours', '9-11hours', '10-11hours', '8-9hours']
}

degree_mapping = {
    "secondary_education": ["Class12", "Class11"],
    "undergraduate": ["bed", "barch", "bcom", "bpharm", "bca", "bba", "bsc", "btech", "llb", "bhm", "ba", "be", "barch"],
    "postgraduate": ["med", "mca", "msc", "llm", "mpharm", "mtech", "mba", "me", "md", "mhm", "mcom", "mbbs", "ma", "march"],
    "doctorate": ["phd"]
}

mappings = {
    'dietary_habits': dietary_mapping,
    'sleep_duration': sleep_mapping,
    'degree': degree_mapping
}


flagging_map = {
    'work_pressure': {
        'not_applicable': {'occupation_status': 'Student'},
        'imputed': {'occupation_status': 'Working Professional'}
    },
    'job_satisfaction': {
        'not_applicable': {'occupation_status': 'Student'},
        'imputed': {'occupation_status': 'Working Professional'}
    },
    'academic_pressure': {
        'not_applicable': {'occupation_status': 'Working Professional'},
        'imputed': {'occupation_status': 'Student'}
    },
    'study_satisfaction': {
        'not_applicable': {'occupation_status': 'Working Professional'},
        'imputed': {'occupation_status': 'Student'}
    },
    'cgpa': {
        'not_applicable': {'occupation_status': 'Working Professional'},
        'imputed': {'occupation_status': 'Student'}
    }
}
