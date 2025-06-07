from depression_classification.preprocessing.custom_transformers.flagging import ConditionalFlagger, InconsistencyFlagger


def get_conditional_flagger(flagging_map: dict) -> tuple[str, ConditionalFlagger]:
    return "conditional_flagger", ConditionalFlagger(flagging_map)


def get_inconsistency_flagger() -> tuple[str, InconsistencyFlagger]:
    return "inconsistency_flagger", InconsistencyFlagger()
