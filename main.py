from depression_classification.utils.settings import CFG_PATH

import train_and_tune
import evaluate
import predict

train_and_tune.main(CFG_PATH)
evaluate.main(CFG_PATH)
predict.main(CFG_PATH)
