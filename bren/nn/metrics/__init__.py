from bren.nn.metrics.Metric import Metric
from bren.nn.metrics.Accuracy import Accuracy
from bren.nn.metrics.MeanSquaredError import MeanSquaredError
from bren.nn.metrics.CategoricalCrossEntropy import CategoricalCrossEntropy
from bren.nn.utils import AliasDict



__all__ = [Metric, Accuracy, MeanSquaredError, CategoricalCrossEntropy]


METRICS = AliasDict()

for cls in __all__:
    METRICS[cls.__name__] = cls
    
METRICS.add(Accuracy.__name__, "accuracy")
METRICS.add(MeanSquaredError.__name__, "mse", "mean_squared_error", "MSE")
METRICS.add(CategoricalCrossEntropy.__name__, "categorical_cross_entropy")

def get_metric(name):
	return METRICS[name]