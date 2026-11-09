from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)


class ArimaForecaster:
    """ARIMA refitted on the last `window` closes at each forecast (rolling forecast)."""

    def __init__(self, order: tuple[int, int, int] = (2, 1, 0), window: int = 50) -> None:
        self.order = order
        self.window = window

    def forecast_change(self, closes: npt.NDArray[np.float64]) -> float:
        """Predicted change of the next close (0.0 when there is not enough data or the fit fails)."""
        series = np.asarray(closes, dtype=float)
        if len(series) < self.window:
            return 0.0
        try:
            model_fit = ARIMA(series[-self.window :], order=self.order).fit()
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.warning("ARIMA%s fit failed: %s", self.order, e)
            return 0.0
        return float(model_fit.forecast()[0]) - float(series[-1])
