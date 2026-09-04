"""Historical feature materialization using the same pure feature function."""

from nifty_terminal.domain.candle import Timeframe
from nifty_terminal.features.engine import PriceFeatureEngine
from nifty_terminal.history.repository import HistoricalRepository


class HistoricalFeatureMaterializer:
    def __init__(
        self,
        *,
        repository: HistoricalRepository,
        engine: PriceFeatureEngine,
    ) -> None:
        self._repository = repository
        self._engine = engine

    def run(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        timeframe: Timeframe,
    ) -> int:
        candles = self._repository.load_latest_candles(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            timeframe=timeframe,
        )
        rows = self._engine.calculate(candles)
        return self._repository.save_feature_rows(dataset_id=dataset_id, rows=rows)
