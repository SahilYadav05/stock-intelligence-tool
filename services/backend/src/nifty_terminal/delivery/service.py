"""Atomic publication service for HTTP and WebSocket consumers."""

from nifty_terminal.delivery.hub import MarketStateHub, SequencedMarketState
from nifty_terminal.delivery.models import MarketStateView
from nifty_terminal.delivery.read_model import InMemoryMarketStateReadModel


class MarketStateDeliveryService:
    def __init__(
        self,
        *,
        read_model: InMemoryMarketStateReadModel | None = None,
        hub: MarketStateHub | None = None,
    ) -> None:
        self.read_model = read_model or InMemoryMarketStateReadModel()
        self.hub = hub or MarketStateHub()

    async def publish(self, view: MarketStateView) -> SequencedMarketState:
        self.read_model.put(view)
        return await self.hub.publish(view)
