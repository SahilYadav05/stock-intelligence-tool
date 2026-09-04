"""Canonical instruments and provider symbol mappings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CanonicalInstrument:
    instrument_id: str
    display_name: str
    venue: str
    asset_class: str
    timezone: str
    currency: str
    volume_supported: bool


@dataclass(frozen=True, slots=True)
class ProviderInstrumentMapping:
    provider: str
    provider_instrument_id: str
    canonical_instrument_id: str


class InstrumentRegistry:
    """Resolves provider identifiers without leaking them into canonical records."""

    def __init__(
        self,
        instruments: tuple[CanonicalInstrument, ...],
        mappings: tuple[ProviderInstrumentMapping, ...],
    ) -> None:
        self._instruments = {item.instrument_id: item for item in instruments}
        self._mappings = {
            (item.provider.casefold(), item.provider_instrument_id): item
            for item in mappings
        }

        for mapping in mappings:
            if mapping.canonical_instrument_id not in self._instruments:
                raise ValueError(
                    "Provider mapping references an unknown canonical instrument: "
                    f"{mapping.canonical_instrument_id}"
                )

    def resolve(self, provider: str, provider_instrument_id: str) -> CanonicalInstrument:
        mapping = self._mappings.get((provider.casefold(), provider_instrument_id))
        if mapping is None:
            raise KeyError(
                f"No canonical mapping for provider={provider!r}, "
                f"instrument={provider_instrument_id!r}"
            )
        return self._instruments[mapping.canonical_instrument_id]

    def get(self, instrument_id: str) -> CanonicalInstrument:
        try:
            return self._instruments[instrument_id]
        except KeyError as error:
            raise KeyError(f"Unknown canonical instrument: {instrument_id}") from error


def build_mvp_instrument_registry() -> InstrumentRegistry:
    """Build the MVP registry without leaking provider tokens downstream."""

    nifty_50 = CanonicalInstrument(
        instrument_id="NIFTY50_SPOT",
        display_name="NIFTY 50",
        venue="NSE",
        asset_class="CASH_INDEX",
        timezone="Asia/Kolkata",
        currency="INR",
        volume_supported=False,
    )

    return InstrumentRegistry(
        instruments=(nifty_50,),
        mappings=(
            ProviderInstrumentMapping(
                provider="replay",
                provider_instrument_id="NIFTY50_TEST",
                canonical_instrument_id=nifty_50.instrument_id,
            ),
            ProviderInstrumentMapping(
                provider="angelone",
                provider_instrument_id="NSE:99926000",
                canonical_instrument_id=nifty_50.instrument_id,
            ),
            ProviderInstrumentMapping(
                provider="angelone",
                provider_instrument_id="NSE:26000",
                canonical_instrument_id=nifty_50.instrument_id,
            ),
        ),
    )
