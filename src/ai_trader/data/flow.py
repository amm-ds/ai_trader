"""Flow-window value objects."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class FlowWindow:
    """Aggregated flow in a left-open, right-closed event window."""

    start: datetime
    end: datetime
    events: int
    buy_volume: float
    sell_volume: float

    @property
    def delta(self) -> float:
        """Return aggressor buy volume minus aggressor sell volume."""

        return self.buy_volume - self.sell_volume

    @property
    def volume(self) -> float:
        """Return total aggressor volume."""

        return self.buy_volume + self.sell_volume
