"use client";

import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  LineSeries,
  LineStyle,
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import { useEffect, useRef, type MutableRefObject } from "react";

import type { AnalysisViewContract } from "@/src/lib/analysis-contracts";
import type { CandleContract } from "@/src/lib/market-contracts";
import type { PriceActionAnalysisContract } from "@/src/lib/price-action-contracts";

export interface IndicatorVisibility {
  ema20: boolean;
  ema50: boolean;
  levels: boolean;
  signals: boolean;
}

export function TerminalChart({
  finalized,
  developing,
  analysis,
  priceAction,
  indicators,
}: {
  finalized: CandleContract[];
  developing: CandleContract | null;
  analysis: AnalysisViewContract | null;
  priceAction: PriceActionAnalysisContract | null;
  indicators: IndicatorVisibility;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const ema20Ref = useRef<ISeriesApi<"Line"> | null>(null);
  const ema50Ref = useRef<ISeriesApi<"Line"> | null>(null);
  const markerPluginRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const priceLinesRef = useRef<IPriceLine[]>([]);
  const datasetRef = useRef<{ timeframe: string; count: number } | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const chart = createChart(container, {
      width: container.clientWidth,
      height: container.clientHeight,
      layout: {
        background: { type: ColorType.Solid, color: "#080d13" },
        textColor: "#667789",
        attributionLogo: false,
        fontFamily: '"IBM Plex Mono", Consolas, monospace',
        fontSize: 10,
      },
      localization: {
        locale: "en-IN",
        timeFormatter: (time) => formatIstCrosshairTime(time),
      },
      grid: {
        vertLines: { color: "rgba(29, 44, 58, 0.42)" },
        horzLines: { color: "rgba(29, 44, 58, 0.42)" },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: {
        borderColor: "#1b2a38",
        scaleMargins: { top: 0.08, bottom: 0.08 },
      },
      timeScale: {
        borderColor: "#1b2a38",
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 7,
        barSpacing: 9,
        tickMarkFormatter: (time, tickMarkType) => formatIstTick(time, tickMarkType),
      },
      handleScroll: true,
      handleScale: true,
    });
    const candles = chart.addSeries(CandlestickSeries, {
      upColor: "#24c995",
      downColor: "#ec6571",
      borderUpColor: "#24c995",
      borderDownColor: "#ec6571",
      wickUpColor: "#24c995",
      wickDownColor: "#ec6571",
      priceLineVisible: true,
      lastValueVisible: true,
    });
    const ema20 = chart.addSeries(LineSeries, {
      color: "#42d4ff",
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    const ema50 = chart.addSeries(LineSeries, {
      color: "#b899ff",
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    const resizeObserver = new ResizeObserver(() => {
      chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    });
    resizeObserver.observe(container);
    chartRef.current = chart;
    candleSeriesRef.current = candles;
    ema20Ref.current = ema20;
    ema50Ref.current = ema50;
    markerPluginRef.current = createSeriesMarkers(candles, []);
    return () => {
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      ema20Ref.current = null;
      ema50Ref.current = null;
      markerPluginRef.current = null;
      priceLinesRef.current = [];
      datasetRef.current = null;
    };
  }, []);

  useEffect(() => {
    const candles = candleSeriesRef.current;
    if (!candles) return;
    candles.setData(finalized.map(toChartCandle));
    if (developing) candles.update(toChartCandle(developing));
    ema20Ref.current?.setData(indicators.ema20 ? calculateEma(finalized, 20) : []);
    ema50Ref.current?.setData(indicators.ema50 ? calculateEma(finalized, 50) : []);
    if (finalized.length) {
      const timeframe = finalized[0].timeframe;
      const previous = datasetRef.current;
      const substantialHistoryChange =
        previous === null
        || previous.timeframe !== timeframe
        || Math.abs(finalized.length - previous.count) > 20;
      if (substantialHistoryChange) {
        const totalBars = finalized.length + (developing ? 1 : 0);
        const visibleBars = Math.min(150, totalBars);
        chartRef.current?.timeScale().setVisibleLogicalRange({
          from: totalBars - visibleBars,
          to: totalBars + 5,
        });
      }
      datasetRef.current = { timeframe, count: finalized.length };
    }
  }, [developing, finalized, indicators.ema20, indicators.ema50]);

  useEffect(() => {
    const series = candleSeriesRef.current;
    if (!series) return;
    for (const line of priceLinesRef.current) series.removePriceLine(line);
    priceLinesRef.current = [];
    markerPluginRef.current?.setMarkers([]);
    if (!analysis && !priceAction) return;

    if (indicators.levels) {
      const supportLevels = priceAction?.support_levels.map((item) => item.price)
        ?? analysis?.market_context.support_levels
        ?? [];
      const resistanceLevels = priceAction?.resistance_levels.map((item) => item.price)
        ?? analysis?.market_context.resistance_levels
        ?? [];
      for (const value of supportLevels) {
        addPriceLine(series, priceLinesRef, value, "SUPPORT", "#2aa67e", LineStyle.Dotted);
      }
      for (const value of resistanceLevels) {
        addPriceLine(series, priceLinesRef, value, "RESISTANCE", "#d27f85", LineStyle.Dotted);
      }
      const levels = priceAction?.trade_plan ?? analysis?.signal.risk_levels;
      if (levels) {
        addPriceLine(series, priceLinesRef, levels.entry_low, "ENTRY LOW", "#42d4ff", LineStyle.Dashed);
        addPriceLine(series, priceLinesRef, levels.entry_high, "ENTRY HIGH", "#42d4ff", LineStyle.Dashed);
        addPriceLine(series, priceLinesRef, levels.stop, "STOP", "#ec6571", LineStyle.Solid);
        addPriceLine(series, priceLinesRef, levels.target1, "TARGET 1", "#24c995", LineStyle.Dashed);
        addPriceLine(series, priceLinesRef, levels.target2, "TARGET 2", "#24c995", LineStyle.Dotted);
        addPriceLine(series, priceLinesRef, levels.target3, "TARGET 3", "#24c995", LineStyle.Dotted);
      }
    }

    if (indicators.signals && analysis) {
      const markers: SeriesMarker<Time>[] = analysis.historical_signals.map((item) => ({
        time: toTimestamp(item.occurred_at),
        position: item.direction === "BUY" ? "belowBar" : "aboveBar",
        color: item.direction === "BUY" ? "#24c995" : "#ec6571",
        shape: item.direction === "BUY" ? "arrowUp" : "arrowDown",
        text: item.status,
      }));
      if (analysis.signal.direction !== "WAIT") {
        markers.push({
          time: toTimestamp(analysis.signal.decision_time),
          position: analysis.signal.direction === "BUY" ? "belowBar" : "aboveBar",
          color: analysis.signal.direction === "BUY" ? "#4ce4ac" : "#ff7b86",
          shape: analysis.signal.direction === "BUY" ? "arrowUp" : "arrowDown",
          text: `${analysis.signal.direction} · ACTIVE`,
        });
      }
      markers.sort((left, right) => Number(left.time) - Number(right.time));
      markerPluginRef.current?.setMarkers(markers);
    }
  }, [analysis, priceAction, indicators.levels, indicators.signals]);

  return (
    <div
      className="chart-canvas"
      ref={containerRef}
      aria-label="Interactive NIFTY 50 candlestick chart"
    />
  );
}

function addPriceLine(
  series: ISeriesApi<"Candlestick">,
  reference: MutableRefObject<IPriceLine[]>,
  value: string,
  title: string,
  color: string,
  lineStyle: LineStyle,
) {
  const price = Number(value);
  if (!Number.isFinite(price)) return;
  reference.current.push(
    series.createPriceLine({
      price,
      color,
      lineWidth: 1,
      lineStyle,
      axisLabelVisible: true,
      title,
    }),
  );
}

function calculateEma(candles: CandleContract[], period: number) {
  if (!candles.length) return [];
  const multiplier = 2 / (period + 1);
  let value = Number(candles[0].close);
  return candles.map((candle, index) => {
    const close = Number(candle.close);
    value = index === 0 ? close : close * multiplier + value * (1 - multiplier);
    return { time: toTimestamp(candle.opens_at), value };
  });
}

function toChartCandle(candle: CandleContract) {
  return {
    time: toTimestamp(candle.opens_at),
    open: Number(candle.open),
    high: Number(candle.high),
    low: Number(candle.low),
    close: Number(candle.close),
  };
}

function toTimestamp(value: string) {
  return Math.floor(new Date(value).getTime() / 1_000) as UTCTimestamp;
}

const IST_CLOCK = new Intl.DateTimeFormat("en-IN", {
  timeZone: "Asia/Kolkata",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

const IST_CLOCK_WITH_SECONDS = new Intl.DateTimeFormat("en-IN", {
  timeZone: "Asia/Kolkata",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hourCycle: "h23",
});

const IST_DAY = new Intl.DateTimeFormat("en-IN", {
  timeZone: "Asia/Kolkata",
  day: "2-digit",
  month: "short",
});

const IST_MONTH = new Intl.DateTimeFormat("en-IN", {
  timeZone: "Asia/Kolkata",
  month: "short",
});

const IST_YEAR = new Intl.DateTimeFormat("en-IN", {
  timeZone: "Asia/Kolkata",
  year: "numeric",
});

const IST_CROSSHAIR = new Intl.DateTimeFormat("en-IN", {
  timeZone: "Asia/Kolkata",
  day: "2-digit",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

function formatIstTick(time: Time, tickMarkType: number): string {
  const value = chartTimeToDate(time);
  if (tickMarkType === 0) return IST_YEAR.format(value);
  if (tickMarkType === 1) return IST_MONTH.format(value);
  if (tickMarkType === 2) return IST_DAY.format(value);
  if (tickMarkType === 4) return IST_CLOCK_WITH_SECONDS.format(value);
  return IST_CLOCK.format(value);
}

function formatIstCrosshairTime(time: Time): string {
  return `${IST_CROSSHAIR.format(chartTimeToDate(time))} IST`;
}

function chartTimeToDate(time: Time): Date {
  if (typeof time === "number") return new Date(time * 1_000);
  if (typeof time === "string") return new Date(`${time}T00:00:00Z`);
  return new Date(Date.UTC(time.year, time.month - 1, time.day));
}
