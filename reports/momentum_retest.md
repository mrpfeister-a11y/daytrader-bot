# 12-month momentum rule: honest re-test
Run 2026-11-27 04:41 UTC. Tested 2017-01-13 to 2026-11-24 (the first year of data is needed to measure 12-month momentum). Alpaca SIP daily prices, split/dividend adjusted. Top 5, equal dollars, buy at the week's first open, 8% stop, sell at the close of day 10. Paper research only.

| Version | Per year | Worst drop | Sharpe | Trades won | Trades/yr | Last 12 months |
|---|---|---|---|---|---|---|
| A. Original style: today's S&P 1500, no costs, stops fill at exactly -8% | +58.0% | -30% | 1.47 | 49% | 126 | +51.1% |
| B. Today's S&P 1500, real stop fills + 0.10% costs | +46.7% | -32% | 1.26 | 48% | 126 | +43.3% |
| C. All US stocks that still trade today (survivors only) | -1.7% | -79% | 0.17 | 36% | 126 | -23.9% |
| D. HONEST: all US stocks incl. delisted/bankrupt | -2.3% | -79% | 0.16 | 35% | 126 | -23.9% |
| E. Honest, higher costs (0.25% per trade) | -8.4% | -81% | 0.01 | 35% | 126 | -27.7% |
| F. Honest, no stop-loss | +6.9% | -84% | 0.41 | 49% | 126 | -56.7% |
| G. Honest, top 10 instead of top 5 | +7.9% | -57% | 0.39 | 37% | 252 | -5.5% |
| S&P 500 (SPY) buy & hold | +14.1% | -34% | 0.83 | | | +10.2% |

Random 5 stocks from the same honest list (20 tries): median +8.8%/yr, range -1.1% to +26.0%. The rule beat 0 of 20.

## Year by year
| Year | A | B | C | D | E | F | G | S&P 500 |
|---|---|---|---|---|---|---|---|---|
| 2017 | +10% | +5% | +2% | -8% | -12% | +3% | +9% | +20% |
| 2018 | +8% | +2% | +21% | +26% | +20% | +9% | +5% | -5% |
| 2019 | +32% | +26% | -9% | -7% | -11% | +40% | -4% | +31% |
| 2020 | +153% | +143% | +34% | +30% | +24% | +4% | +92% | +18% |
| 2021 | +42% | +34% | +63% | +68% | +56% | +74% | +31% | +29% |
| 2022 | +63% | +48% | +19% | +18% | +6% | +77% | +30% | -18% |
| 2023 | +88% | +60% | -3% | -3% | -7% | +19% | -9% | +26% |
| 2024 | +106% | +98% | -4% | -4% | -9% | +37% | +4% | +25% |
| 2025 | +14% | +8% | -65% | -65% | -66% | -63% | -46% | +18% |
| 2026 | +27% | +22% | -10% | -10% | -14% | -41% | +4% | +6% |

Worst honest trades: ALMS 2026-08 -56%, OSUR 2017-10 -29%, IOVA 2026-10 -23%, EH 2023-11 -19%, EOSE 2025-05 -16%, IBRX 2026-11 -13%, RGTI 2025-05 -13%, APEI 2024-08 -13%
Best honest trades: BTBT 2020-12 +389%, RIOT 2021-02 +145%, CLSK 2024-02 +116%, VKTX 2023-03 +90%, AXTI 2026-02 +89%

Universe: 7,831 US stocks (6,585 still trading), funds removed by name; today's S&P 1500 matched 1,484. Prices are dividend-adjusted, so the $5-$25 filter is slightly approximate in early years. Took 1 min.
