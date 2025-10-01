# Cross domain investigation: applying structural dynamics theory to financial market prediction and risk assessment

A research proposal by kashani@alum.sharif.edu
PhD student at University of Tabriz
Civil engineering department
Structural engineering

## Highlights

* Structural systems have similarities with financial systems
* Structural dynamics theory is mostly unexplored in financial markets
* There are innovation opportunities due to the cross-domain gap
* Novel models can be presented for financial systems based on structural dynamics

## Keywords

Structural dynamics, financial markets, econophysics, algorithmic 
trading, cross-domain modeling, quantitative finance

## Abstract

Financial markets exhibit nonlinear dynamic behaviors significantly similar to structural engineering systems: oscillations, resonances, shock responses, and collapses. Nonetheless the sophisticated mathematical toolkit developed for structural dynamics remains largely unapplied to financial analysis. This research proposes a systematic investigation for applying structural dynamics methods — probably including modal analysis, spectral decomposition, and finite element modeling - into quantitative finance. With algorithmic trading now comprising `92%` of market volume and daily forex turnover exceeding `$7.5` trillion, even marginal improvements in prediction accuracy could provide significant economic impact. Preliminary results by proof-of-concept implementations on financial timeseries are revealing patterns and insights not visible by traditional financial tools. This multidisciplinary approach could provide opportunities for market understanding, risk management, and algorithmic trading strategy development.

## Research philosophy and motivation

Structural systems and financial markets share foundational similarities:

* Dynamic behavior under external forces
   * Financial forces of buyers against sellers
   * Financial forces of bulls against bears
* Extreme events
   * Structural collapse
   * Market crashes
* Characteristic frequency and damping behavior
* Need for real-time monitoring and anomaly detection
* Support and resistance terminology for both structures and price action
* Similar patterns of stress buildup and sudden release
* Similarities between structural vibrations and market fluctuations
* Equilibrium concept
   * Structural forces
   * Supply and demand across macro and micro economy
* Financial friction and mechanical one
* Nonlinear dynamics behavior
* Stochastic nature of seismic loads and financial shocks
* Shock propagation through the financial network and structural network
* Thresholds
   * Financial tipping points
   * Structural buckling
* Forensic
   * Structural forensic analysis involves investigation and inspection of structures
   * Technical analysis involves study of past market data for forecasting price direction
* ...

Despite the similarities, the rich structural dynamics toolkit remains mostly unexplored in quantitative finance. This gap presents opportunities for practical innovation for:

* Theoretical contributions to econophysics
* Novel predictive tools based on physical systems theory
* Improved understanding of financial markets through physical systems

## Related work

The book "The Physics of Wall Street" by James Owen Weatherall mentions a geophysicist who used a model designed for earthquakes to predict a massive stock market crash [1]. Also, it discusses about a physicist-run hedge fund that earned `2478.6%` over the course of the 1990s. Financial industry has been using physical models to effectively figure out the financial behavior. There are some academic studies carried out, but systematic investigation of structural dynamics remains mostly absent.

## Literature gap

TODO.

## Background

The author's educational and industrial background is:

* Civil engineering, BSc
* Earthquake engineering, MSc
* Structural engineering, MSc
* 10+ years of professional software development by C++, Go, Python, JS, C#, and more
* Experience with algorithm development

This multidisciplinary background would provide the opportunity to:

* Investigate mathematical similarities between physical and financial systems
* Implement efficient algorithms
* Innovate theoretical toolkits for practical applications

## Significance

This research could have significant impact due to:

* Significant academic gap between engineering fields and quantitative finance
   * Lack of enough multidisciplinary researchers in both fields
* Significant trading volume of financial markets
   * Currency market alone has had a `$7.5` trillion _daily_ volume in 2022 [3]
   * `92%` of trading is done by algorithms and computers in 2019 [4]

## General research questions

This research proposal would try to answer these questions, at least:

* Can structural dynamics methods identify specific characteristics of financial markets?
* Can structural dynamics methods help with financial timeseries forecasting?
* Can structural dynamics methods predict market crashes?
* Can structural dynamics methods detect market regime changes?
* How do structural dynamics methods apply to financial markets?

## Specific research hypotheses

Beyond the general questions, this research might test specific hypotheses:

* Simple mass-spring systems can predict the price action with a win rate of above `50%`.
* Damping ratios calculated from price oscillations can predict volatility regime changes `2-5` days in advance with `>70%` accuracy.
* Modal decomposition of price series will reveal periodic components invisible to traditional technical analysis.
* Finite element modeling will identify risk `24-48` hours before traditional methods.
* Shock response from news announcements follow laws similar to earthquake relationships.
* ...

## Research methodology

The following plan can be outlined.

* Stage 1: literature review
* Stage 2: investigation and selection of structural dynamic methods
   * Investigate correspondence between physical and financial systems
   * Outline mathematical mapping between physical and financial systems, like:
      - Mass (m): market capitalization or liquidity
      - Stiffness (k): mean reversion strength
      - Damping (c): volatility decay rate
      - Displacement (x): price deviation from equilibrium
      - Velocity (ẋ): returns or price change rate
      - Acceleration (ẍ): return acceleration or volatility
      - Force (F): order flow imbalance or buy-sell pressure
      - ...
   * Investigate candidates of investigation, like these structural dynamic methods:
      - Mass-spring systems
      - Modal analysis
      - Natural frequency detection
      - Spectral analysis
      - Fourier transform
      - Structural health monitoring methods
      - Damping ratio
      - Finite element analysis, FEA
      - ...
* Stage 3: applying structural dynamic methods to financial markets
   * Novel mathematical framework development
* Stage 4: implementation
   * C++ high performance computing for live testing
   * Python for investigation and back-testing
   * Implementation and visualization on MetaTrader platform
   * Efficiency optimization
* Stage 5: validation and testing
   * Back-testing with multiple decades of historic market data
   * Live testing on demo trading accounts

Potential mapping table for stage 2:

| Physics Term | Symbol | Financial Interpretation and Possible Equivalent |
|--------------|--------|--------------------------------------------------|
| Mass         | m      | market capitalization or liquidity               |
| Stiffness    | k      | mean reversion strength                          |
| Damping      | c      | volatility decay rate                            |
| Displacement | x      | price deviation from equilibrium                 |
| Velocity     | ẋ      | returns or price change rate                     |
| Acceleration | ẍ      | return acceleration or volatility                |
| Force        | F      | order flow imbalance or buy-sell pressure        |
| ...          | ...    | ...                                              |

## Expected contributions

Scientific contributions:

* Novel systematic application of structural dynamics into finance
* New class of theoretical models for market behavior
* Mathematical framework bridging physical and financial systems

Practical contributions:

* Improved market prediction algorithms
* Novel trading signal generation methods
* Innovative risk detection tools

## Timeline

A rough timeline might be approximately estimated as follows.

* Year 1: literature review, investigating methods to adopt
* Year 2 and 3
   * Agile sprints of mathematical framework development for methods
   * Agile sprints of implementation and validation
* Year 4: optimization and dissertation writing

## Preliminary results

As a proof-of-concept, an MQL5 code on platform MetaTrader5 is developed. Hilbert-Huang Transform (HHT) approach is employed to carry out the frequency spectrum analysis. HHT is particularly useful for non-linear and non-stationary signals. HHT is commonly applied to seismic signals, here HHT is applied to financial timeseries.

Some specifics of the proof-of-concept code are:

* HHT is applied to the timeseries of the currency pairs
* Multiple currency pairs like EUR/USD and USD/JPY are investigated
* Analysis duration has been from multiple months to multiple years
* Any timeframe can be tried out, like M1, M5, H1 and so on
* Dominant periods and cycles are identified and visualized
* Number of candles to be analyzed is an adjustable parameter

The preliminary results indicate that systematic investigation could contribute multidisciplinary innovations inspired by structural dynamics into financial market analysis. Novel contributions to both theory and practice are highly probable.

## Expectations

Possibly and potentially, some expectations are kept in mind.

* Expected journal publications, like The Journal of Computational Finance
* Potential industry partnerships
* Opensource software deliverables

## References

[1]: Weatherall, James Owen. 2013. *The Physics of Wall Street*. Retrieved from https://archive.org/details/ThePhysicsOfWallStreetDewey332.63209WEAWeatherallJamesOwen
[2]: Reference removed
[3]: Bank for International Settlements (BIS). 2022. "Foreign Exchange Turnover Statistics." Retrieved from https://www.bis.org/statistics/rpfx22_fx.htm
[4]: Groette, Oddmund. 2024. "What Percentage of Trading Is Algorithmic?" *Quantified Strategies*. Retrieved from https://www.quantifiedstrategies.com/what-percentage-of-trading-is-algorithmic
