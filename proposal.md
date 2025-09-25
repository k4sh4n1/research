# Cross domain investigation: applying structural dynamics theory to financial market prediction and risk assessment

A research proposal by kashani@alum.sharif.edu

## Highlights

* Structural systems have similarities with financial systems
* Structural dynamics theory is mostly unexplored in financial markets
* There are innovation opportunities due to the cross-domain gap
* Novel models can be presented for financial systems based on structural dynamics

## Keywords

Structural dynamics, financial markets, econophysics, algorithmic 
trading, cross-domain modeling, quantitative finance

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
* ...

Despite the similarities, the rich structural dynamics toolkit remains mostly unexplored in quantitative finance. This gap presents opportunities for practical innovation for:

* Theoretical contributions to econophysics
* Novel predictive tools based on physical systems theory
* Improved understanding of financial markets through physical systems

## Related work

The book "The Physics of Wall Street" by James Owen Weatherall mentions a geophysicist who used a model designed for earthquakes to predict a massive stock market crash [1]. Also, it discusses about a physicist-run hedge fund that earned `2478.6%` over the course of the 1990s. Financial industry has been using physical models to effectively figure out the financial behavior. There are some academic studies carried out, but systematic investigation of structural dynamics remains mostly absent.

## Background

The author's educational and industrial background is:

* Civil engineering, BSc
* Earthquake engineering, MSc
* Structural engineering, MSc
* 10+ years of professional software development by C++, Go, Python, JS, C#, and more
* Experience with algorithm development

This multidisciplinary background would enable to:

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
   * Implementation on MetaTrader platform
   * Efficiency optimization
* Stage 5: validation and testing
   * Back-testing with multiple decades of historic market data
   * Live testing on demo trading accounts

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

A rough timeline might be outlined as follows.

* Year 1: literature review, investigating methods to adopt
* Year 2: mathematical framework development for methods
* Year 3: implementation and validation
* Year 4: optimization and dissertation writing

## Preliminary results

Some MQL code on MetaTrader platform are developed.

TODO: specifics.

The preliminary results indicate that systematic investigation could contribute multidisciplinary innovations inspired by structural dynamics into financial market analysis. Novel contributions to both theory and practice are highly probable.

## References

[1]: https://archive.org/details/ThePhysicsOfWallStreetDewey332.63209WEAWeatherallJamesOwen
[3]: https://www.bis.org/statistics/rpfx22_fx.htm
[4]: https://www.quantifiedstrategies.com/what-percentage-of-trading-is-algorithmic
