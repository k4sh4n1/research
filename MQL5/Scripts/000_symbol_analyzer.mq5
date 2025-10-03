//+------------------------------------------------------------------+
//|                                                      ProjectName |
//|                                      Copyright 2020, CompanyName |
//|                                       http://www.companyname.net |
//+------------------------------------------------------------------+
#property copyright "Your Name"
#property link      "https://www.example.com"
#property version   "1.00"
#property script_show_inputs

#include <Math\Stat\Math.mqh>

#include <Trade\SymbolInfo.mqh>

input bool PrintToFile = true;  // Option to print results to a file
input bool PrintToLog = true;   // Option to print results to experts log

//--- parameters for writing data to file
input string FileName="DailySymbolAnalysis_Points.csv";

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class SymbolAnalyzer
  {
private:
   string            symbol;
   int               atrPeriod;
   int               spreadPeriod;
   int               deviationPeriod;

   double            CalculateATR(string symbolName, int period)
     {
      double atr[];
      ArraySetAsSeries(atr, true);
      int atrHandle = iATR(symbolName, PERIOD_D1, period);

      if(atrHandle == INVALID_HANDLE)
        {
         Print("Failed to create ATR indicator for ", symbolName);
         return 0.0;
        }

      int copied = CopyBuffer(atrHandle, 0, 0, period, atr);

      // Release indicator handle
      if(atrHandle != INVALID_HANDLE)
         IndicatorRelease(atrHandle);

      if(copied > 0)
        {
         // Convert ATR to point units
         double point = SymbolInfoDouble(symbolName, SYMBOL_POINT);
         return atr[0] / point;
        }
      return 0.0;
     }

   double            CalculateAverageSpread(string symbolName, int period)
     {
      double spreads[];
      ArrayResize(spreads, period);

      double point = SymbolInfoDouble(symbolName, SYMBOL_POINT);

      for(int i = 0; i < period; i++)
        {
         double bid = SymbolInfoDouble(symbolName, SYMBOL_BID);
         double ask = SymbolInfoDouble(symbolName, SYMBOL_ASK);
         spreads[i] = MathAbs(ask - bid) / point;
        }

      return MathMean(spreads);
     }

   double            CalculateTradeDeviation(string symbolName, int period)
     {
      double deviations[];
      ArrayResize(deviations, period);

      double point = SymbolInfoDouble(symbolName, SYMBOL_POINT);

      for(int i = 0; i < period; i++)
        {
         double dailyHigh[], dailyLow[];
         ArraySetAsSeries(dailyHigh, true);
         ArraySetAsSeries(dailyLow, true);

         CopyHigh(symbolName, PERIOD_D1, i, period, dailyHigh);
         CopyLow(symbolName, PERIOD_D1, i, period, dailyLow);

         if(ArraySize(dailyHigh) > 0 && ArraySize(dailyLow) > 0)
           {
            deviations[i] = MathAbs(dailyHigh[0] - dailyLow[0]) / point;
           }
        }

      return MathMean(deviations);
     }

public:
                     SymbolAnalyzer(int _atrPeriod = 14, int _spreadPeriod = 14, int _deviationPeriod = 14)
     {
      atrPeriod = _atrPeriod;
      spreadPeriod = _spreadPeriod;
      deviationPeriod = _deviationPeriod;
     }

   void              AnalyzeAllSymbols()
     {
      // Prepare file output if enabled
      int fileHandle = INVALID_HANDLE;
      if(PrintToFile)
        {
         fileHandle = FileOpen(FileName, FILE_READ|FILE_WRITE|FILE_CSV);
         if(fileHandle != INVALID_HANDLE)
           {
            FileWrite(fileHandle, "Symbol", "Daily ATR (Points)", "Daily Avg Spread (Points)", "Daily Trade Deviation (Points)");
            Alert("Folder: ", TerminalInfoString(TERMINAL_DATA_PATH), "\\MQL5\\Files\\");
           }
         else
           {
            Alert("Failed to open file: ", FileName, " Error code: ", GetLastError());
           }
        }

      // Get total number of symbols
      int totalSymbols = SymbolsTotal(true);

      for(int i = 0; i < totalSymbols; i++)
        {
         string currentSymbol = SymbolName(i, true);

         // Ensure symbol is selected and has daily data
         if(!SymbolSelect(currentSymbol, true))
            continue;

         // Check if daily data is available
         if(SeriesInfoInteger(currentSymbol, PERIOD_D1, SERIES_SYNCHRONIZED) == false)
           {
            Print("Daily data not synchronized for ", currentSymbol);
            continue;
           }

         // Calculate metrics for daily timeframe in point units
         double atr = CalculateATR(currentSymbol, atrPeriod);
         double avgSpread = CalculateAverageSpread(currentSymbol, spreadPeriod);
         double tradeDeviation = CalculateTradeDeviation(currentSymbol, deviationPeriod);

         // Output results
         string output = StringFormat("Symbol: %s, Daily ATR (Points): %.2f, Daily Avg Spread (Points): %.2f, Daily Trade Deviation (Points): %.2f",
                                      currentSymbol, atr, avgSpread, tradeDeviation);

         if(PrintToLog)
           {
            Print(output);
           }

         if(PrintToFile && fileHandle != INVALID_HANDLE)
           {
            FileWrite(fileHandle, currentSymbol,
                      DoubleToString(atr, 2),
                      DoubleToString(avgSpread, 2),
                      DoubleToString(tradeDeviation, 2));
           }
        }

      // Close file if opened
      if(fileHandle != INVALID_HANDLE)
        {
         FileClose(fileHandle);
        }
     }
  };

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnStart()
  {
   SymbolAnalyzer analyzer;
   analyzer.AnalyzeAllSymbols();
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
