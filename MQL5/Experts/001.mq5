//+------------------------------------------------------------------+
//|                                     SimpleCycleDetector.mq5     |
//|                                    Minimalist Cycle Detector    |
//+------------------------------------------------------------------+
#property copyright "2025"
#property version   "1.00"
#property strict

//--- Input parameters
input string   InpServerURL = "http://localhost:8080/algorithm/001"; // Server URL
input int      InpDataPoints = 200;          // Data points to analyze
input int      InpUpdateSeconds = 2;         // Update interval (seconds)
input bool     InpShowPanel = true;          // Show info panel
input int      InpPanelX = 20;               // Panel X position
input int      InpPanelY = 50;               // Panel Y position
input color    InpPanelBgColor = clrBlack;   // Panel background color
input color    InpTextColor = clrWhite;      // Default text color

//--- Global variables
datetime g_lastUpdateTime = 0;
string g_cycles_info = "Waiting for data...";
string g_prefix = "CYCLE_";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Initial update
   UpdateCycles();

//--- Set timer for periodic updates
   EventSetTimer(InpUpdateSeconds);

   Print("Simple Cycle Detector initialized");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Kill timer
   EventKillTimer();

//--- Clear objects and comment
   ObjectsDeleteAll(0, g_prefix);
   Comment("");
   ChartRedraw();
  }

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
   UpdateCycles();
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Display is handled by timer and UpdateCycles
  }

//+------------------------------------------------------------------+
//| Update cycles from server                                        |
//+------------------------------------------------------------------+
void UpdateCycles()
  {
//--- Get price data
   double prices[];
   ArrayResize(prices, InpDataPoints);

   int copied = CopyClose(_Symbol, PERIOD_CURRENT, 0, InpDataPoints, prices);
   if(copied != InpDataPoints)
     {
      Print("Failed to copy price data");
      return;
     }

//--- Reverse array to match chronological order
   ArrayReverse(prices);

//--- Convert to JSON
   string json_data = "[";
   for(int i = 0; i < InpDataPoints; i++)
     {
      if(i > 0)
         json_data += ",";
      json_data += DoubleToString(prices[i], _Digits);
     }
   json_data += "]";

//--- Send request and get response
   string response = SendRequest(json_data);

//--- Parse response
   if(response != "")
     {
      ParseResponse(response);
      DisplayInfo();  // Update display after parsing
     }
  }

//+------------------------------------------------------------------+
//| Send HTTP request to Python server                               |
//+------------------------------------------------------------------+
string SendRequest(string data)
  {
//--- Prepare request
   char post[], result[];
   string headers = "Content-Type: application/json\r\n";
   StringToCharArray(data, post);
   ArrayResize(post, StringLen(data));

   string result_headers;

//--- Send request
   ResetLastError();
   int res = WebRequest("POST", InpServerURL, headers, 5000, post, result, result_headers);

   if(res == 200)
     {
      return CharArrayToString(result);
     }
   else
      if(res == -1)
        {
         Print("WebRequest failed. Error: ", GetLastError());
         Print("Make sure to add ", InpServerURL, " to allowed URLs in Terminal settings");
        }

   return "";
  }

//+------------------------------------------------------------------+
//| Parse server response                                            |
//+------------------------------------------------------------------+
void ParseResponse(string response)
  {
//--- Simple JSON parsing for our specific response
   g_cycles_info = "=== DOMINANT CYCLES & PERIODS ===\n\n";

//--- Extract status
   string status = ExtractString(response, "\"status\": \"", "\"");
   if(status != "success")
     {
      g_cycles_info = "Error: " + ExtractString(response, "\"message\": \"", "\"");
      return;
     }

//--- Extract dominant periods
   string periods_str = ExtractArray(response, "\"dominant_periods\": [", "]");
   string frequencies_str = ExtractArray(response, "\"dominant_frequencies\": [", "]");

//--- Parse periods
   string periods[];
   int period_count = StringSplit(periods_str, ',', periods);

   g_cycles_info += "Main Periods (in bars):\n";
   for(int i = 0; i < MathMin(period_count, 5); i++)
     {
      StringTrimLeft(periods[i]);
      StringTrimRight(periods[i]);
      double period = StringToDouble(periods[i]);
      if(period > 0)
        {
         g_cycles_info += StringFormat("  • %.1f bars (~%.1f %s)\n",
                                       period,
                                       ConvertPeriodToTime(period),
                                       GetTimeframeUnit());
        }
     }

//--- Parse frequencies
   string frequencies[];
   int freq_count = StringSplit(frequencies_str, ',', frequencies);

   g_cycles_info += "\nFrequencies (cycles/bar):\n";
   for(int i = 0; i < MathMin(freq_count, 5); i++)
     {
      StringTrimLeft(frequencies[i]);
      StringTrimRight(frequencies[i]);
      double freq = StringToDouble(frequencies[i]);
      if(freq > 0)
        {
         g_cycles_info += StringFormat("  • %.4f\n", freq);
        }
     }

//--- Add detection method details
   string fft_periods = ExtractArray(response, "\"fft_based\": [", "]");
   string imf_periods = ExtractArray(response, "\"imf_based\": [", "]");
   string autocorr_periods = ExtractArray(response, "\"autocorr_based\": [", "]");

   g_cycles_info += "\n--- Detection Methods ---\n";
   g_cycles_info += "FFT: " + fft_periods + "\n";
   g_cycles_info += "IMF: " + imf_periods + "\n";
   g_cycles_info += "AutoCorr: " + autocorr_periods + "\n";

//--- Add update time
   g_cycles_info += "\nLast Update: " + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS);

   Print("Cycles updated successfully");
  }

//+------------------------------------------------------------------+
//| Display information as panel                                     |
//+------------------------------------------------------------------+
void DisplayInfo()
  {
   if(!InpShowPanel)
     {
      Comment(g_cycles_info);  // Fall back to comment if panel disabled
      return;
     }

//--- Clear comment since we're using panel
   Comment("");

//--- Clear previous objects
   ObjectsDeleteAll(0, g_prefix);

//--- Split info into lines
   string lines[];
   int line_count = StringSplit(g_cycles_info, '\n', lines);

//--- Calculate panel dimensions
   int panel_width = 400;
   int panel_height = (line_count + 1) * 17 + 10;

//--- Create OPAQUE background rectangle
   string bg_name = g_prefix + "BACKGROUND";
   if(ObjectCreate(0, bg_name, OBJ_RECTANGLE_LABEL, 0, 0, 0))
     {
      ObjectSetInteger(0, bg_name, OBJPROP_XDISTANCE, InpPanelX - 5);
      ObjectSetInteger(0, bg_name, OBJPROP_YDISTANCE, InpPanelY - 5);
      ObjectSetInteger(0, bg_name, OBJPROP_XSIZE, panel_width);
      ObjectSetInteger(0, bg_name, OBJPROP_YSIZE, panel_height);
      ObjectSetInteger(0, bg_name, OBJPROP_BGCOLOR, InpPanelBgColor);  // Solid black will be maximum opacity
      ObjectSetInteger(0, bg_name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(0, bg_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, bg_name, OBJPROP_WIDTH, 1);
      ObjectSetInteger(0, bg_name, OBJPROP_BACK, false);  // Foreground rendering
      ObjectSetInteger(0, bg_name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, bg_name, OBJPROP_SELECTED, false);
      ObjectSetInteger(0, bg_name, OBJPROP_HIDDEN, false);  // Fully visible
      ObjectSetInteger(0, bg_name, OBJPROP_ZORDER, 1000);  // High z-order
     }

//--- Create border
   string border_name = g_prefix + "BORDER";
   if(ObjectCreate(0, border_name, OBJ_RECTANGLE_LABEL, 0, 0, 0))
     {
      ObjectSetInteger(0, border_name, OBJPROP_XDISTANCE, InpPanelX - 6);
      ObjectSetInteger(0, border_name, OBJPROP_YDISTANCE, InpPanelY - 6);
      ObjectSetInteger(0, border_name, OBJPROP_XSIZE, panel_width + 2);
      ObjectSetInteger(0, border_name, OBJPROP_YSIZE, panel_height + 2);
      ObjectSetInteger(0, border_name, OBJPROP_BGCOLOR, clrNONE);
      ObjectSetInteger(0, border_name, OBJPROP_BORDER_TYPE, BORDER_RAISED);
      ObjectSetInteger(0, border_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, border_name, OBJPROP_COLOR, clrGray);
      ObjectSetInteger(0, border_name, OBJPROP_STYLE, STYLE_SOLID);
      ObjectSetInteger(0, border_name, OBJPROP_WIDTH, 1);
      ObjectSetInteger(0, border_name, OBJPROP_BACK, false);  // Also foreground
      ObjectSetInteger(0, border_name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, border_name, OBJPROP_SELECTED, false);
      ObjectSetInteger(0, border_name, OBJPROP_HIDDEN, false);  // Fully visible
      ObjectSetInteger(0, border_name, OBJPROP_ZORDER, 1001);  // Higher than background
     }

//--- Create text labels for each line (rest of the code remains the same)
   for(int i = 0; i < line_count && i < 25; i++)  // Limit to 25 lines
     {
      string obj_name = g_prefix + "LINE_" + IntegerToString(i);

      if(ObjectCreate(0, obj_name, OBJ_LABEL, 0, 0, 0))
        {
         ObjectSetInteger(0, obj_name, OBJPROP_XDISTANCE, InpPanelX);
         ObjectSetInteger(0, obj_name, OBJPROP_YDISTANCE, InpPanelY + i * 17);
         ObjectSetString(0, obj_name, OBJPROP_TEXT, lines[i]);
         ObjectSetString(0, obj_name, OBJPROP_FONT, "Consolas");
         ObjectSetInteger(0, obj_name, OBJPROP_FONTSIZE, 14);
         ObjectSetInteger(0, obj_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
         ObjectSetInteger(0, obj_name, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER);
         ObjectSetInteger(0, obj_name, OBJPROP_SELECTABLE, false);
         ObjectSetInteger(0, obj_name, OBJPROP_SELECTED, false);
         ObjectSetInteger(0, obj_name, OBJPROP_HIDDEN, false);  // Fully visible
         ObjectSetInteger(0, obj_name, OBJPROP_ZORDER, 1002);  // Highest z-order

         //--- Color coding based on content
         color text_color = InpTextColor;
         if(StringFind(lines[i], "===") >= 0)
            text_color = clrGold;
         else
            if(StringFind(lines[i], "Main Periods") >= 0 || StringFind(lines[i], "Frequencies") >= 0)
               text_color = clrAqua;
            else
               if(StringFind(lines[i], "---") >= 0)
                  text_color = clrSilver;
               else
                  if(StringFind(lines[i], "Last Update") >= 0)
                     text_color = clrGray;
                  else
                     if(StringFind(lines[i], "•") >= 0)
                        text_color = clrLightGreen;
                     else
                        if(StringFind(lines[i], "FFT:") >= 0 || StringFind(lines[i], "IMF:") >= 0 || StringFind(lines[i], "AutoCorr:") >= 0)
                           text_color = clrKhaki;

         ObjectSetInteger(0, obj_name, OBJPROP_COLOR, text_color);
        }
     }

   ChartRedraw();
  }

//+------------------------------------------------------------------+
//| Extract string value from JSON                                   |
//+------------------------------------------------------------------+
string ExtractString(string json, string start_marker, string end_marker)
  {
   int start = StringFind(json, start_marker);
   if(start < 0)
      return "";

   start += StringLen(start_marker);
   int end = StringFind(json, end_marker, start);
   if(end < 0)
      return "";

   return StringSubstr(json, start, end - start);
  }

//+------------------------------------------------------------------+
//| Extract array content from JSON                                  |
//+------------------------------------------------------------------+
string ExtractArray(string json, string start_marker, string end_marker)
  {
   int start = StringFind(json, start_marker);
   if(start < 0)
      return "";

   start += StringLen(start_marker);
   int end = StringFind(json, end_marker, start);
   if(end < 0)
      return "";

   string result = StringSubstr(json, start, end - start);
   StringReplace(result, " ", "");
   return result;
  }

//+------------------------------------------------------------------+
//| Convert period in bars to time                                   |
//+------------------------------------------------------------------+
double ConvertPeriodToTime(double period_bars)
  {
   int timeframe = PeriodSeconds() / 60; // Current timeframe in minutes
   return period_bars * timeframe;
  }

//+------------------------------------------------------------------+
//| Get timeframe unit string                                        |
//+------------------------------------------------------------------+
string GetTimeframeUnit()
  {
   int tf_minutes = PeriodSeconds() / 60;

   if(tf_minutes < 60)
      return "minutes";
   else
      if(tf_minutes < 1440)
         return "hours";
      else
         if(tf_minutes < 10080)
            return "days";
         else
            return "weeks";
  }
//+------------------------------------------------------------------+
