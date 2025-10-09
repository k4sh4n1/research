# Note

Apparently, MetaQuotes saves the robot algorithm `*.mq5` files here:

```bash
C:\Users\username\AppData\Roaming\MetaQuotes\Terminal\A_LONG_HASH\MQL5\Experts\*.mq5
```

# MetaTrader

To have access to MQL5 Expert Advisor or EA code on the MetaTrader platform:

```bash
cd C:\Users\m3\AppData\Roaming\MetaQuotes\Terminal\HASH\MQL5\Experts\

mklink /J "MQL5-Experts" "C:\Users\m3\repos\research-proposal\MQL5\Experts"
```

For the MQL5 script code:

```bash
cd C:\Users\m3\AppData\Roaming\MetaQuotes\Terminal\HASH\MQL5\Scripts\

mklink /J "MQL5-Scripts" "C:\Users\m3\repos\research-proposal\MQL5\Scripts"
```
