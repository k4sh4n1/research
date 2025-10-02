//+------------------------------------------------------------------+
//| Include ZMQ library                                            |
//+------------------------------------------------------------------+
#include <Zmq/Zmq.mqh>

class PythonHHT
{
private:
   Context context;
   Socket  socket;

public:
   bool Initialize()
   {
      socket = context.socket(ZMQ_REQ);
      return socket.connect("tcp://localhost:5555");
   }

   string AnalyzePrices(double &prices[])
   {
      // Convert prices to JSON string
      string message = "[";
      for(int i = 0; i < ArraySize(prices); i++)
      {
         message += DoubleToString(prices[i], 5);
         if(i < ArraySize(prices)-1) message += ",";
      }
      message += "]";

      // Send and receive
      ZmqMsg request(message);
      socket.send(request);

      ZmqMsg reply;
      socket.recv(reply);

      return reply.getData();
   }
};
