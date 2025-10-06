int OnInit()
{
   string data = "[1,2,3,4,5,6,7,8,9,10]";
   char post[], result[];
   string headers = "Content-Type: application/json\r\n";
   StringToCharArray(data, post, 0, StringLen(data));

   int res = WebRequest("POST", "http://127.0.0.1:8080/algorithm/001", 
                        headers, 5000, post, result, headers);
   Print("Res: ", res, " LastError: ", GetLastError());
   return INIT_SUCCEEDED;
}
