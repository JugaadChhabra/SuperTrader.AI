# # import base64 
# # import socketio
# # from datetime import datetime


# # #Get User ID and Session Token
# # session_key = "SESSION_TOKEN_FROM_CUSTOMER_DETAILS_API"
# # #e.g session_key = "QUYyOTUzMTM6NjY5ODc5NzY="

# # user_id, session_token = base64.b64decode(session_key.encode('ascii')).decode('ascii').split(":")
# # #e.g Decoded value - AF296713:66987976, after split user_id = AF295313, session_token = 6698797

# # # Python Socket IO Client
# # sio = socketio.Client()

# # script_code = "4.1!2885" #Subscribe more than one stock at a time

# # auth = {"user": user_id, "token": session_token} 

# # sio.connect("https://livestream.icicidirect.com", headers={"User-Agent":"python-socketio[client]/socket"}, 
# #                 auth=auth, transports="websocket", wait_timeout=3)

# # if sio.connected:
# #     print("Connection established successfully!")
# # else:
# #     print("Failed to connect.")

# # tux_to_user_value = {
# #     "orderFlow": {
# #         "B": "Buy",
# #         "S": "Sell",
# #         "N": "NA"
# #     },
# #     "limitMarketFlag": {
# #         "L": "Limit",
# #         "M": "Market",
# #         "S": "StopLoss"
# #     },
# #     "orderType": {
# #         "T": "Day",
# #         "I": "IoC",
# #         "V": "VTC"
# #     },
# #     "productType": {
# #         "F": "Futures",
# #         "O": "Options",
# #         "P": "FuturePlus",
# #         "U": "FuturePlus_sltp",
# #         "I": "OptionPlus",
# #         "C": "Cash",
# #         "Y": "eATM",
# #         "B": "BTST",
# #         "M": "Margin",
# #         "T": "MarginPlus"
# #     },
# #     "orderStatus": {
# #         "A": "All",
# #         "R": "Requested",
# #         "Q": "Queued",
# #         "O": "Ordered",
# #         "P": "Partially Executed",
# #         "E": "Executed",
# #         "J": "Rejected",
# #         "X": "Expired",
# #         "B": "Partially Executed And Expired",
# #         "D": "Partially Executed And Cancelled",
# #         "F": "Freezed",
# #         "C": "Cancelled"
# #     },
# #     "optionType": {
# #         "C": "Call",
# #         "P": "Put",
# #         "*": "Others"
# #     },
# # }

# # # parse market depth

# # def parse_market_depth(self, data, exchange):
# #     depth = []
# #     counter = 0
# #     for lis in data:
# #         counter += 1
# #         dict = {}
# #         if exchange == '1':
# #             dict["BestBuyRate-"+str(counter)] = lis[0]
# #             dict["BestBuyQty-"+str(counter)] = lis[1]
# #             dict["BestSellRate-"+str(counter)] = lis[2]
# #             dict["BestSellQty-"+str(counter)] = lis[3]
# #             depth.append(dict)
# #         else:
# #             dict["BestBuyRate-"+str(counter)] = lis[0]
# #             dict["BestBuyQty-"+str(counter)] = lis[1]
# #             dict["BuyNoOfOrders-"+str(counter)] = lis[2]
# #             dict["BuyFlag-"+str(counter)] = lis[3]
# #             dict["BestSellRate-"+str(counter)] = lis[4]
# #             dict["BestSellQty-"+str(counter)] = lis[5]
# #             dict["SellNoOfOrders-"+str(counter)] = lis[6]
# #             dict["SellFlag-"+str(counter)] = lis[7]
# #             depth.append(dict)
# #     return depth


# # # parsing logic
# # def parse_data(data):
# #     if data and type(data) == list and len(data) > 0 and type(data[0]) == str and "!" not in data[0]:
# #         order_dict = {}
# #         order_dict["sourceNumber"] = data[0]                            #Source Number
# #         order_dict["group"] = data[1]                                   #Group
# #         order_dict["userId"] = data[2]                                  #User_id
# #         order_dict["key"] = data[3]                                     #Key
# #         order_dict["messageLength"] = data[4]                           #Message Length
# #         order_dict["requestType"] = data[5]                             #Request Type
# #         order_dict["messageSequence"] = data[6]                         #Message Sequence
# #         order_dict["messageDate"] = data[7]                             #Date
# #         order_dict["messageTime"] = data[8]                             #Time
# #         order_dict["messageCategory"] = data[9]                         #Message Category
# #         order_dict["messagePriority"] = data[10]                        #Priority
# #         order_dict["messageType"] = data[11]                            #Message Type
# #         order_dict["orderMatchAccount"] = data[12]                      #Order Match Account
# #         order_dict["orderExchangeCode"] = data[13]                      #Exchange Code
# #         if data[11] == '4' or data[11] == '5':
# #             order_dict["stockCode"] = data[14]                     #Stock Code
# #             order_dict["orderFlow"] = tux_to_user_value['orderFlow'].get(str(data[15]).upper(),str(data[15]))                          # Order Flow
# #             order_dict["limitMarketFlag"] = tux_to_user_value['limitMarketFlag'].get(str(data[16]).upper(),str(data[16]))                    #Limit Market Flag
# #             order_dict["orderType"] = tux_to_user_value['orderType'].get(str(data[17]).upper(),str(data[17]))                          #OrderType
# #             order_dict["orderLimitRate"] = data[18]                     #Limit Rate
# #             order_dict["productType"] = tux_to_user_value['productType'].get(str(data[19]).upper(),str(data[19]))                        #Product Type
# #             order_dict["orderStatus"] = tux_to_user_value['orderStatus'].get(str(data[20]).upper(),str(data[20]))                        # Order Status
# #             order_dict["orderDate"] = data[21]                          #Order  Date
# #             order_dict["orderTradeDate"] = data[22]                     #Trade Date
# #             order_dict["orderReference"] = data[23]                     #Order Reference
# #             order_dict["orderQuantity"] = data[24]                      #Order Quantity
# #             order_dict["openQuantity"] = data[25]                       #Open Quantity
# #             order_dict["orderExecutedQuantity"] = data[26]              #Order Executed Quantity
# #             order_dict["cancelledQuantity"] = data[27]                  #Cancelled Quantity
# #             order_dict["expiredQuantity"] = data[28]                    #Expired Quantity
# #             order_dict["orderDisclosedQuantity"] = data[29]             # Order Disclosed Quantity
# #             order_dict["orderStopLossTrigger"] = data[30]               #Order Stop Loss Triger
# #             order_dict["orderSquareFlag"] = data[31]                    #Order Square Flag
# #             order_dict["orderAmountBlocked"] = data[32]                 # Order Amount Blocked
# #             order_dict["orderPipeId"] = data[33]                        #Order PipeId
# #             order_dict["channel"] = data[34]                            #Channel
# #             order_dict["exchangeSegmentCode"] = data[35]                #Exchange Segment Code
# #             order_dict["exchangeSegmentSettlement"] = data[36]          #Exchange Segment Settlement 
# #             order_dict["segmentDescription"] = data[37]                 #Segment Description
# #             order_dict["marginSquareOffMode"] = data[38]                #Margin Square Off Mode
# #             order_dict["orderValidDate"] = data[40]                     #Order Valid Date
# #             order_dict["orderMessageCharacter"] = data[41]              #Order Message Character
# #             order_dict["averageExecutedRate"] = data[42]                #Average Exited Rate
# #             order_dict["orderPriceImprovementFlag"] = data[43]          #Order Price Flag
# #             order_dict["orderMBCFlag"] = data[44]                       #Order MBC Flag
# #             order_dict["orderLimitOffset"] = data[45]                   #Order Limit Offset
# #             order_dict["systemPartnerCode"] = data[46]                  #System Partner Code
# #         elif data[11] == '6' or data[11] == '7':
# #             order_dict["stockCode"] = data[14]                         #stockCode
# #             order_dict["productType"] =  tux_to_user_value['productType'].get(str(data[15]).upper(),str(data[15]))                        #Product Type
# #             order_dict["optionType"] = tux_to_user_value['optionType'].get(str(data[16]).upper(),str(data[16]))                         #Option Type
# #             order_dict["exerciseType"] = data[17]                       #Exercise Type
# #             order_dict["strikePrice"] = data[18]                        #Strike Price
# #             order_dict["expiryDate"] = data[19]                         #Expiry Date
# #             order_dict["orderValidDate"] = data[20]                     #Order Valid Date
# #             order_dict["orderFlow"] = tux_to_user_value['orderFlow'].get(str(data[21]).upper(),str(data[21]))                          #Order  Flow
# #             order_dict["limitMarketFlag"] = tux_to_user_value['limitMarketFlag'].get(str(data[22]).upper(),str(data[22]))                    #Limit Market Flag
# #             order_dict["orderType"] = tux_to_user_value['orderType'].get(str(data[23]).upper(),str(data[23]))                          #Order Type
# #             order_dict["limitRate"] = data[24]                          #Limit Rate
# #             order_dict["orderStatus"] = tux_to_user_value['orderStatus'].get(str(data[25]).upper(),str(data[25]))                        #Order Status
# #             order_dict["orderReference"] = data[26]                     #Order Reference
# #             order_dict["orderTotalQuantity"] = data[27]                 #Order Total Quantity
# #             order_dict["executedQuantity"] = data[28]                   #Executed Quantity
# #             order_dict["cancelledQuantity"] = data[29]                  #Cancelled Quantity
# #             order_dict["expiredQuantity"] = data[30]                    #Expired Quantity
# #             order_dict["stopLossTrigger"] = data[31]                    #Stop Loss Trigger
# #             order_dict["specialFlag"] = data[32]                        #Special Flag
# #             order_dict["pipeId"] = data[33]                             #PipeId
# #             order_dict["channel"] = data[34]                            #Channel
# #             order_dict["modificationOrCancelFlag"] = data[35]           #Modification or Cancel Flag
# #             order_dict["tradeDate"] = data[36]                          #Trade Date
# #             order_dict["acknowledgeNumber"] = data[37]                  #Acknowledgement Number
# #             order_dict["stopLossOrderReference"] = data[37]             #Stop Loss Order Reference
# #             order_dict["totalAmountBlocked"] = data[38]                 # Total Amount Blocked
# #             order_dict["averageExecutedRate"] = data[39]                #Average Executed Rate
# #             order_dict["cancelFlag"] = data[40]                         #Cancel Flag
# #             order_dict["squareOffMarket"] = data[41]                    #SquareOff Market
# #             order_dict["quickExitFlag"] = data[42]                      #Quick Exit Flag
# #             order_dict["stopValidTillDateFlag"] = data[43]              #Stop Valid till Date Flag
# #             order_dict["priceImprovementFlag"] = data[44]               #Price Improvement Flag
# #             order_dict["conversionImprovementFlag"] = data[45]          #Conversion Improvement Flag
# #             order_dict["trailUpdateCondition"] = data[45]               #Trail Update Condition
# #             order_dict["systemPartnerCode"] = data[46]                  #System Partner Code
# #         return order_dict
# #     exchange = str.split(data[0], '!')[0].split('.')[0]
# #     data_type = str.split(data[0], '!')[0].split('.')[1]
# #     if exchange == '6':
# #         data_dict = {}
# #         data_dict["symbol"] = data[0]
# #         data_dict["AndiOPVolume"] = data[1]
# #         data_dict["Reserved"] = data[2]
# #         data_dict["IndexFlag"] = data[3]
# #         data_dict["ttq"] = data[4]
# #         data_dict["last"] = data[5]
# #         data_dict["ltq"] = data[6]
# #         data_dict["ltt"] = datetime.fromtimestamp(data[7]).strftime('%c')
# #         data_dict["AvgTradedPrice"] = data[8]
# #         data_dict["TotalBuyQnt"] = data[9]
# #         data_dict["TotalSellQnt"] = data[10]
# #         data_dict["ReservedStr"] = data[11]
# #         data_dict["ClosePrice"] = data[12]
# #         data_dict["OpenPrice"] = data[13]
# #         data_dict["HighPrice"] = data[14]
# #         data_dict["LowPrice"] = data[15]
# #         data_dict["ReservedShort"] = data[16]
# #         data_dict["CurrOpenInterest"] = data[17]
# #         data_dict["TotalTrades"] = data[18]
# #         data_dict["HightestPriceEver"] = data[19]
# #         data_dict["LowestPriceEver"] = data[20]
# #         data_dict["TotalTradedValue"] = data[21]
# #         marketDepthIndex = 0
# #         for i in range(22, len(data)):
# #             data_dict["Quantity-"+str(marketDepthIndex)] = data[i][0]
# #             data_dict["OrderPrice-"+str(marketDepthIndex)] = data[i][1]
# #             data_dict["TotalOrders-"+str(marketDepthIndex)] = data[i][2]
# #             data_dict["Reserved-"+str(marketDepthIndex)] = data[i][3]
# #             data_dict["SellQuantity-"+str(marketDepthIndex)] = data[i][4]
# #             data_dict["SellOrderPrice-"+str(marketDepthIndex)] = data[i][5]
# #             data_dict["SellTotalOrders-"+str(marketDepthIndex)] = data[i][6]
# #             data_dict["SellReserved-"+str(marketDepthIndex)] = data[i][7]
# #             marketDepthIndex += 1
# #     elif data_type == '1':
# #         data_dict = {
# #             "symbol": data[0],
# #             "open": data[1],
# #             "last": data[2],
# #             "high": data[3],
# #             "low": data[4],
# #             "change": data[5],
# #             "bPrice": data[6],
# #             "bQty": data[7],
# #             "sPrice": data[8],
# #             "sQty": data[9],
# #             "ltq": data[10],
# #             "avgPrice": data[11],
# #             "quotes": "Quotes Data"
# #         }
# #         # For NSE & BSE conversion
# #         if len(data) == 21:
# #             data_dict["ttq"] = data[12]
# #             data_dict["totalBuyQt"] = data[13]
# #             data_dict["totalSellQ"] = data[14]
# #             data_dict["ttv"] = data[15]
# #             data_dict["trend"] = data[16]
# #             data_dict["lowerCktLm"] = data[17]
# #             data_dict["upperCktLm"] = data[18]
# #             data_dict["ltt"] = datetime.fromtimestamp(
# #                 data[19]).strftime('%c')
# #             data_dict["close"] = data[20]
# #         # For FONSE & CDNSE conversion
# #         elif len(data) == 23:
# #             data_dict["OI"] = data[12]
# #             data_dict["CHNGOI"] = data[13]
# #             data_dict["ttq"] = data[14]
# #             data_dict["totalBuyQt"] = data[15]
# #             data_dict["totalSellQ"] = data[16]
# #             data_dict["ttv"] = data[17]
# #             data_dict["trend"] = data[18]
# #             data_dict["lowerCktLm"] = data[19]
# #             data_dict["upperCktLm"] = data[20]
# #             data_dict["ltt"] = datetime.fromtimestamp(
# #                 data[21]).strftime('%c')
# #             data_dict["close"] = data[22]
# #     else:
# #         data_dict = {
# #             "symbol": data[0],
# #             "time": datetime.fromtimestamp(data[1]).strftime('%c'),
# #             "depth": parse_market_depth(data[2], exchange),
# #             "quotes": "Market Depth"
# #         }
# #     if exchange == '4' and len(data) == 21:
# #         data_dict['exchange'] = 'NSE Equity'
# #     elif exchange == '1':
# #         data_dict['exchange'] = 'BSE'
# #     elif exchange == '13':
# #         data_dict['exchange'] = 'NSE Currency'
# #     elif exchange == '4' and len(data) == 23:
# #         data_dict['exchange'] = 'NSE Futures & Options'
# #     elif exchange == '6':
# #         data_dict['exchange'] = 'Commodity'
# #     return data_dict

# # # CallBack functions to receive feeds
# # def on_ticks(data):
# #     ticks = parse_data(data)
# #     print(ticks)

# # sio.emit('join', script_code)
# # sio.on('stock', on_ticks)

# # sio.wait()

# # #Unwatch from the stock
# # sio.emit("leave", script_code)

# # #Disconnect from the server
# # sio.emit("disconnect", "transport close")



# import requests
# import json

# url = "https://api.icicidirect.com/breezeapi/api/v1/customerdetails"

# payload = json.dumps({
#   "SessionToken": "",
#   "AppKey": ""
# })
# headers = {
#   'Content-Type': 'application/json',
# }
# response = requests.request("GET", url, headers=headers, data=payload)
# data = json.loads(response.text)
# print(data)