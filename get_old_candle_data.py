import requests as r

# from binance.client import Client
# from binance.websockets import BinanceSocketManager
# client = Client('OEWj7QP2h6ROVHPWWSt58zB4NCg9tsjwhyzU5F0s7q469f6IAkL17vcg8AYKRGwO', 'CWj7tNqeNsTbzrjvIAdQPiRY3pEFQe4VWDlZK4LA2ndMxqpI4CIT1oezFR4VMkyD')
import krakenex
import time
testData = []
def getOldCandleData():
    #oldData = client.get_historical_klines("BTCUSDT", '5m', "1 Jan, 2018")
    xxbt = r.get('https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=5').json()
    for keys, vals in xxbt.items():
        if keys == 'result':
            #print(vals)
            for key, val in vals.items():
                if type(val) == list:
                    #print('oldData', val[len(val)-2])
                    y = list(map(lambda x: [float(x[1]),float(x[2]),float(x[3]),float(x[4])], val))
    testData = y[-261:-1]
    #print(y[-10:])
    #print('testDataOld',len(testData))
    return testData
