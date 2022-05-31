# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

#from helper import helper

# --- Do not remove these libs ---
from email.policy import default
from unicodedata import decimal
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, RealParameter,
                                IStrategy, IntParameter, informative)

from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real  # noqa


# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Scalping(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1m'
    can_short: bool = True
    

    multiple_leverage = 5

    class HyperOpt:
        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.025, -0.015, decimals=4, name='stoploss')]
        # Define custom ROI space
        def roi_space():
            return [
                Integer(0, 120, name='roi_t1'),
                Integer(0, 60, name='roi_t2'),
                Integer(0, 40, name='roi_t3'),
                SKDecimal(0.025, 0.040, decimals=3, name='roi_p1'),
                SKDecimal(0.025, 0.040, decimals=3, name='roi_p2'),
                SKDecimal(0.025, 0.040, decimals=3, name='roi_p3'),
            ]
    profit = DecimalParameter(0.025,0.040,decimals=4 ,default=0.005,space="buy")
    loss = DecimalParameter(-0.025,-0.015,decimals=4, default=-0.004,space="buy")
    minimal_roi = {
        "0": profit.value
    }



    time_to_play = IntParameter(2, 4, default=3, space="buy")
    #spcific diff ema
    BTC = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    ETH = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    BCH = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    BNB = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    SOL = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    XRP = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    DOT = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    AVAX = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    ADA = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    ETC = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    FTM = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    FIL = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    TRX = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    AXS = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    SAND = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    NEAR = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    DOGE = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    MATIC = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    LTC = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    LINK = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    MANA = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    ATOM = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    KNC = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    XMR = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    SHIB = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    EOS = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    CRV = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    AAVE = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    WAVES = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    GALA = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    RUNE = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    SUSHI = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    XTZ = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    UNI = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    ZIL = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    LRC = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    XLM = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    KAVA = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    EGLD = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    ALGO = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    ZEC = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    THETA = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    YFI = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    INCH = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    VET = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    MKR = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    KSM = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    GRT = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    OMG = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    COMP = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")
    C98 = DecimalParameter(0.3,2.0,decimals=3,default=0.5,space="buy")


    stoploss = loss.value
    trailing_stop = False
    process_only_new_candles = False
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    startup_candle_count: int = 1000

    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'ema50_5m': {'color':'orange'},
            'ema200_5m': {'color': 'blue'}
        },
        'subplots': {
            "MACD": {
                'macd_hist': {'type': 'bar', 'plotly': {'opacity': 0.9}, 'color':'green'}
            },
        }
    }

    def informative_pairs(self):
        return [ ("BTC/USDT","5m")
            ,("ETH/USDT","5m")
            ,("BCH/USDT","5m")
            ,("BNB/USDT","5m")
            ,("SOL/USDT","5m")
            ,("XRP/USDT","5m")
            ,("DOT/USDT","5m")
            ,("AVAX/USDT","5m")
            ,("ADA/USDT","5m")
            ,("ETC/USDT","5m")
            ,("FTM/USDT","5m")
            ,("FIL/USDT","5m")
            ,("TRX/USDT","5m")
            ,("AXS/USDT","5m")
            ,("SAND/USDT","5m")
            ,("NEAR/USDT","5m")
            ,("DOGE/USDT","5m")
            ,("MATIC/USDT","5m")
            ,("LTC/USDT","5m")
            ,("LINK/USDT","5m")
            ,("MANA/USDT","5m")
            ,("ATOM/USDT","5m")
            ,("KNC/USDT","5m")
            ,("XMR/USDT","5m")
            ,("1000SHIB/USDT","5m")
            ,("EOS/USDT","5m")
            ,("CRV/USDT","5m")
            ,("AAVE/USDT","5m")
            ,("WAVES/USDT","5m")
            ,("GALA/USDT","5m")
            ,("RUNE/USDT","5m")
            ,("SUSHI/USDT","5m")
            ,("XTZ/USDT","5m")
            ,("UNI/USDT","5m")
            ,("ZIL/USDT","5m")
            ,("LRC/USDT","5m")
            ,("XLM/USDT","5m")
            ,("KAVA/USDT","5m")
            ,("EGLD/USDT","5m")
            ,("ALGO/USDT","5m")
            ,("ZEC/USDT","5m")
            ,("THETA/USDT","5m")
            ,("YFI/USDT","5m")
            ,("1INCH/USDT","5m")
            ,("VET/USDT","5m")
            ,("MKR/USDT","5m")
            ,("KSM/USDT","5m")
            ,("GRT/USDT","5m")
            ,("OMG/USDT","5m")
            ,("COMP/USDT","5m")
            ,("C98/USDT","5m")
        ]

    @informative('5m')
    def populate_indicators_5m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema50'] = ta.EMA(dataframe['close'], timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe['close'], timeperiod = 200)
        return dataframe

    
    def leverage(self, pair: str, current_time: 'datetime', current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        return self.multiple_leverage


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['macd_hist'] = ta.MACD(dataframe['close'], timeperiod=14)[2]
        #dataframe = self.cross_over(dataframe)

        percent_ema_diff = []
        macd_mean = []
        macd_count = 0
        macd_sum = 0
        already_cross = False
        ema_crossed = [False]
        for i in range(len(dataframe)):
            macd = dataframe.iloc[i]['macd_hist']
            if macd > 0 or macd < 0:
                macd_count += 1
                macd_sum += abs(macd)
            if macd_count != 0:
                macd_mean.append(macd_sum/macd_count)
            else:
                macd_mean.append(None)
            #print(macd_count,macd_sum)
            if i>=1:
                if already_cross:
                    ema_crossed.append(True)
                else:
                    ema50_1 = dataframe.iloc[i-1]['ema50_5m']
                    ema200_1 = dataframe.iloc[i]['ema200_5m']
                    ema50_2 = dataframe.iloc[i]['ema50_5m']
                    ema200_2 = dataframe.iloc[i]['ema200_5m']
                    if (ema50_1 > ema200_1 and ema50_2 < ema200_2) or (ema50_1 < ema200_1 and ema50_2 > ema200_2):
                        already_cross = True
                        ema_crossed.append(True)
                    else:
                        ema_crossed.append(False)
            percent_ema_diff.append(self.percent_ema_diff(dataframe.iloc[i]['ema50_5m'],dataframe.iloc[i]['ema200_5m']))

        dataframe['percent_ema_diff'] = percent_ema_diff
        dataframe['ema_crossed'] = ema_crossed
        dataframe['macd_mean'] = macd_mean 
        #helper.save_dataframe(dataframe)
        return dataframe

    
    def percent_ema_diff(self,ema1,ema2) -> bool:
        if ema2>ema1:
            ema1,ema2 = ema2,ema1
        return ((ema1-ema2)/ema1)*100

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        #TODO: remove and set static
        #self.stoploss = self.loss.value

       #long position
        time_open_long = 0
        enter_long = []
        opening_long = False
        opening_long_price = 0
        # short position
        time_open_short = 0
        enter_shot = []
        opening_short = False
        opening_short_price = 0

        min_ema_diff = self.pick_min_ema_diff(metadata['pair'])
        
        for i in range(len(dataframe)):
            current = dataframe.iloc[i]
            #reset data
            if current['ema50_5m'] > current['ema200_5m']:
                time_open_short = 0
            else:
                time_open_long = 0

            #Long position
            if (current['ema50_5m'] > current['ema200_5m']) and current['percent_ema_diff']>min_ema_diff and current['ema_crossed'] and (current['macd_hist']<-current['macd_mean']) and not opening_long and time_open_long < self.time_to_play.value:
                enter_long.append(True)
                opening_long = True
                time_open_long += 1
                opening_long_price = current['close']
            else:
                enter_long.append(False)
            if opening_long and (current['high'] >= opening_long_price*(1+self.profit.value)): #for take profit and stop loss
                opening_long = False
                opening_long_price = 0

            #Short position
            if (current['ema50_5m'] < current['ema200_5m']) and current['percent_ema_diff']>min_ema_diff and current['ema_crossed'] and (current['macd_hist']>current['macd_mean']) and not opening_short and time_open_short < self.time_to_play.value:
                enter_shot.append(True)
                opening_short = True
                time_open_short += 1
                opening_short_price = current['close']
            else:
                enter_shot.append(False)
            if opening_short and (current['low'] <= opening_short_price*(1-self.profit.value)):
                opening_short = False
                opening_short_price = 0

        dataframe['enter_long'] = enter_long
        dataframe['enter_short'] = enter_shot

        #helper.save_dataframe(dataframe,str(metadata['pair'].split('/')[0]) + '.csv')
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe 
    
    def pick_min_ema_diff(self,pair:str) -> float:
        #print(f"pair={pair}")
        c = pair.split('/')[0]
        if c=="BTC":
            return self.BTC.value
        elif c=="ETH":
            return self.ETH.value
        elif c=="ETH":
            return self.ETH.value
        elif c=="BCH":
            return self.BCH.value
        elif c=="BNB":
            return self.BNB.value
        elif c=="SOL":
            return self.SOL.value
        elif c=="XRP":
            return self.XRP.value
        elif c=="DOT":
            return self.DOT.value
        elif c=="AVAX":
            return self.AVAX.value
        elif c=="ADA":
            return self.ADA.value
        elif c=="ETC":
            return self.ETC.value
        elif c=="FTM":
            return self.FTM.value
        elif c=="FIL":
            return self.FIL.value
        elif c=="TRX":
            return self.TRX.value
        elif c=="AXS":
            return self.AXS.value
        elif c=="SAND":
            return self.SAND.value
        elif c=="NEAR":
            return self.NEAR.value
        elif c=="DOGE":
            return self.DOGE.value
        elif c=="MATIC":
            return self.MATIC.value
        elif c=="LTC":
            return self.LTC.value
        elif c=="LINK":
            return self.LINK.value
        elif c=="MANA":
            return self.MANA.value
        elif c=="ATOM":
            return self.ATOM.value
        elif c=="KNC":
            return self.KNC.value
        elif c=="XMR":
            return self.XMR.value
        elif c=="1000SHIB":
            return self.SHIB.value
        elif c=="EOS":
            return self.EOS.value
        elif c=="CRV":
            return self.CRV.value
        elif c=="ETH":
            return self.ETH.value
        elif c=="AAVE":
            return self.AAVE.value
        elif c=="WAVES":
            return self.WAVES.value
        elif c=="GALA":
            return self.GALA.value
        elif c=="RUNE":
            return self.RUNE.value
        elif c=="SUSHI":
            return self.SUSHI.value
        elif c=="XTZ":
            return self.XTZ.value
        elif c=="UNI":
            return self.UNI.value
        elif c=="ZIL":
            return self.ZIL.value
        elif c=="LRC":
            return self.LRC.value
        elif c=="XLM":
            return self.XLM.value
        elif c=="KAVA":
            return self.KAVA.value
        elif c=="EGLD":
            return self.EGLD.value
        elif c=="ALGO":
            return self.ALGO.value
        elif c=="ZEC":
            return self.ZEC.value
        elif c=="THETA":
            return self.THETA.value
        elif c=="YFI":
            return self.YFI.value
        elif c=="1INCH":
            return self.INCH.value
        elif c=="VET":
            return self.VET.value
        elif c=="MKR":
            return self.MKR.value
        elif c=="KSM":
            return self.KSM.value
        elif c=="GRT":
            return self.GRT.value
        elif c=="OMG":
            return self.OMG.value
        elif c=="COMP":
            return self.COMP.value
        elif c=="C98":
            return self.C98.value
        assert(False)