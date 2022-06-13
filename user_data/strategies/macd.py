# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

#from helper import helper

# --- Do not remove these libs ---
from email.policy import default
from unicodedata import decimal
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
import math
import datetime
from typing import Optional
from freqtrade.persistence import Trade


from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, RealParameter,
                                IStrategy, IntParameter, informative)

from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real  # noqa


# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class MACD(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1h'
    big_timeframe = '4h'
    can_short: bool = True
    minimal_roi = {
        "0": 100
    }
    position_adjustment_enable = True

    distance_price = 0.02 #distance when calculate support,resistance
    r2r = DecimalParameter(1.5,2.0,decimals=3,default=1.819,space="buy")
    position_lost_percentage = DecimalParameter(0.5,1,decimals=3,default=0.935,space="buy")
    risk_level = DecimalParameter(0.02,0.10,decimals=3,default=0.09,space="buy")
    
    n1 = 3 #count candle to find support,resistance
    n2 = 3
    
    stoploss = -1
    trailing_stop = False
    process_only_new_candles = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    startup_candle_count: int = 800
    percentage_stop = -1 # record when order, diff between open price and stoploss

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
            f'ema200_{big_timeframe}': {'color':'blue'},
            f'bb_upper_{big_timeframe}': {'color':'white'},
            f'bb_middle_{big_timeframe}': {'color': 'white'},
            f'bb_lower_{big_timeframe}': {'color': 'white'}
        },
        'subplots': {
            "MACD": {
                'macd_hist': {'type': 'bar', 'plotly': {'opacity': 0.9}, 'color':'green'}
            },
        }
    }

    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            entry_tag: Optional[str], side: str, **kwargs) -> float:
        # position size = account size x account risk / invalidation point
        position_size = (max_stake*self.risk_level.value)/self.position_lost_percentage.value
        if position_size < min_stake:
            print(f"WARNING: enter position {pair} with risk level = {(min_stake*self.position_lost_percentage.value)/max_stake}")
            return min_stake
        return position_size


    @informative(big_timeframe)
    def populate_indicators_big_timeframe(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema200'] = ta.EMA(dataframe['close'], timeperiod = 200)
        n1 = self.n1
        n2 = self.n2
        support = []
        resistance = []

        touch_support = [False] * len(dataframe)
        touch_resistance = [False] * len(dataframe)
        is_support = [False] * len(dataframe)
        is_resistance = [False] * len(dataframe)

        for i in range(0,len(dataframe)):
            for j in range(len(support)):
                if dataframe.iloc[i]['low'] <= support[j] and dataframe.iloc[i]['close'] > support[j]:
                    touch_support[i] = True
                    break
            for j in range(len(resistance)):
                if dataframe.iloc[i]['high'] >= resistance[j] and dataframe.iloc[i]['close'] < resistance[j]:
                    touch_resistance[i] = True
                    break
            if self.is_support(dataframe,i,n1,n2):
                support.append(dataframe.iloc[i]['low'])
                is_support[i] = True
            if self.is_resistance(dataframe,i,n1,n2):
                resistance.append(dataframe.iloc[i]['high'])
                is_resistance[i] = True
        
        dataframe['is_support'] = is_support
        dataframe['is_resistance'] = is_resistance
        dataframe['touch_support'] = touch_support
        dataframe['touch_resistance'] = touch_resistance

        bb_upper, bb_middle, bb_lower = ta.BBANDS(dataframe['close'], 20, 2.0, 2.0)
        dataframe['bb_upper'] = bb_upper
        dataframe['bb_middle'] = bb_middle
        dataframe['bb_lower'] = bb_lower

        #dataframe = self.plot_support_resistance(dataframe)
        return dataframe

    
    def leverage(self, pair: str, current_time: 'datetime', current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        l = max(math.floor(self.position_lost_percentage.value/self.percentage_stop),1)
        #print(f"percent_stop = {self.percentage_stop}, leverage={l}")
        return l


    def distance_enough(self,p1,p2):
        p1 = float(p1)
        p2 = float(p2)
        return (abs(p1-p2)/min(p1,p2))*100 >= self.distance_price

  

    def is_support(self,dataframe:DataFrame,i:int,n1:int,n2:int) -> bool:
        for j in range(i-n1,i):
            if j<0 or j+1>=len(dataframe):
                return False
            if not self.distance_enough(dataframe.iloc[j]['low'],dataframe.iloc[j+1]['low']):
                return False
            if not(dataframe.iloc[j]['low'] > dataframe.iloc[j+1]['low']):
                return False
        for j in range(i,i+n2+1):
            if j<0 or j+1>=len(dataframe):
                return False
            if not self.distance_enough(dataframe.iloc[j]['low'],dataframe.iloc[j+1]['low']):
                return False
            if not(dataframe.iloc[j]['low'] < dataframe.iloc[j+1]['low']):
                return False
        return True
    

    def is_resistance(self,dataframe:DataFrame,i:int,n1:int,n2:int) -> bool:
        for j in range(i-n1,i):
            if j<0 or j+1>=len(dataframe):
                return False
            if not self.distance_enough(dataframe.iloc[j]['high'],dataframe.iloc[j+1]['high']):
                return False
            if not(dataframe.iloc[j]['high'] < dataframe.iloc[j+1]['high']):
                return False
        for j in range(i,i+n2+1):
            if j<0 or j+1>=len(dataframe):
                return False
            if not self.distance_enough(dataframe.iloc[j]['high'],dataframe.iloc[j+1]['high']):
                return False
            if not(dataframe.iloc[j]['high'] > dataframe.iloc[j+1]['high']):
                return False
        return True
    
    def plot_support_resistance(self,dataframe:DataFrame):
        main_plot = self.plot_config['main_plot']
        for i in range(len(dataframe)):
            if dataframe.iloc[i]['is_support']:
                dataframe[f'support_{i}'] = [dataframe.iloc[i]['low']] * len(dataframe)
                main_plot[f'support_{i}_{self.big_timeframe}'] = {'color':'orange'}
            if dataframe.iloc[i]['is_resistance']:
                dataframe[f'resistance_{i}'] = [dataframe.iloc[i]['high']] * len(dataframe)
                main_plot[f'resistance_{i}_{self.big_timeframe}'] = {'color':'purple'}
        return dataframe
        

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['macd_hist'] = ta.MACD(dataframe['close'], timeperiod=14)[2]
        #helper.save_dataframe(dataframe)
        return dataframe

    
    def touch_support(self,dataframe:DataFrame,i:int,interesting_back_candle:int):
        for j in range(i-interesting_back_candle,i):
            if dataframe.iloc[j][f'touch_support_{self.big_timeframe}']:
                return (True,dataframe.iloc[j][f'low_{self.big_timeframe}'])
        return (False,None)
    
    def touch_resistance(self,dataframe:DataFrame,i:int,interesting_back_candle:int):
        for j in range(i-interesting_back_candle,i):
            if dataframe.iloc[j][f'touch_resistance_{self.big_timeframe}']:
                return (True,dataframe.iloc[j][f'high_{self.big_timeframe}'])
        return (False,None)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        distance_stop_loss = 0.02

        interesting_back_candle = 10
        enter_long = [False] * len(dataframe)
        enter_short = [False] * len(dataframe)
        exit_long = [False] * len(dataframe)
        exit_short = [False] * len(dataframe)
        entering = None
        price_stop_loss = -1
        exit_price = -1

        for i in range(interesting_back_candle,len(dataframe)):
            touch_support,s = self.touch_support(dataframe,i,interesting_back_candle)
            current = dataframe.iloc[i]
            #enter long
            if entering==None and current['close'] > current[f'ema200_{self.big_timeframe}'] and current['low'] > current[f'ema200_{self.big_timeframe}'] and current['macd_hist']>0 and dataframe.iloc[i-1]['macd_hist']<0 and touch_support:
                enter_long[i] = True
                entering = 'long'
                price_stop_loss = s - (s*distance_stop_loss)
                self.percentage_stop = (current['close'] - price_stop_loss) / price_stop_loss
                exit_price = (current['close'] - price_stop_loss)*self.r2r.value + current['close']
                #print(f"enter long position {current['close']} with exit_price={exit_price} stoploss = {price_stop_loss} percentage stop = {self.percentage_stop}")
            #take profit long
            if entering=='long' and current['close'] >= exit_price:
                entering = None
                exit_long[i] = True
            #stop loss long
            if entering=='long' and current['close']<=price_stop_loss:
                entering = None
                exit_long[i] = True
            
            touch_resistance,r = self.touch_resistance(dataframe,i,interesting_back_candle)
            #enter short
            # if touch_resistance:
            #     print(f"date={current['date']} {entering} {current['close']} {current[f'ema200_{self.big_timeframe}']} {current['high']} {current[f'ema200_{self.big_timeframe}']} {current['macd_hist']}  {dataframe.iloc[i-1]['macd_hist']} ")
            if entering==None and current['close'] < current[f'ema200_{self.big_timeframe}'] and current['high'] < current[f'ema200_{self.big_timeframe}'] and current['macd_hist']<0 and dataframe.iloc[i-1]['macd_hist']>0 and touch_resistance:
                enter_short[i] = True
                entering = 'short'
                price_stop_loss = r + (r*distance_stop_loss)
                self.percentage_stop = (price_stop_loss - current['close']) / current['close']
                exit_price = current['close'] - (price_stop_loss-current['close'])*self.r2r.value
            #take profit short
            if entering=='short' and current['close']<=exit_price:
                entering = None
                exit_short[i] = True
            #stop loss short
            if entering=='short' and current['close']>=price_stop_loss:
                entering = None
                exit_short[i] = True

        dataframe['enter_long'] = enter_long
        dataframe['enter_short'] = enter_short
        dataframe['exit_long'] = exit_long
        dataframe['exit_short'] = exit_short
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe 
    