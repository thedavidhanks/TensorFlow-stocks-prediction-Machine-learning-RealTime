import os
import gc
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from keras.optimizers import Adam

from Utils.Utils_logger import print_to_file

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
tutorial_path = 'Tutorial/'
# get timestamp for log file
log_file = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
path_to_log = tutorial_path + 'outputs/' + log_file
print(os.getenv('TF_GPU_ALLOCATOR'))
physical_devices = tf.config.experimental.list_physical_devices('CPU')

from utilW3 import buy_sell_point, Feature_selection_create_json, get_info_model_evaluation
from utilW3.data import normalise_data, get_window_data, split_data, set_save_class_weight, get_load_class_weight, \
    remove_strong_correlations_columns, get_best_Y_correlations_columns
from features_W3_old.extract import uncorrelate_selection , get_tree_correlation
from Tutorial.CONFIG import *

# Disable Tensorflow useless warnings
from tensorflow import keras
# from tensorflow_addons.metrics import F1Score
from keras.callbacks import ModelCheckpoint, EarlyStopping
tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#CONFIGURATE
# INDicator_label_3W51560_ = "_3W51560_"
INDicator_label_3W5_ = "_3W5_"
# INDicator_label_3W515_ = "_3W515_"
# INDicator_label_3W560_ = "_3W560_"
#CONFIGURATE

# list_options_w =  [INDicator_label_3W5_, INDicator_label_3W51560_ ,INDicator_label_3W515_, INDicator_label_3W560_ ]
# stocks_list = [  "U", "DDOG","UPST", "RIVN", "SNOW", "LYFT",  "GTLB"] + [ "MDB", "HUBS", "TTD", "ASAN",    "APPS" , "DOCN", "SHOP", "RBLX", "NIO"]
stocks_list =  ["RIVN"]

def get_metrics(num_classes=3):
    return [
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        # F1Score(name='f1_macro', num_classes=num_classes, average='macro'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    ]

def print_to_file_with_location(string):
    print_to_file(string, path_to_log)

def main():

    print_to_file("Tutorial START\n", path_to_log, overwrite=True)
    INDI = INDicator_label_3W5_
    indicator_timeframe = INDI
    # for INDI in list_options_w:
    REF_MODEL = INDI+"_tut_A1_"
    for symbol in stocks_list : #stocks_list:

        features_X_col = []
        ind  = "5min"
        # [1]  Read ohlcv data
        file_rwa_alpha =  f'd_price/tutorial_pre_{symbol}_{ind}' + ".csv"
        #TIP to get more olhcv data can use 0_API_alphavantage_get_old_history.py (in this project) or alpaca API
        df_raw = pd.read_csv(file_rwa_alpha, parse_dates=['Date'], index_col='Date', sep='\t')
        for c in df_raw.columns:
            df_raw = df_raw.rename(columns={c: c.lower()})
        df_raw.index = pd.to_datetime(df_raw.index, errors="coerce")
        print_string = f"\n[1] Read data ohlcv data  path {file_rwa_alpha} Shape_raw: {df_raw.shape}"
        print_to_file(print_string, path_to_log)


        # [2] Get the 292 tech paterns
        print_string = f"\n[2] Calculating Technical indicators. stock: {symbol}"
        print_to_file(print_string, path_to_log)
        # df_bars = features_W3.ta.extract_features(df_raw)#TODO optimise, calculate the 830, It just need the +-90 from the .json
        #TODO review why using the 800 indicators reduces the effectiveness of the 270 indicators.
        from features_W3_old import extract_features_v5 as extract_features
        df_bars = extract_features(df_raw, extra_columns=False)  # IT WORKS   Tech indicators Count:  292
        # from features_W3_old.ta import extract_features
        # df_bars = extract_features(df_raw)# IT does NOT WORKS   Tech indicators Count:  800
        # TIP The technical indicators have been cleaned (LIST_TECH_REMOVE) , but it is possible that some of them may be GET data from the future, in order to make the calculation, these must be discarded for a correct functioning in real time.
        # TIP tip avoid using data older than 2 years, old data not useful data. So old data is bad data ??
        df_bars = df_bars.loc[df_bars.index > '2018-01-01 08:00:00']
        print_string = f"[2.1] Calculated Technical indicators. stock: {symbol} Tech indicator count: {df_bars.shape}"
        print_to_file(print_string, path_to_log)
        df_bars = df_bars.drop(columns=LIST_TECH_REMOVE+ ['Date','date', 'Date.1','date.1', 'Date1','date1'], errors='ignore') #TODO no generate LIST_TECH_REMOVE, it is bad tech paterns
        # see the different bettween two list set(df_bars.columns) ^ set(df_S_2.columns)

        # [3] Get the target_Y (ground true)
        print_string = f"\n[3] Calculate the target_Y (ground true), what is the target to detect?  the highest and lowest peaks stock: {symbol}"
        print_to_file(print_string, path_to_log)
        #TODO How do you know that the clustering Ground True (buy, sell, nothing) is similar and correct?
        # Is it necessary to cluster with more labels (buy_A, sell_A , nothing_A, buy_B, sell_B , nothing_B, buy_C, sell_C)? How do I know the number of tags needed ?
        # To solve this question you can use the "K-means Clustering" https://neptune.ai/blog/clustering-algorithms
        target_Y , df_l, _= buy_sell_point.get_buy_sell_point_HT_pp(df_bars.copy(), period=TIME_WINDOW_GT, rolling=TIME_WINDOW_GT * 2)
        # TIP the GT detection should be improved, it only detects the maximum and minimum points in time window,
        # in my opinion THE MOST IMPORTANT it should detect when the candles are "sharpened", to detect the entry of the "sharpening".   more info: https://github.com/Leci37/stocks-prediction-Machine-learning-RealTime-telegram#11-data-collection
        print_string = "[3.1]target_Y.shape: " + str(target_Y.shape)
        print_to_file(print_string, path_to_log)
        features_X_ALL = df_bars

    # Create correlation matrix https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
        print_string = f"[3.2] DEBUG All technical indicators Extracted Count: {len(features_X_ALL.columns)}" #," Names : ",",".join( features_X_ALL.columns ))
        print_to_file(print_string, path_to_log)

        # [4] Do selection best tech params
        print_string = "\n[4] Calculation of correlation strength. What are the +-100 best technical indicators and which are noise. Remove strong correlacted columns" 
        print_string += "\n[4.1]uncorrelate_selection path: " + f'{tutorial_path}data/columns_select/{symbol}_{REF_MODEL}_columns.pkl'
        print_to_file(print_string, path_to_log)
        
        features_X_ALL = remove_strong_correlations_columns(features_X_ALL,factor=0.85)
        features_X_col_corr = get_best_Y_correlations_columns(features_X_ALL.copy(), target_Y, num_columns=NUMS_FEATURES)
        features_X_col_tree = get_tree_correlation(features_X_ALL.copy(), target_Y, num_features=18)
        # TIP  all these convolutions have been tried to find out which has the strongest relationship in Tutorial/CONFIG.dict_correlations
        #TODO remove columns that have more than 0.9 correlation between them, as they are redundant. More info help: https://machinelearningmastery.com/basic-data-cleaning-for-machine-learning/
        columns_json_BOTH = list(set(list(features_X_col_corr) + list(features_X_col_tree)  ))
        Feature_selection_create_json.created_json_feature_selection(list_all_columns=columns_json_BOTH,path_json=f'{tutorial_path}data/columns_select/{symbol}_{REF_MODEL}_corr.json')
        features_X = features_X_ALL[columns_json_BOTH]
        print_string = "[4.2] Select the best technical patterns to train with features_W3 Extracted Count: " + str(len(features_X.columns)) + " Names : " + ",".join(features_X.columns)
        print_string += "\n[4.3] features_W3 index Dates: from " + str(features_X.index[0]) + " to " + str(features_X.index[-1]) + " Shape: " + str(features_X.shape)
        print_to_file(print_string, path_to_log)
        target_Y = target_Y[target_Y.index.isin(features_X.index)]
        features_X = features_X[features_X.index.isin(target_Y.index)]

        # [5] Normalizated
        print_to_file("\n[5] For correct training, for correct training the values must be normalised to between 0 and 1  ", path_to_log)
        #TIP   how to normalise this data is stored in file 'data/scalers/{symbol}_{REF_MODEL}_scalex.pkl' for use in the future realtime.
        print_string = "[5.1] Normalise_data path: " + f'{tutorial_path}data/scalers/{symbol}_{REF_MODEL}_scalex.pkl'
        print_to_file(print_string, path_to_log)
        features_X, scaleX = normalise_data(features_X, SPLIT_RATIO_X_Y, f'{tutorial_path}data/scalers/{symbol}_{REF_MODEL}_scalex.pkl')

        # [6] Get Window Data
        print_to_file("\n[6] Currently you have for each target_Y value a row of technical indicators, you add a 'window'"
            " to make the decision to predict whether the +-48 rows above will be taken (about 4 hours of previous indicators, splited in 5min).", path_to_log)
        X, y, index = get_window_data(features_X, target_Y, window_size=WINDOWS_SIZE)
        # TIP currently only 5min windows are used, but there is also a multi-window model that collects the 5min, 15min and 60min windows, ask the author
        # Example 'INFO created array x_array Created x_array.shape : (54000, 3, 48, 101) ,EX: 54000 rows, 3 clandle times (5min,15mim.60min) , windows size 48, 101tech parents' Maybe add a one minute candle as well. method get_window_data_3Window()
        print_string = "[6.1] get_window_data Shapes Y: " + str(y.shape) + " X: " + str(X.shape)
        print_to_file(print_string, path_to_log)

        # [7] SPLIT the Data
        print_to_file("\n[7] data split between training and validation ", path_to_log)
        X_train, y_train, X_test, y_test, index_train, index_test = split_data(X, y, index, split_ratio=SPLIT_RATIO_X_Y)
        print_string = "[7.1] Shapes: X_train: " + str(X_train.shape) + " y_train: " + str(y_train.shape) + " index_train: " + str(index_train.shape)
        print_to_file(print_string, path_to_log)
        # TIP  the tests I did by shuffling the data , destroy the result , I don't understand why there are time windows to save the relationship.

        # [8] Get  class_weight
        print_to_file("\n[8] Ground True data are unbalanced  Given that there is a lot of 'do nothing' 0, and very little 'do buy' 1 or 'do sell' 2,"
            " weight balancing is required, to give more importance to the minorities.  ", path_to_log)
        class_weight = compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
        dict_class_weight = dict(enumerate(class_weight))
        print_string = "[8.1] Class weight  path: " + f'{tutorial_path}data/class_weight/{symbol}_{REF_MODEL}_class_weight.pkl' + " Dict: " + str(dict_class_weight)
        print_to_file(print_string, path_to_log)
        set_save_class_weight(dict_class_weight, f'{tutorial_path}data/class_weight/{symbol}_{REF_MODEL}_class_weight.pkl')
        # TIP tip the balancing to check the unbalance is very necessary, I would also like to add that it penalizes more in the training the false positives,
        # there are other methods for balancing as SMOTE, they have been tested without improvement, in my opinion, I don't like them, they destroy the data, if it is applied it must be very delicate.

        # [9] Create a model TF
        print_to_file("\n[9] Creation of the TF model architecture. must respect the input_shape and output_shape and the 'softmax' ,"
            " from there EXPRIMENT combinations ", path_to_log)
        input_shape = X_train.shape[1:]
        output_shape = 1 if len(y_train.shape) == 1 else y_train.shape[1]
        print_string = "[9.1] arrays to use the TF model  , input_shape: " + str(input_shape) + " output_shape: " + str(output_shape)
        print_to_file(print_string, path_to_log)
        
        # TIP to really understand this point, visit and play with neuron web https://playground.tensorflow.org/#activation=relu&batchSize=15&dataset=spiral&regDataset=reg-plane&learningRate=0.001&regularizationRate=0&noise=50&networkShape=8,8,4,2&seed=0.94804&showTestData=false&discretize=false&percTrainData=50&x=false&y=false&xTimesY=true&xSquared=false&ySquared=false&cosX=false&sinX=true&cosY=false&sinY=true&collectStats=false&problem=classification&initZero=false&hideText=false
        #TODO TIP , train multiples models types and get the best  , The creation of the TF architecture can be done in a thousand ways, here is just one example, try the different ways:
            # Models to test on https://github.com/Leci37/stocks-prediction-Machine-learning-RealTime-telegram#combine-the-power-of-the-17-models
            # Review of all forms of time series prediction: lstm,gru,cnn and rnn https://github.com/Leci37/stocks-prediction-Machine-learning-RealTime-telegram#review-of-all-forms-of-time-series-prediction-lstmgrucnn-and-rnn
            # Recommended reading LSTM plus stock price FAIL https://github.com/Leci37/stocks-prediction-Machine-learning-RealTime-telegram#recommended-reading-lstm-plus-stock-price-fail
        #TODO TIP  test massively  with TUNNERS https://github.com/Leci37/stocks-prediction-Machine-learning-RealTime-telegram#better-use-tuners
            # MODELNote: The use of class_weights changes the range of the loss. This may affect the stability of training depending on the optimiser.ptimizers whose step size depends on the magnitude of the gradient, such as tf.keras.optimizers.SGD , may fail.
            # The optimiser used here, tf.keras.optimisizers.Adam , is not affected by the change in scale. Note also that due to weighting, the total losses are not comparable between the two models.
        ACTIVATION_2_ReLU = tf.keras.layers.LeakyReLU(alpha=0.02)
        model = keras.Sequential([
            keras.layers.Dense(WINDOWS_SIZE*3, activation=ACTIVATION_2_ReLU, input_shape = input_shape ),
            keras.layers.Dense(72, activation=ACTIVATION_2_ReLU),
            keras.layers.Dropout(0.32),
            keras.layers.Dense(64, activation=ACTIVATION_2_ReLU),
            keras.layers.Dense(64, activation=ACTIVATION_2_ReLU),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(12, activation=ACTIVATION_2_ReLU),
            keras.layers.Dropout(0.1),
            keras.layers.Flatten(),
            keras.layers.Dense(output_shape, activation='softmax')#, bias_initializer = output_bias_k),
        ])
        # TIP inspired by the kaggle winner  https://www.kaggle.com/code/viktortaran/ps-s-3-e-6#%E2%9C%855.12.-Keras
        loss = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
        reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1,min_delta=0.001, min_lr=0.000001, mode='min')
        early_stopping = keras.callbacks.EarlyStopping(patience=11, monitor="val_loss", mode='min', verbose=1,restore_best_weights=True)
        callbacks_save = ModelCheckpoint(filepath=f'{tutorial_path}outputs/{symbol}_{REF_MODEL}_buysell.h5',monitor='val_loss', verbose=1, save_best_only=True)
        tf.keras.utils.plot_model(model, to_file=f'{tutorial_path}outputs/plots/{symbol}_{REF_MODEL}_buysell.png')
        print_string = "[9.3] print diagram of the TF model Path: " + f'{tutorial_path}outputs/plots/{symbol}_{REF_MODEL}_buysell.png'
        print_to_file(print_string, path_to_log)
        
        # [10] Trian the model
        print_string = "\n[10] Start training TF model " + bcolors.OKBLUE + f' {tutorial_path}outputs/{symbol}_{REF_MODEL}_buysell.h5 '+ bcolors.ENDC
        print_string += "[10.1] To adjust and measure the model, callbacks is required, the callbacks will stop before the overfit. "
        print_to_file(print_string, path_to_log)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005 ), loss=loss, metrics=get_metrics(num_classes=len(target_Y.columns)))
        # TIP f1 is the BEST metric to unbalance cases
        history = model.fit(x=X_train, y=y_train, batch_size=128, epochs=280, class_weight=dict_class_weight, verbose=1,
                            validation_split=0.33, callbacks=[reduce_lr_loss, early_stopping, callbacks_save])
        # TIP on a monitor= you can stop training when you get the best precision    recall  f1-score , but the one that works best is val_loss.
        print_string = "[10.2] Model initial_weights saved :"+ f'{tutorial_path}data/initial_weights/{symbol}_{REF_MODEL}_initial_weights.tf'
        print_to_file(print_string, path_to_log)
        model.save_weights( f'{tutorial_path}data/initial_weights/{symbol}_{REF_MODEL}_initial_weights.tf')

        # [11] predict and eval
        print_string = "\n[11] Do a predict and eval. Load path: " + f'{tutorial_path}outputs/{symbol}_{REF_MODEL}_buysell.h5'
        print_string += "[11.1] array to predit x_test.shape: " + str(X_test.shape)
        print_to_file(print_string, path_to_log)
        gc.collect()  # IMPORTANT clean the ram
        # load_class_weight = get_load_class_weight(f'data/class_weight/{symbol}_{REF_MODEL}_class_weight.pkl')
        resampled_model_2 = keras.models.load_model(f'{tutorial_path}outputs/{symbol}_{REF_MODEL}_buysell.h5')
        y_pred_1 = resampled_model_2.predict(X_test, batch_size=6)
        y_pred_1 = np.argmax(np.squeeze(y_pred_1), axis=1)
        y_test_a = np.argmax(y_test, axis=1)
        # TIP argmax is used to detect which of the 3 preconditions ('do nothing', 'do buy' or 'do sell') is the best,
        # but it can be adjusted to be more accurate and discard half results. ex: only validating the ones above 0.65
        print_string = "[11.2] read the evaluation: " + str(X_test.shape)
        print_to_file(print_string, path_to_log)
        get_info_model_evaluation.get_info_model_evaluation(y_pred_1, y_test_a, symbol=symbol, indicator_timeframe= REF_MODEL, pwd=tutorial_path, print_fn=print_to_file_with_location)
        # TIP The ideal would be to increase the precision (in 1 buy and 2 sell) above 50%, but without sacrificing the recall f1-score values.
        # TIP check the y_pred y_test table , to know what is the proportion of failures and successes in accurate prediction.
        # TODO improve the evaluation system of the model, not answering the question "how many times did I get it right", but the question "how much money did I win or lose?
        # TODO for more ideas on POSSIBLE IMPROVEMNETS, check out the section  https://github.com/Leci37/stocks-prediction-Machine-learning-RealTime-telegram#possible-improvements
        gc.collect()
        print_to_file("\n", path_to_log)
    print_to_file("END", path_to_log)



