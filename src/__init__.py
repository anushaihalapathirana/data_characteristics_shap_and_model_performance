import sys
import numpy as np
np.random.seed(25)
import tensorflow as tf
import random
import os
import umap

tf.random.set_seed(25)
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedShuffleSplit

# disable warnings
import warnings
import os
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.constants import TRAIN_SIZE_ARR, ALGO_LIST, DATA_FILE_PATH, SHAP_IMBALANCE_BACKGROUND_DATA_BOX_PLOT, \
        SHAP_BALANCE_BACKGROUND_DATA_UMAP, SHAP_BALANCE_BACKGROUND_DATA_BOX_PLOT, SHAP_IMBALANCE_BACKGROUND_DATA_UMAP
from src.helpers import imputation, print_data_shapes, get_shap_values, get_test_data, get_perc_of_train_data,\
        plot_global, plot_dependency, scoring_function, overall, plot_shap_box_plots, plot_shap_bar_plots, \
        plot_combined_graphs

class BaseTrainer:
    
    def __init__(self):
         
        self.script_directory = os.path.dirname(os.path.abspath(__file__))
        self.file_path_data = os.path.join(self.script_directory, DATA_FILE_PATH)
        self.fig_balance_box_plot_save_path = os.path.join(self.script_directory, SHAP_BALANCE_BACKGROUND_DATA_BOX_PLOT)
        self.fig_imbalance_box_plot_save_path = os.path.join(self.script_directory, SHAP_IMBALANCE_BACKGROUND_DATA_BOX_PLOT)
        self.fig_balance_umap_save_path = os.path.join(self.script_directory, SHAP_BALANCE_BACKGROUND_DATA_UMAP)
        self.fig_imbalance_umap_save_path = os.path.join(self.script_directory, SHAP_IMBALANCE_BACKGROUND_DATA_UMAP)
        

        self.islda = self.isxgb = self.isnn = False
        if 'xgb' in ALGO_LIST:
            self.isxgb = True
        if 'lda' in ALGO_LIST:
            self.islda = True
        if 'nn' in ALGO_LIST:
            self.isnn = True
            
        self.isfi = True
        self.isdependency = False
        self.isglobal = False
        self.islocal = False
        self.iscv = True

        self.case_no = 0
        self.size = 1200
        self.splits = 10
        self.response_col = 'c4'
        
        self.exclude = ["D10A","G02B","H05A","J04B","J06B","L01A","M01C","N06B","N07X",
           'ï»¿ID','c4_p1','c4_p2','c4_p3','c5','c5_p1','c5_p2','c5_p3','c5_p4',
           "pelkka_ins0","n_hba1c","n_measures",'aika_diag','t2d_vuosi','obese',
           'metformin','A10A']
        
        self.Clinical_predictors  = ['sp','ika_14','aika_diag_y','gluk','bmi','kol','ldl','p_krea','n_of_dis',
                        'microvasc_enn','macrovasc_enn','obese','dyslipimet','hyperten','chd','atr_fib',
                        'heart_fail','pad','stroke','kidney_dis','neuropathy','blindness','T2D_ketoasid',
                        'T2D_kidneydis','T2D_opthalmic','T2D_neurolog','T2D_bloodcirc','T2D_misanc',
                        'T2D_severalcomp','T2D_unspecif','T2D_nocomp','cancer','asthma','gout',
                        'glaucoma','depression','dementia','mental_dis','copd','rheuma','osteoporosis',
                        'neuromusc_dis','liver_dis','discordant_dis','dg101','dg201','dg301','dg303','dg401',
                        'dg402','dg403','dg404','dg405','dg406','dg501','dg502','dg503','dg601','dg602','dg701',
                        'dg702','dg703','dg704','dg705','dg801','dg802','dg901','dg902','dg903','dg904','dg111',
                        'dg121','dg122','dg123','dg124','dg131','dg132','dg133','concordant_dis','obesity']

        self.Treatment_related_factors = ['pelkka_met0','met_oad0','pelkka_ins0','ins_oad0','pelkka_muu0','ei_mitaan0',
                             'sum_diab_drugs','puhelut_hoitaja','puhelut_laakari','kaynnit_hoitaja',
                             'kaynnit_laakari','sum_contacts','hba1c_mitt','ldl_mitt','bmi_mitt','n_hba1c',
                             'n_measures','A01A','A10A','A02B','A03F','A04A','A05A','A06A','A07A','A07E','A09A',
                             'A10B','A11C','A12A','A10B',
                             'B01A','B02A','B03B','B03X','C01A','C01B','C01C','C01D','C02A','C02C','C02K','C03A',
                             'C03B','C03C','C03D','C03E','C04A','C05A','C07A','C07B','C07F','C08C','C08D','C09A',
                             'C09B','C09C','C09D','C10A','D01A','D01B','D02A','D05A','D05B','D06B','D07A','D07B',
                             'D07C','D07X','D09A','D10A','G01A','G02B','G03B','G03C','G03D','G03F','G03H','G03X',
                             'G04B','G04C','H01B','H01C','H02A','H02B','H03A','H03B','H04A','H05A','H05B','J01A',
                             'J01C','J01D','J01E','J01F','J01M','J01X','J02A','J04A','J04B','J05A','J06B','L01A',
                             'L01B','L01C','L01X','L02A','L02B','L03A','L04A','M01A','M01C','M02A','M03B','M04A',
                             'M05B','N02A','N02B','N02C','N03A','N04A','N04B','N05A','N05B','N05C','N06A','N06B',
                             'N06C','N06D','N07A','N07B','N07C','N07X','P01A','P01B','R01A','R03A','R03B','R03C',
                             'R03D','R06A','S01A','S01B','S01C','S01E','S01G','S01X','V03A','laake_lkm','insul']

        self.ratio_val = ['ko_perus_r', 'ko_koul_r', 'ko_yliop_r', 'ko_ammat_r', 'ko_al_kork_r', 
             'ko_yl_kork_r', 'hr_pi_tul_r', 'hr_ke_tul_r', 'hr_hy_tul_r', 
             'tr_pi_tul_r', 'tr_ke_tul_r', 'tr_hy_tul_r', 'tp_alku_a_r', 
             'tp_jalo_bf_r', 'tp_palv_gu_r', 'tp_a_maat_r', 'tp_b_kaiv_r', 
             'tp_c_teol_r', 'tp_d_ener_r', 'tp_e_vesi_r', 'tp_f_rake_r', 
             'tp_g_kaup_r', 'tp_h_kulj_r', 'tp_i_majo_r', 'tp_j_info_r', 
             'tp_k_raho_r', 'tp_l_kiin_r', 'tp_m_erik_r', 'tp_n_hall_r', 
             'tp_o_julk_r', 'tp_p_koul_r', 'tp_q_terv_r', 'tp_r_taid_r', 
             'tp_s_muup_r', 'tp_t_koti_r', 'tp_u_kans_r', 'tp_x_tunt_r', 
             'pt_tyovy_r', 'pt_tyoll_r', 'pt_tyott_r', 'pt_tyovu_r', 'pt_0_14_r', 
             'pt_opisk_r', 'pt_elakel_r', 'pt_muut_r', 'te_nuor_r', 'te_eil_np_r', 
             'te_laps_r', 'te_plap_r', 'te_aklap_r', 'te_klap_r', 'te_teini_r', 
             'te_aik_r', 'te_elak_r', 'te_omis_as_r', 'te_vuok_as_r', 'te_muu_as_r', 
             'ra_pt_as_r', 'ra_kt_as_r']
        
        self. original_val = ['ra_raky', 'te_taly', 'pt_vakiy', 'tp_tyopy', 'tr_kuty', 'hr_tuy', 'ko_ika18y']
        
        self.background_factors = ['hba1c_2012','hba1c_2013']

        self.cols = []
        self.cols.append(['T2D duration in years', 'Other cardiac diseases', 'Fasting plasma glucose', 'HbA1c 2 years before',
                    'HbA1c 1 year before']) #Other diseases of heart and pulmonary circulation #Other cardiac disease
        self.cols.append(['T2D several complications', 'T2D duration in years', 'Fasting plasma glucose', 
                    'HbA1c 2 years before', 'HbA1c 1 year before', 'Heart failure', 'Insulin + OAD', 'Insulin only',
                    'Metformin', 'Number of antidiabetic drugs'])
        self.cols.append(['Estrogens', 'Adrenergic inhalations', 'T2D duration in years', 'Diseases of female sex organ',
                    'Sleep disorders', 'Discordant diseases', 'Fasting plasma glucose', 'HbA1c 2 years before',
                    'HbA1c 1 year before', 'Insulin + OAD', 'Insulin only', 'Metformin', 'Number of antidiabetic drugs', 
                    'Ratio of households with no childrens', 'Ratio of households in lowest income groups'])
    
        self.best_predictors = {}
        self.columns= {}
        ## ==== Clinical ==== ##
        self.best_predictors[0] = ['aika_diag_y', 'dg703', 'gluk', 'hba1c_2012', 'hba1c_2013','c4']

        self.sv = {}
        self.fi = {}
        
        self.results_var = {}
        self.results = {}
        self.threshold = 0.5
        self.sample = [1337,3,2,0,89]
    
    def feature_selection(self):
        """method to combine features. Not used in this experiment, since we only focus on clinical predictors

        Returns:
            include: combined feature list
        """
        SES_factors = np.append(self.ratio_val, self.original_val)
        include= {}
        ## ==== For feature selection ==== ##
        include[0] = np.append(self.Clinical_predictors,'c4')
        # include[0] = np.append(include[0],'viim_hba1c_bl')
        include[0] = np.append(include[0], self.background_factors)

        include[1] = np.append(include[0], self.Treatment_related_factors)
        include[2] = np.append(include[1], SES_factors)
        return include

    def preprocess_data(self, filename, response_col, exclude_cols, include_features, include, which, perc, impute):
        """methods to preprocess data

        Args:
            filename : datafile name
            response_col : response column name
            exclude_cols : list of columns to exclude
            include_features : list of features to include
            include 
            which :  0 for 1 vs 2, 1 for 3 vs rest, 2 for c3 (1+2+3 vs 4), else/3 for exact import
            perc : if a feature is missing 'perc' or more than 'perc' percent of data --> drop the features
            impute : data imputaion or not.  do imputation = 1 

        Returns:
            X, y : preprocessed data
        """
        # Load the dataset as a pandas DataFrame
        data =  pd.read_csv(filename ,sep = ';',decimal = ',', encoding = 'unicode_escape', engine ='python')

        # Include only mentioned features
        if include_features == 1:
            data = data[include]
            
        # Exclude columns that need to be excluded
        for col in data.columns:
            if col in self.exclude:
                del data[col]
        # Convert all "object" columns to numeric (according to data description !!)
        to_convert = data.select_dtypes('object').columns
        data.loc[:,to_convert] = data[to_convert].apply(pd.to_numeric, downcast='float', errors='coerce')

        if which == 0:
            # Eliminate class 3
            data = data[data[response_col] < 3]
        elif which == 1:
            # Merge class 1 and 2
            for i in range(0,len(data)):
                if data[response_col][i] == 1:
                    data.loc[i, response_col] = 2
        elif which == 2:
            # Merge class 1, 2 and 3
            for i in range(0,len(data)):
                if data[response_col][i] == 1 or data[response_col][i] == 2:
                    data.loc[i, response_col] = 3
        else:
            data = data
            
        # Delete columns with 'perc' percent or more values missing
        min_count = int(((100-perc)/100)*data.shape[0] + 1)
        data = data.dropna(axis = 1, thresh = min_count, subset = None, inplace = False)
        
        # Impute or not
        if impute == 0:
            data = data.dropna()
        
        # Split into input (X) and output (y) variables
        X = data[data.columns.difference([response_col])]#, sort = False)]
        if which == 0:
            y = data[response_col] -2  # made the output from 0 to 1 
        elif which == 1:
            y = data[response_col] -2 # made the output from 0 to 1 (0 for class three and 1 for 1 and 2)
        elif which == 2:
            y = data[response_col] -3
        else:
            y = data[response_col] -1
        return X, y
    

    def get_preprocess_data(self):
        """method to get preprocessed data
        """
        which = 1 # 0 for 1 vs 2, 1 for 3 vs rest, 2 for c3 (1+2+3 vs 4), else/3 for exact import
        perc = 40 # if a feature is missing 'perc' or more than 'perc' percent of data --> drop the features
        include_features = 1 # use only include features = 1 (only usefull for all, All predictors model = 0)
        impute = 1 # do imputation = 1 
        include = self.feature_selection()
        
        data = {}
        for i in range(1):
            X, Y = self.preprocess_data(self.file_path_data, "c4", self.exclude,\
                include_features, self.best_predictors[i],which,perc,impute)
            print('shape of the data ', X.shape)
            
            data[i] = [X,Y]
        data[0][0].columns = self.cols[0]
        return data, X, Y
    
    
    def shapley_feature_ranking(self, model, X_train, X_val, size, cv_loop, name,\
                                is_check_additivity_false=False, featurenames = []):
        """method to get shap values, NN handled differently

        Args:
            model 
            X_train 
            X_val 
            size 
            cv_loop 
            name 
            featurenames (list, optional): feature list. Defaults to [].

        Returns:
            sv : shap values 
        """
        isnn = False
        if name=='nn':
            isnn = True
        shap_values = get_shap_values(model,  X_train, X_val, is_check_additivity_false, isnn, featurenames)
        shap_values = shap_values.values
        if self.isfi:
            self.sv[(size, cv_loop, name)] = np.mean(np.abs(shap_values), axis=0)
        else:
            self.sv[(size, cv_loop, name)] = np.mean(np.abs(shap_values), axis=0)
        return self.sv 

    def get_feat_importance(self, model, size, cv_loop, name):
        """method to calculate feature importance

        Args:
            model 
            size 
            cv_loop 
            name 

        Returns:
            fi : feature importance
        """
        if name == 'lda':
            feature_imp = model.coef_[0]
        elif name == 'nn':
            # Get the feature
            feature_imp = model.importances_mean
        else:
            feature_imp = model.feature_importances_
        self.fi[(size, cv_loop, name)] = feature_imp
        return self.fi
    
    def train_test(self, data, m, sample, method, train_size = 25):
        """methods to cross valudation, train models and claculate model performance, shap values and 
        feature importance for different data characteristics.

        Args:
            data 
            m
            sample
            method 
            train_size (int, optional): size of the train data. Defaults to 25.
        """
        train_data, test_data = get_test_data(data, n_samples = 500)
    #     train_data = perc_of_original_train_data(train_data, use_perc=100)
        # use that 10% data to generate new data
        X, Y = get_perc_of_train_data(train_data, train_size, method)

        X_test = test_data[test_data.columns.difference([self.response_col])]
        Y_test = test_data[self.response_col]
        
        X_, Y_, X_test, Y_test = imputation(X, Y, X_test, Y_test, self.response_col)
        
        # sampler = RandomUnderSampler(sampling_strategy='majority')
        #  X_, Y_ = sampler.fit_resample(X_, Y_)
        myfeatures = X_test.columns

        scaler = StandardScaler()
        X_ = pd.DataFrame(scaler.fit_transform(X_))
        X_test = pd.DataFrame(scaler.transform(X_test))
            
        X_.columns = myfeatures
        X_test.columns = myfeatures

        # kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=0) #2
        kfold = StratifiedShuffleSplit(n_splits=self.splits, test_size=0.25, random_state=0) 
        results = {}
        results_var = {}
        myfeatures = []
        i = 0
        j = 0
        acc_arr_LDA = []
        acc_arr_XGB = []
        acc_arr_NN = []
        
        f1_arr_LDA = []
        f1_arr_XGB = []
        f1_arr_NN = []
        
        auc_arr_LDA = []
        auc_arr_XGB = []
        auc_arr_NN = []
        
        sv_list = []
        fi_list = []
        
        print('train set shape ', X_.shape)
        print('test set shape ', X_test.shape)
        
        for train, test in kfold.split(X, Y):      
            X_train, Y_train, X_val, Y_val = imputation(X.iloc[train], Y.iloc[train], X.iloc[test],\
                                                        Y.iloc[test], response_variable=self.response_col)
            myfeatures = X_test.columns

            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train))
            X_val = pd.DataFrame(scaler.transform(X_val))

            X_train.columns = myfeatures
            X_val.columns = myfeatures
            
            print("******************************")
            print('train set shape ', pd.DataFrame(Y_train).value_counts())
            print("*+++++++++++++++++++++++++++++*")
            print('Validation set shape ', pd.DataFrame(Y_val).value_counts())
            print("******************************")
        
            j += 1
            ## ==== LDA ==== ##
            if self.islda:
                clf = LinearDiscriminantAnalysis(solver = 'eigen', shrinkage= 'auto', tol = 1e-2)
                clf = clf.fit(X_train, Y_train)
                pred = clf.predict(X_val)
                predictions = clf.predict_proba(X_val)
                probs = predictions[:,1]
                acc_arr_LDA.append(balanced_accuracy_score(Y_val, clf.predict(X_val)))
                f1_arr_LDA.append(f1_score(Y_val, clf.predict(X_val),average='macro'))
                auc_arr_LDA.append(roc_auc_score(Y_val, clf.predict(X_val)))
        
        
                if self.iscv:
                    sv_list = self.shapley_feature_ranking(clf, X_train, X_val, train_size, j, 'lda', False)
                    fi_list = self.get_feat_importance(clf, train_size, j, 'lda')
                
                if self.isglobal and self.iscv:
                    plot_global(clf, X_train, X_val, islocal= self.islocal)
                
                if self.isdependency and self.iscv:
                    plot_dependency(clf, X_train, X_val)
            
            if self.isxgb:

                xgb = XGBClassifier(
                                eta = 0.05,#eta between(0.01-0.2)
                                max_depth = 6, #values between(3-10)
                                colsample_bytree = 0.7,#values between(0.5-1)
                                tree_method = "auto",
                                booster='gbtree',
                                eval_metric='mlogloss',
                )
                xgb = xgb.fit(X_train, Y_train)
                pred = xgb.predict(X_val)
                predictions = xgb.predict_proba(X_val)
                probs = predictions[:,1]
                acc_arr_XGB.append(balanced_accuracy_score(Y_val, xgb.predict(X_val)))
                f1_arr_XGB.append(f1_score(Y_val, xgb.predict(X_val),average='macro'))
                auc_arr_XGB.append(roc_auc_score(Y_val, xgb.predict(X_val)))
                
                            
                if self.iscv:
                    # get feature importance
                    sv_list = self.shapley_feature_ranking(xgb, X_train, X_val, train_size, j, 'xgb', True)
                    fi_list = self.get_feat_importance(xgb, train_size, j, 'xgb')
                    
                if self.isglobal and self.iscv:
                    plot_global(xgb, X_train, X_val, is_check_additivity_false=True)
                
                if self.isdependency and self.iscv:
                    plot_dependency(xgb, X_train, X_val, isxgb = True)


            
            if self.isnn:
            # ==== NN ==== ##
                np.random.seed(42)
                tf.random.set_seed(42)
                random.seed(42)
        
                shape = (len(X_train.T),)
                if shape[0] == 5:
                    np.random.seed(42)
                    tf.random.set_seed(42)
                    os.environ['PYTHONHASHSEED'] = '0'
                    # Set random seed for GPU
                    if tf.test.is_gpu_available():
                        tf.config.experimental.set_seed(42)

                    # Disable non-deterministic behavior in TensorFlow
                    os.environ['TF_DETERMINISTIC_OPS'] = '1'
                    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
                    tf.keras.utils.set_random_seed(1)
                    model = tf.keras.models.Sequential([tf.keras.layers.Dense(35, input_shape=shape, activation = "elu"),
                                                            tf.keras.layers.Dense(25, activation = "elu"),
                                                            tf.keras.layers.Dense(25, activation = "elu"),
                                                            tf.keras.layers.Dense(1, activation = "sigmoid")]) 
                    optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1) #Adagrad
                    model.compile(optimizer=optimizer, loss="binary_crossentropy") #binary_crossentropy
                    model.fit(X_train, Y_train, batch_size=512, epochs=100, verbose=0)

    #             elif shape[0] == 10:
    #                 model = tf.keras.models.Sequential([tf.keras.layers.Dense(35, input_shape=shape, activation = "elu"),
    #                                                     tf.keras.layers.Dense(25, activation = "elu"),
    #                                                         tf.keras.layers.Dense(25, activation = "elu"),
    #                                                         tf.keras.layers.Dense(25, activation = "elu"),
    #                                                         tf.keras.layers.Dense(1, activation = "sigmoid")]) 
    #                 optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.07) #Adagrad
    #                 model.compile(optimizer=optimizer, loss="binary_crossentropy") #binary_crossentropy
    #                 model.fit(X_train, Y_train, batch_size=512, epochs=100, verbose=0)

    #             else:
    #                 model = tf.keras.models.Sequential([tf.keras.layers.Dense(35, input_shape=shape, activation = "elu"),
    #                                                         tf.keras.layers.Dense(25, activation = "elu"),
    #                                                         tf.keras.layers.Dense(25, activation = "elu"),
    #                                                         tf.keras.layers.Dense(25, activation = "elu"),
    #                                                         tf.keras.layers.Dense(25, activation = "elu"),
    #                                                         tf.keras.layers.Dense(1, activation = "sigmoid")]) 
    #                 optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.06) #Adagrad
    #                 model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['binary_accuracy']) #binary_crossentropy
    #                 model.fit(X_train, Y_train, batch_size=512, epochs=100, verbose=0)

                probs = model.predict(X_val)
                acc_arr_NN.append(balanced_accuracy_score(Y_val, np.round(probs)))
                f1_arr_NN.append(f1_score(Y_val, np.round(probs),average='macro'))
                auc_arr_NN.append(roc_auc_score(Y_val, np.round(probs)))
            
                i += 1

                # bar plot
                X_train_np = np.array(X_train) 
                X_test_np = np.array(X_val)  # Convert X_test to a NumPy array
                feature_names = X_train.columns
                
                if self.iscv:
                    sv_list = self.shapley_feature_ranking(model, X_train_np, X_test_np, train_size, j, 'nn', False, featurenames=feature_names)
                    perm = permutation_importance(model, X_train, Y_train, scoring=scoring_function, n_repeats=5, random_state=42)
                    fi_list = self.get_feat_importance(perm, train_size, j, 'nn')
                    
                if self.isglobal and self.iscv:
                    plot_global(model, X_train_np, X_test_np, isnn=True, featurenames=feature_names )
                if self.isdependency and self.iscv:
                    plot_dependency(model, X_train, X_val)
        
        results['true'] = Y_test
        
        if self.islda:
            clf = clf.fit(X_, Y_)
            pred = clf.predict(X_test)
            predictions = clf.predict_proba(X_test)
            probs = predictions[:,1]
            results['LDA','probs'] = probs
            results_var['LDA_var','BA'] = np.var(acc_arr_LDA, ddof=1)
            results_var['LDA_var','F1'] = np.var(f1_arr_LDA, ddof=1)
            results_var['LDA_var','AUC'] = np.var(auc_arr_LDA, ddof=1)
            
            if self.isglobal and not self.iscv:
                plot_global(clf, X_test, X_test)
            if self.isdependency and not self.iscv:
                plot_dependency(clf, X_test, X_test)
            if self.isfi and not self.iscv:
                # get feature importance
                sv_list = self.shapley_feature_ranking(clf, X_test, X_test, train_size, 0, 'lda', False)
                fi_list = self.get_feat_importance(clf, train_size, 0, 'lda')

            print(' Accuracy for validation set LDA: ', list(np.around(np.array(acc_arr_LDA),2)), ", CV mean: ", sum(acc_arr_LDA) / len(acc_arr_LDA))

        if self.isxgb:
            xgb = xgb.fit(X_, Y_)
            pred = xgb.predict(X_test)
            predictions = xgb.predict_proba(X_test)
            probs = predictions[:,1]
            results['XGB','probs'] = probs
            results_var['XGB_var','BA'] = np.var(acc_arr_XGB, ddof=1)
            results_var['XGB_var','F1'] = np.var(f1_arr_XGB, ddof=1)
            results_var['XGB_var','AUC'] = np.var(auc_arr_XGB, ddof=1)
            
            if self.isglobal and not self.iscv:
                plot_global(xgb, X_test, X_test, is_check_additivity_false = True)
            if self.isdependency and not self.iscv:
                plot_dependency(xgb, X_test, X_test, isxgb=True)
            if self.isfi and not self.iscv:
                sv_list = self.shapley_feature_ranking(xgb, X_test, X_test, train_size, 0, 'xgb', True)
                fi_list = self.get_feat_importance(xgb, train_size, 0, 'xgb')

            print(' Accuracy for validation set XGB: ', list(np.around(np.array(acc_arr_XGB),2)), ", CV mean: ", sum(acc_arr_XGB) / len(acc_arr_XGB))

        if self.isnn:
            model.fit(X_, Y_, batch_size=512, epochs=100, verbose=0)
            probs = model.predict(X_test)
            results['NN','probs'] = probs
            results_var['NN_var','BA'] = np.var(acc_arr_NN, ddof=1)
            results_var['NN_var','F1'] = np.var(f1_arr_NN, ddof=1)
            results_var['NN_var','AUC'] = np.var(auc_arr_NN, ddof=1)

            X_test_np = np.array(X_test)  # Convert X_test to a NumPy array
            feature_names = X_test.columns
            if self.isglobal and not self.iscv:
                plot_global(model, X_test_np,X_test_np, isnn = True, featurenames = feature_names)
            if self.isdependency and not self.iscv:
                plot_dependency(model, X_test, X_test)
            if self.isfi and not self.iscv:
                # get feature importance
                sv_list = self.shapley_feature_ranking(model, X_test_np, X_test_np, train_size, 0, 'nn', False, featurenames=feature_names)
                perm = permutation_importance(model, X_, Y_, scoring=scoring_function, random_state=42)
    #             perm = PermutationImportance(model, random_state=1, scoring=scoring_function).fit(X_,Y_)
                fi_list = self.get_feat_importance(perm, train_size, 0, 'nn')

            print(' Accuracy for validation set NN: ', list(np.around(np.array(acc_arr_NN),2)), ", CV mean: ", sum(acc_arr_NN) / len(acc_arr_NN))

        return results, results_var, myfeatures, sv_list, fi_list

    def display_results(self, data, sample, algo_list, train_size_arr, method):
        """method to generate results for every size of data in TRAIN_SIZE_ARR

        Args:
            data 
            sample 
            algo_list 
            train_size_arr 
            method 
        """
        for t in train_size_arr:
            for i in range(1):
                if(self.case_no == 2):
                    j = 1
                elif(self.case_no == 3):
                    j = 2
                else:
                    j = 0
                print('------------- '+'loop train size: '+str(t)+'%  - case: '+str(j)+'-----------------')        
                self.results[t, j], self.results_var[t,j] , myfeatures, sv_list, fi_list = self.train_test(data[j],j+1,sample, method, train_size = t)
                print('-------------------------results_var[t,j]--------------------------\n')
                print(self.results_var[t,j])
                print('-------------------------results_var[t,j]--------------------------\n')

        ## ==== Results ==== ##
        eval_mat={}
        for t in train_size_arr:
            for i in range(1):
                if(self.case_no == 2):
                    j = 1
                elif(self.case_no == 3):
                    j = 2
                else:
                    j = 0
                eval_mat[t, j] = overall(self.results[t,j],j, train_size=t, \
                                        islda = self.islda, isxgb = self.isxgb, isnn = self.isnn)

            
        for key in eval_mat:
            print('\n ---------- train size '+ str(key[0]) +'% case: '+str(key[1])+'----------')
            for items in eval_mat[key].items():
                print(items)
            
        for i, algo in enumerate(algo_list):
            f1_arr = []
            ba_arr = []
            auc_arr = []
            f1_arr_c1 = []
            ba_arr_c1 = []
            auc_arr_c1 = []
            f1_arr_c2 = []
            ba_arr_c2 = []
            auc_arr_c2 = []

            for key in eval_mat:
                for items, value in eval_mat[key].items():
                    if items==algo and key[1]==0:
                        f1 = value[(algo.upper(), 'F1')]
                        ba = value[(algo.upper(), 'Bal_Acc')]
                        auc = value[(algo.upper(), 'AUC')]
                        f1_arr.append(f1)
                        ba_arr.append(ba)
                        auc_arr.append(auc)
    #                 elif items==algo and key[1]==1:
    #                     f1 = value[(algo.upper(), 'F1')]
    #                     ba = value[(algo.upper(), 'Bal_Acc')]
    #                     auc = value[(algo.upper(), 'AUC')]
    #                     f1_arr_c1.append(f1)
    #                     ba_arr_c1.append(ba)
    #                     auc_arr_c1.append(auc)
    #                 elif items==algo and key[1]==2:
    #                     f1 = value[(algo.upper(), 'F1')]
    #                     ba = value[(algo.upper(), 'Bal_Acc')]
    #                     auc = value[(algo.upper(), 'AUC')]
    #                     f1_arr_c2.append(f1)
    #                     ba_arr_c2.append(ba)
    #                     auc_arr_c2.append(auc)

            print('case 0 balance acc:  ', list(np.around(np.array(ba_arr),2)))
        
        return sv_list, fi_list, myfeatures
    
    def plot_data(self, sv_list, myfeatures):
        """method to plot box and bar plots for every data characteristics

        Args:
            sv_list : shap value list
            myfeatures : feature list

        """
        data ={}
        df = pd.DataFrame(sv_list).T.reset_index()
        
        if self.iscv:
            df.columns = ['size','cvloop','model'] + myfeatures.to_list()
        else:
            df.columns = ['size', 'cvloop','model'] + myfeatures.to_list()
        
        models = {'lda': self.islda, 'xgb': self.isxgb, 'nn': self.isnn}
        
        for model, flag in models.items():
            if flag:
                df_model = df[df['model'] == model]
                
                if self.iscv:
                    plot_shap_box_plots(df_model, model.upper(), myfeatures, iscv=self.iscv)
                data[model] = plot_shap_bar_plots(df_model, model.upper(), myfeatures, iscv=self.iscv)
        return data, df
    
    def plot_together(self, data, df, fi_list, myfeatures, file_name):
        """method to plot bar and box plots together

        Args:
            data
            df
            fi_list : feature important list
            myfeatures : feature list
            file_name : figure name
        """
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 18
        custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        figsize = (18, 4)
        models = {'lda': self.islda, 'xgb': self.isxgb, 'nn': self.isnn}
        
        values_df = data['nn'].groupby(['size', 'features'])['value'].mean().reset_index()
        df_fi = pd.DataFrame(fi_list).T.reset_index()

        df_fi.columns = ['size','cvloop','model'] + myfeatures.to_list()
        plot_combined_graphs(models, df, file_name, myfeatures, custom_colors, iscv=self.iscv)


    def plot_umap_for_models(self, models_list, df, saveFile):
        """Calculate and plot umap

        Args:
            models_list : list of models
            df : data
            saveFile : figure name

        """
        plt.rcParams['font.family'] = 'serif'
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        figsize = (18, 4)
        
        features = ['Fasting plasma glucose', 'HbA1c 1 year before', 'HbA1c 2 years before', 'Other cardiac diseases', 'T2D duration in years']
        target = 'size'

        fig, axes = plt.subplots(1, len(models_list), figsize=figsize, gridspec_kw={'wspace': 0.2})
        fig.tight_layout()
        # fig.text(0.5, 0.99, 'Imbalanced Background Data', fontsize=18, fontweight='bold', ha='center')


        for i, model in enumerate(models_list):
            df_model = df.loc[df['model'] == model]
            X = df_model[features].values
            Y = df_model[target].values

            reducer = umap.UMAP(random_state=123)
            embedding = reducer.fit_transform(X)

            scatter = axes[i].scatter(
                embedding[:, 0],
                embedding[:, 1],
                cmap='viridis',
                c=Y,
                s=50,
                edgecolors='w',
                label='Size',
                zorder=2
            )

            axes[i].set_title(model.upper(), fontsize=16, fontweight='bold')
            axes[i].set_xlabel('UMAP Dimension 1', fontsize=14)
            axes[i].set_ylabel('UMAP Dimension 2', fontsize=14)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].tick_params(axis='both', which='major', labelsize=9)
            axes[i].grid(color='gray', linestyle='--', linewidth=0.5)  # Adding grids
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].legend(fontsize=12, title_fontsize=9)

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label('Size', fontsize=12)
        cbar.ax.tick_params(labelsize=12)

        plt.savefig(saveFile, dpi=300, bbox_inches='tight')
        plt.show()
        return df_model

    def calculate_varience(self, df, myfeatures):
        """method to calculate variance 

        Args:
            df : data
            myfeatures : feature list
        """
        data = pd.melt(df, id_vars=['size'], value_vars=myfeatures, var_name='features')

        variances = data.groupby(['size', 'features'])['value'].var()
        print("Variances for each group:")
        print(variances)
        
if __name__ == "__main__":
    print("Initiate model...")
    # Train models with balanced data. 
    # To change the oversample method change method name in baseT.display_results. 
    # available method names 'adasyn', 'smote' and 'synthpop'
    print('######## Models with balanced data #########')
    baseT = BaseTrainer()
    data, X, Y = baseT.get_preprocess_data()
    print('Shape of the data after preprocessing :' ,data[0][0].shape)
    
    print_data_shapes(X, Y, response_variable = 'c4')
    np.random.seed(55)
    tf.random.set_seed(55)
    
    sv_list, fi_list, myfeatures= baseT.display_results(data, baseT.sample, ALGO_LIST,\
                                                        TRAIN_SIZE_ARR, method = 'smote')
    
    data, df = baseT.plot_data(sv_list, myfeatures)
    baseT.plot_together(data, df, fi_list, myfeatures, baseT.fig_balance_box_plot_save_path)
    X = baseT.plot_umap_for_models(ALGO_LIST, df, baseT.fig_balance_umap_save_path)
    baseT.calculate_varience(df, myfeatures)
    
    # Train models with imbalanced data. 
    # To train models with original (imbalaned) data, change method name in baseT.display_results to ''. 
    print('######## Models with imbalanced data #########')
    baseT_imbalance = BaseTrainer()
    data, X, Y = baseT_imbalance.get_preprocess_data()
    print('Shape of the data after preprocessing :' ,data[0][0].shape)
    
    print_data_shapes(X, Y, response_variable = 'c4')
    np.random.seed(55)
    tf.random.set_seed(55)
    
    sv_list, fi_list, myfeatures= baseT_imbalance.display_results(data, baseT_imbalance.sample, ALGO_LIST,\
                                                        TRAIN_SIZE_ARR, method = '')
    
    data, df = baseT_imbalance.plot_data(sv_list, myfeatures)
    baseT_imbalance.plot_together(data, df, fi_list, myfeatures, baseT_imbalance.fig_imbalance_box_plot_save_path)
    X = baseT_imbalance.plot_umap_for_models(ALGO_LIST, df, baseT_imbalance.fig_imbalance_umap_save_path)
    baseT_imbalance.calculate_varience(df, myfeatures)
    
    
    
    