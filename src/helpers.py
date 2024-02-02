import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
from synthpop import Synthpop
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from pdpbox import pdp
from sklearn.manifold import TSNE
import plotly.express as px

def select_features(select, method, feature):
    """This function use SelectKBest method to select features from preprocessed dataset

    Args:
        select: data
        method: score function
        feature: number of features to select

    Returns:
        _type_: _description_
    """
    fs = SelectKBest(score_func=method, k=feature)
    selected = fs.fit_transform(select[0], select[1])
    idx = fs.get_support(indices=True)
    return selected, fs, idx

def imputation(X_train, Y_train, X_test, Y_test, response_variable, t=False):
    """Handle the data imputation

    Args:
        X_train: Training data
        Y_train: Training data
        X_test: Test data
        Y_test: Test data
        response_variable: Resoponse variable name
        t (bool, optional): Defaults to False. Check to remove NA values from test dataset or not

    Returns:
        X_train, Y_train, X_test, Y_test: Imputed datasets
    """
    Train = X_train.copy()
    Train.loc[:,response_variable] = Y_train
    
    datasets= {}
    by_class = Train.groupby(response_variable)
    for groups, data in by_class:
        datasets[groups] = data

    X_train_0 = datasets[0][datasets[0].columns.difference([response_variable], sort = False)]
    Y_train_0 = datasets[0].loc[:,response_variable].copy()
    X_train_1 = datasets[1][datasets[1].columns.difference([response_variable], sort = False)]
    Y_train_1 = datasets[1].loc[:,response_variable].copy()

    imp_0 = SimpleImputer(strategy = "most_frequent")
    X_train_0 = imp_0.fit_transform(X_train_0)

    imp_1 = SimpleImputer(strategy = "most_frequent")
    X_train_1 = imp_1.fit_transform(X_train_1)

    X_train = np.concatenate((X_train_0,X_train_1))
    Y_train = np.concatenate((Y_train_0,Y_train_1))
    
    if(not t):
    # Remove NA from test
        Test = X_test.copy()
        Test.loc[:,response_variable] = Y_test
        Test = Test.dropna()
        X_test = Test[Test.columns.difference(['c4'], sort = False)]
        Y_test = Test.loc[:,response_variable].copy()
    
    return X_train, Y_train, X_test, Y_test

def conf_repo(true, probs, model):
    """This handles the model performance matrices.

    Args:
        true : True values
        probs : Predicted probabilities
        model : Trained model
    
    Returns
        eval_mat: evaluation matrices including AUC, f1 score and balance accuracy
    """
    eval_mat = {}
    confusion_matrix_df = confusion_matrix(true, np.round(probs))
    print(confusion_matrix_df)
    kappa = cohen_kappa_score(true, np.round(probs))
    print(kappa)
    f1 = f1_score(true,np.round(probs),average='macro')
    print('F1 score: ', f1)
    acc = accuracy_score(true, np.round(probs), normalize=True, sample_weight=None)
    bal_acc = balanced_accuracy_score(true, np.round(probs))
    print("Accuracy: ", acc)
    print("Balanced Accuracy: ", bal_acc)
    auc_ = roc_auc_score(true, probs)
    print("AUC: ", auc_)
    
    eval_mat[model, 'F1'] = f1
    eval_mat[model, 'Bal_Acc'] = bal_acc
    eval_mat[model, 'AUC'] = auc_
    
    fpr, tpr, _ = roc_curve(true, probs)
#     plt.plot(fpr,tpr,label=model+' (AUC='+str(auc_)+")")
#     plt.xlabel('False positive rate')
#     plt.ylabel('True positive rate')
#     plt.title("ROC Curve Comparison")
#     plt.legend(loc=4)
#     plt.show()
    return eval_mat

def overall(results, k, train_size, islda = False, isxgb = False, isnn = False):
    """Handles the model evaluation for each model in the ALGO_LIST

    Args:
        results : dictionary to store data
        k : Case number
        train_size : Size of the training data
        islda (bool, optional): LDA model is in the ALGO_LIST or not. Defaults to False.
        isxgb (bool, optional): XGB model is in the ALGO_LIST or not. Defaults to False.
        isnn (bool, optional): NN model is in the ALGO_LIST or not. Defaults to False.
    """
    eval_mat = {}
    true, probs_LDA, probs_XGB, probs_NN = ([],)*4
#     for i in range(splits):
    true = np.append(true,results['true'])
    
    if islda:
        probs_LDA = np.append(probs_LDA,results['LDA', 'probs'])
        print('Final results for LDA' +' case: '+ str(k) + ' train size: '+ str(train_size))
        eval_mat['lda'] = conf_repo(true,probs_LDA,model='LDA')
    if isxgb:
        probs_XGB = np.append(probs_XGB,results['XGB', 'probs'])
        print('Final results for XGB' +' case: '+ str(k) + ' train size: '+ str(train_size))
        eval_mat['xgb'] = conf_repo(true,probs_XGB,model='XGB')
    if isnn:
        probs_NN = np.append(probs_NN,results['NN', 'probs'])
        print('Final results for NN' +' case: '+ str(k) + ' train size: '+ str(train_size))
        eval_mat['nn'] = conf_repo(true,probs_NN,model='NN')
    return eval_mat

def print_data_shapes(X, Y, response_variable):
    """This function use to display shape of the data

    Args:
        X : data with input features
        Y : response data
        response_variable : name of the response variable
    """
    data_ori = X
    out_ori = Y

    print('shape of data: ', data_ori.shape)
    print('shape of output: ', out_ori.shape)

    dd = pd.concat([X, Y],axis=1)
    print('Response variable value count : ', dd[response_variable].value_counts())
    
def get_data_types(data):
    """Function to return data types

    Args:
        data

    Returns:
        converted dataframe
    """
    dt = data.dtypes.to_dict()
    for k, v in dt.items():
        if v=='int64':
            dt[k] = 'int'
        elif v == 'float64':
            dt[k] = 'float'
        elif v == 'category':
            dt[k]='category'
    return dt

def create_artificial_data(data, percentage = 120, method = 'smote'):
    """This function generate data samples using three different methods, smote, adasyn and synthpop.

    Args:
        data 
        percentage (int, optional): percentage of size want to increate. Defaults to 120.
        method (str, optional): oversampling method name. Defaults to 'smote'.
    """
    no_samples = int(np.round(data.shape[0]*(percentage/100)))
    artificial_samples_count = no_samples
    response_col = 'c4'
    
    X = data[data.columns.difference([response_col])]
    y = data[response_col]
    X, Y, X_test, Y_test = imputation(X, y, None, None, t = True, response_variable='c4')
    Y = pd.DataFrame(Y, columns=['c4'])
    
    print('SHAPE OF Y:', Y.value_counts())
    
    if method == 'smote':
        print('Use SMOTE')
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, Y)
        X_res = pd.DataFrame(X_res)
        print('SHAPE OF Y AFTER SMOTE:',  y_res.value_counts())
        y_res = pd.DataFrame(y_res)
    elif method == 'adasyn':
        print('Use Adasyn')
        sm = ADASYN(random_state=42)
        X_res, y_res = sm.fit_resample(X, Y)
        X_res = pd.DataFrame(X_res)
        print('SHAPE OF Y AFTER ADASYN:',  y_res.value_counts())
        y_res = pd.DataFrame(y_res)
    elif method == 'synthpop':
        print('Use synthpop')
        model = Synthpop()
        my_data_types = get_data_types(data)
        no_samples = len(Y[Y['c4']==0]) - len(Y[Y['c4']==1])
        
        data_to_fit = data[data['c4']==1]
        model.fit(data_to_fit, dtypes=my_data_types)
        result = model.generate(k=no_samples)
        result = pd.concat([result, data], axis=0)
        
        print('synth shape ',result.shape)
        print('synth value count \n ', result['c4'].value_counts())
        X_res = result[result.columns.difference([response_col])]
        y_res = result[response_col]
    else:
        # No artificial data
        X_res = data[data.columns.difference([response_col])]
        y_res = data[response_col]
        
    return X_res, y_res

def get_test_data(data, n_samples):
    """Function to randomly pick n_samples number of test data

    Args:
        data 
        n_samples : number of test samples

    Returns:
        result : dataset after removed the selected test data
        test_data : selected test data
    """
    result = pd.concat([data[0], data[1]], axis=1)
    test_data = result.sample(n=n_samples, random_state=1)
    idx_to_drop = result.sample(n=n_samples, random_state=1).index
    result.drop(idx_to_drop, inplace=True, axis=0)
    test_data = test_data.dropna()
    
    while (test_data.shape[0] < n_samples):
        samples = n_samples-test_data.shape[0]        
        t = result.sample(n=samples, random_state=1)
        idx_to_drop = result.sample(n=samples, random_state=1).index
        result.drop(idx_to_drop, inplace=True, axis=0)
        t = t.dropna()
        test_data=pd.concat([test_data,t])
    
    return result, test_data

def perc_of_original_train_data(data, use_perc):
    """select percetage of data

    Args:
        data 
        use_perc : Percentage to select

    Returns:
        data : data after selecting the use_perc of original data
    """
    # get 10% of original data
    data = data.head(int(len(data)*(use_perc/100)))
    return data

def get_perc_of_train_data(data, perc, method):
    """function to select percentage of data, and oversample

    Args:
        data 
        perc : percentage
        method : oversampling method

    Returns:
        X_res, y_res : oversampled data
    """
    data = data.drop_duplicates()
    if(perc > 100):
        X_res, y_res = create_artificial_data(data, percentage=perc, method = method)
    else:
        # get first perc% of data
        data = data.head(int(len(data)*(perc/100)))
        X_res, y_res = create_artificial_data(data, percentage=perc, method = method)
        
    return X_res, y_res

def get_shap_values(model,  X_train, X_val, is_check_additivity_false = False, isnn=False, featurenames=[]):
    """Calculate the shap values. NN handle differently in this method

    Args:
        model : trained model
        X_train : training data
        X_val : validation data
        is_check_additivity_false (bool, optional): check_additivity hyper parameter value. Defaults to False.
        isnn (bool, optional): model is NN or not. Defaults to False.
        featurenames (list, optional): list of selected feature names. Defaults to [].
    """
    explainer = shap.Explainer(model, X_val)
    if isnn:
#         explainer_N = shap.DeepExplainer(model, np.array(X_train)) #KernelExplainer #DeepExplainer
#         shap_vals = explainer_N.shap_values(np.array(X_val))
#         fig = plt.figure(figsize=(10,7))
#         shap.summary_plot(shap_vals[0], X_val, feature_names=featurenames, cmap = "cividis")
#         plt.show()
        explainer.feature_names = featurenames
    if is_check_additivity_false:
        shap_values = explainer(X_val, check_additivity = False)
    else:
        shap_values = explainer(X_val)
    return shap_values

def plot_global(model, X_train, X_val, is_check_additivity_false = False, \
                isnn=False, featurenames=[], islocal=False):
    """function to plot shap plots.

    Args:
        model 
        X_train 
        X_val 
        is_check_additivity_false (bool, optional): check_additivity hyper parameter value. Defaults to False.
        featurenames (list, optional): List of feature names. Defaults to [].
        islocal (bool, optional): check plot is local or global. Defaults to False.
    """

    shap_values = get_shap_values(model,  X_train, X_val, is_check_additivity_false, isnn, featurenames)
    # shap.plots.bar(shap_values[0])
    shap.plots.beeswarm(shap_values)
    
    if islocal:
        shap.plots.force(shap_values[1], matplotlib=True)
        # plt.savefig('force_plot.png')

def plot_dependency(model, X_train, X_val, isxgb=False):
    """function to plot dependency plot.

    Args:
        model 
        X_train 
        X_val 
        isxgb (bool, optional): Defaults to False.
    """
    feature_names=X_val.columns
    for i, feature_name in enumerate(feature_names):

        # Compute the partial dependence values
        pdp_values = pdp.pdp_isolate(
            model=model,
            dataset=X_val,
            model_features=feature_names,
            feature=feature_name
            )
        # Plot the Partial Dependence Plot
        pdp.pdp_plot(pdp_values, feature_name, plot_pts_dist=True, plot_lines=True, figsize=(4,5))
        plt.tight_layout()
        plt.show()

def scoring_function(model, X, y):
    """score function. This will use in NN, when calculating the permutation importance

    Args:
        model : NN model instance
        X : data
        y : Y

    Returns:
        score
    """
    y_pred = model.predict(X)
    return np.sqrt(np.mean((y - y_pred) ** 2))


def plot_shap_box_plots(df, title, myfeatures, iscv=False):
    """function to plot box plots of mean absolute shap values

    Args:
        df 
        title (sring): title of the plot
        myfeatures : feature list
        iscv (bool, optional): cross validation or not. Defaults to False.
    """
    dd=pd.melt(df,id_vars=['size'],value_vars=myfeatures,var_name='features')
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    ax = sns.boxplot(x='size',y='value',data=dd,hue='features')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(.5, 1), ncol=1, title='Features', frameon=False)
    if iscv:
        ax.set(xlabel ='Training percentages (%)', ylabel = "Mean SHAP values of cross validation loops", 
               title = title)
    else: 
        ax.set(xlabel ='Training percentages (%)', ylabel = "SHAP values", 
               title = title)
    plt.show()
    
def plot_shap_bar_plots(df, title, myfeatures, iscv = False):
    """function to plot bar plots of mean absolute shap values

    Args:
        df 
        title (sring): title of the plot
        myfeatures : feature list
        iscv (bool, optional): cross validation or not. Defaults to False.
    """
    dd1 = pd.melt(df,id_vars=['size'],value_vars=myfeatures,var_name='features')
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    ax = sns.barplot(x='size',y='value',data = dd1,hue='features')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(.5, 1), ncol=1, title='Features', frameon=False)
    if iscv:
        ax.set(xlabel ='Training percentages (%)', ylabel = "Mean SHAP values of cross validation loops", 
               title = title)
    else: 
        ax.set(xlabel ='Training percentages (%)', ylabel = "SHAP values", 
               title = title)
    plt.show()
    return dd1

def plot_shap_box_plots_axis(ax, df, title, myfeatures, custom_colors, iscv=False):
    """function to plot box plots of mean absolute shap values, to use in paper 

    Args:
        ax : axis
        df : data
        title (sring): title of the plot
        myfeatures : feature list
        custom_colors : plot colors
        iscv (bool, optional): cross validation or not. Defaults to False.
    """
    dd = pd.melt(df, id_vars=['size'], value_vars=myfeatures, var_name='features')
    sns.boxplot(x='size', y='value', data=dd, hue='features', ax=ax, palette=custom_colors)
    ax.get_legend().remove()

    if iscv:
        ax.set(xlabel='Training percentages (%)', ylabel="Mean Absolute SHAP values",
               title=title)
    else:
        ax.set(xlabel='Training percentages (%)', ylabel="SHAP values",
               title=title)

    ax.set_xlabel('Training percentages (%)', fontsize=14)
    ax.set_ylabel('Mean Absolute SHAP values', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.grid(color='gray', linestyle='--', linewidth=0.5)  # Adding grids
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
def plot_shap_bar_plots_axis(ax, df, title, myfeatures, custom_colors, iscv=False):
    """function to plot bar plots of mean absolute shap values, to use in paper 

    Args:
        ax : axis
        df : data
        title (sring): title of the plot
        myfeatures : feature list
        custom_colors : plot colors
        iscv (bool, optional): cross validation or not. Defaults to False.
    """
    dd1 = pd.melt(df, id_vars=['size'], value_vars=myfeatures, var_name='features')
    sns.barplot(x='size', y='value', data=dd1, hue='features', ax=ax, palette=custom_colors)
    ax.get_legend().remove()

    if iscv:
        ax.set(xlabel='Training percentages (%)', ylabel="Mean Absolute SHAP values",
               title=title)
    else:
        ax.set(xlabel='Training percentages (%)', ylabel="SHAP values",
               title=title)

    ax.set_xlabel('Training percentages (%)', fontsize=14)
    ax.set_ylabel('Mean Absolute SHAP values', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(color='gray', linestyle='--', linewidth=0.5)  # Adding grids
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
def min_max_normalization(group):
    selected_columns = ['Fasting plasma glucose', 'HbA1c 1 year before', 'HbA1c 2 years before', 'Other cardiac diseases',
    'T2D duration in years']
    group[selected_columns] = (group[selected_columns] - group[selected_columns].min()) / (group[selected_columns].max() - group[selected_columns].min())
    return group

def log_min_max_normalization(group):
    selected_columns = ['Fasting plasma glucose', 'HbA1c 1 year before', 'HbA1c 2 years before', 'Other cardiac diseases', 'T2D duration in years']

    # Apply log transformation to the selected columns
    group[selected_columns] = np.log1p(group[selected_columns])
    return group

def plot_combined_graphs(models, df_fi, saveFile, myfeatures, custom_colors, iscv=False):
    """function to combine bar and box plots 

    Args:
        models 
        df_fi : data
        saveFile : figure name
        myfeatures : feature list
        custom_colors : plot colors
        iscv (bool, optional): used cross validation or not. Defaults to False.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    i = 0

    for model, flag in models.items():
        df_model = df_fi[df_fi['model'] == model]

        if flag:
            plot_shap_box_plots_axis(axes[i], df_model, model.upper(), myfeatures, custom_colors, iscv)
            #plot_shap_bar_plots_axis(axes[1, i], df_model, model.upper(), myfeatures, custom_colors, iscv)
            i += 1

    fig.subplots_adjust(wspace=0.4, hspace=0.4)  # Increase the vertical spacing between subplots

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', title='Features',
               frameon=False, ncol=3, fontsize=12, bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the top margin to create space for the title

    plt.savefig(saveFile, dpi=300, bbox_inches='tight')
    plt.show()
    

def plot_tsne(df, model, perplexity):
    """function to plot t-SNE

    Args:
        df : data
        model : trained model
        perplexity : _description_
    """
    df1=df.loc[df['model'] == model]
    
    #if(case_no == 2):
    #    features = df.iloc[:, 3:13].columns.values
    #elif (case_no == 3):
    #    features = df.iloc[:, 3:18].columns.values
    #else:
    #    features = ['Fasting plasma glucose', 'HbA1c 1 year before', 'HbA1c 2 years before', 'Other cardiac diseases',
    #       'T2D duration in years']
    features = ['Fasting plasma glucose', 'HbA1c 1 year before', 'HbA1c 2 years before', 'Other cardiac diseases',
           'T2D duration in years']   
    target = 'size'
    X = df1[features].values
    Y = df1[target].values

    tsne = TSNE(n_components=2, n_iter = 5000, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)
    tsne.kl_divergence_

    fig = px.scatter(x=X_tsne[:, 1], y=X_tsne[:, 0], color=Y, color_continuous_scale=px.colors.diverging.RdYlGn)

    fig.update_layout(
        title= model + " TSNE visualization of training size and features",
        xaxis_title="First t-SNE",
        yaxis_title="Second t-SNE",
    )
    fig.show()
