
from sklearn.metrics import mean_squared_error
def plot(actual, predict, title):
    plt.figure(figsize=(8, 4))
    plt.scatter(actual, predict, s=20, c='steelblue')
    plt.title('Predicted vs. Actual '+ title)
    plt.xlabel('Actual Sale Price')
    plt.ylabel('Predicted Sale Price')

    plt.plot([min(actual), max(actual)], [min(actual), max(actual)],c ='red')
    plt.tight_layout()
    
def error_plot(actual, predict, title):
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1,2,1)
    
    plt.scatter(range(1,len(actual) + 1),actual-predict, s=20)
    plt.title('Sales Price Error '+ title)    
    plt.ylabel('Sales Price Error(log Actual - log Predict)')
    ax2 = fig.add_subplot(1,2,2)
    stats.probplot(actual - predict, dist="norm", plot=pylab)
    pylab.show()

    
    
    
def summary(model,train_features, train_labels, valid_features, valid_labels):
    ## Because I have done log transform to the labels already, so we use mean squared error here
    ## is actually mean suqared log error
    train_predicts = model.predict(train_features)
    valid_predicts = model.predict(valid_features)
    print('Cross Val (Train set)', np.sqrt(-cross_val_score(model, train_features, train_labels,
                scoring = 'neg_mean_squared_error')))
    print('Cross Val (valid set)',np.sqrt(-cross_val_score(model, valid_features, valid_labels,
                      scoring = 'neg_mean_squared_error')))
    print('Accuracy Val(Train set)', model.score(train_features, train_labels))
    print('Accuracy Val (Valid set)',model.score(valid_features, valid_labels))
    print('Roo tMean Squarted Log Error (Train set): ', mean_squared_error(train_labels,
                                                    train_predicts)**0.5)
    print('Root Mean Squarted Log Error (Valid set): ', mean_squared_error(valid_labels,
                                                    valid_predicts)**0.5)
    
    ## Graph Actual vs Predict
    plot(train_labels, train_predicts, 'Training Set')
    plot(valid_labels, valid_predicts, 'Validating Set')
    
    ## Error Plot
    error_plot(train_labels, train_predicts, 'Training Set')
    error_plot(valid_labels, valid_predicts, 'Validating Set')
    
   