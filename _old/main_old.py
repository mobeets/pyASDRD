import pylab
import pickle
import numpy as np

def run():
    with open("modelfitDatabase1.dat",'rb') as f:    
        dd = pickle.load(f)

    node = dd.children[0]
    activities = node.data["training_set"]
    training_inputs = node.data["training_inputs"]
    validation_activities = node.data["validation_set"]
    validation_inputs  = node.data["validation_inputs"]

    (p,q) = np.shape(training_inputs)
    q = int(np.sqrt(q))
    
    X = np.vstack([ i * np.ones((1,q)) for i in xrange(0,q)]).flatten()
    Y = np.hstack([ i * np.ones((q,1)) for i in xrange(0,q)]).flatten()
    
    params = {}
    params["Method"] = __main__.__dict__.get('Method','ASD')

    dist = np.zeros((len(X),len(X)))
    for i in xrange(0,len(X)):
        for j in xrange(0,len(X)):
            dist[i][j] = np.sqrt(np.power(X[i] - X[j],2) + np.power(Y[i] - Y[j],2))/q
        
        
    np.savetxt('/home/antolikjan/MATLAB/inputs.csv', training_inputs, fmt='%.6f', delimiter=';')     
    np.savetxt('/home/antolikjan/MATLAB/val_inputs.csv', validation_inputs, fmt='%.6f', delimiter=';')
    np.savetxt('/home/antolikjan/MATLAB/activities.csv', activities, fmt='%.6f', delimiter=';')
    np.savetxt('/home/antolikjan/MATLAB/distances.csv', dist, fmt='%.6f', delimiter=';')
    return
    #w,S = ASD(np.mat(training_inputs),np.mat(activities[:,0]).T,np.array(dist))
    #w = ARD(np.mat(training_inputs),np.mat(activities[:,0]).T)
    
    
    S = dd.children[0].children[1].data["S"][0]
    w = dd.children[0].children[1].data["RFs"][0]
    
    w = ASDRD(np.mat(training_inputs),np.mat(activities[:,0]).T,S)
    return w
    
    node = node.get_child(params)
    
    RFs = []
    S = []
    for i in xrange(0,103):
        w,s = ASD(np.mat(training_inputs),np.mat(activities[:,i]).T,np.array(dist))            
        RFs.append(w)
        S.append(s)
    
    node.add_data("RFs",RFs,force=True)
    node.add_data("S",S,force=True)
    
    #pylab.figure()
    #m = np.max(np.abs(w))
    #pylab.imshow(w.reshape(q,q),vmin=-m,vmax=m,cmap=pylab.cm.jet,interpolation='nearest')
    #pylab.colorbar()
    #return w.reshape(q,q)
    with open("modelfitDB2.dat",'wb') as f:
        pickle.dump(dd,f,-2)

    return RFs
