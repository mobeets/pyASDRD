# import pickle
# import pylab
import numpy as np
import matplotlib.pyplot as plt
from resp import Resp
from stim import Stim

def plot(S, R):
    plt.plot(np.sum(S.Sb, 1), R.R, '.')
    # plt.plot(np.sum(S.Sb, 1), '.')
    # plt.plot(R.R, '.')
    plt.show()
    return

    plt.imshow(np.var(S.S, 0))
    plt.set_cmap('gray')
    plt.colorbar()
    plt.show()

def main():
    n, ns, nt = 1000, 19, 7
    S = Stim(n, nt, ns)
    R = Resp(S.S, S.xy)
    plot(S, R)

# def run():
#     with open("modelfitDatabase1.dat",'rb') as f:    
#         dd = pickle.load(f)

#     node = dd.children[0]
#     activities = node.data["training_set"]
#     training_inputs = node.data["training_inputs"]
#     validation_activities = node.data["validation_set"]
#     validation_inputs  = node.data["validation_inputs"]

#     (p,q) = numpy.shape(training_inputs)
#     q = int(numpy.sqrt(q))
    
#     X = numpy.vstack([ i * numpy.ones((1,q)) for i in xrange(0,q)]).flatten()
#     Y = numpy.hstack([ i * numpy.ones((q,1)) for i in xrange(0,q)]).flatten()
    
#     params = {}
#     params["Method"] = __main__.__dict__.get('Method','ASD')

#     dist = numpy.zeros((len(X),len(X)))
#     for i in xrange(0,len(X)):
#         for j in xrange(0,len(X)):
#             dist[i][j] = numpy.sqrt(numpy.power(X[i] - X[j],2) + numpy.power(Y[i] - Y[j],2))/q
        
        
#     numpy.savetxt('/home/antolikjan/MATLAB/inputs.csv', training_inputs, fmt='%.6f', delimiter=';')     
#     numpy.savetxt('/home/antolikjan/MATLAB/val_inputs.csv', validation_inputs, fmt='%.6f', delimiter=';')
#     numpy.savetxt('/home/antolikjan/MATLAB/activities.csv', activities, fmt='%.6f', delimiter=';')
#     numpy.savetxt('/home/antolikjan/MATLAB/distances.csv', dist, fmt='%.6f', delimiter=';')
#     return
#     #w,S = ASD(numpy.mat(training_inputs),numpy.mat(activities[:,0]).T,numpy.array(dist))
#     #w = ARD(numpy.mat(training_inputs),numpy.mat(activities[:,0]).T)
    
    
#     S = dd.children[0].children[1].data["S"][0]
#     w = dd.children[0].children[1].data["RFs"][0]
    
#     w = ASDRD(numpy.mat(training_inputs),numpy.mat(activities[:,0]).T,S)
#     return w
    
#     node = node.get_child(params)
    
#     RFs = []
#     S = []
#     for i in xrange(0,103):
#         w,s = ASD(numpy.mat(training_inputs),numpy.mat(activities[:,i]).T,numpy.array(dist))            
#         RFs.append(w)
#         S.append(s)
    
#     node.add_data("RFs",RFs,force=True)
#     node.add_data("S",S,force=True)
    
#     #pylab.figure()
#     #m = numpy.max(numpy.abs(w))
#     #pylab.imshow(w.reshape(q,q),vmin=-m,vmax=m,cmap=pylab.cm.jet,interpolation='nearest')
#     #pylab.colorbar()
#     #return w.reshape(q,q)
#     with open("modelfitDB2.dat",'wb') as f:
#         pickle.dump(dd,f,-2)

#     return RFs

if __name__ == '__main__':
    main()
