#---------------------------------------------#
#-------| Written By: Shahwaiz |-------#
#---------------------------------------------#

# A good heuristic is to choose sqrt(nfeatures) to consider for each node...
import weakLearner as wl
import numpy as np
import scipy.stats as stats


#---------------Instructions------------------#

# Here you will have to reproduce the code you have already written in
# your previous assignment.

# However one major difference is that now each node non-terminal node of the
# tree  object will have  an instance of weaklearner...

# Look for the missing code sections and fill them.
#-------------------------------------------#

class Node:
    #def __init__(self,klasslabel='',pdistribution=[],score=0,wlearner=None,fidx=-1):
    def __init__(self,purity,klasslabel='',score=0,split=[],fidx=-1):
        """
               Input:
               --------------------------
               klasslabel: to use for leaf node
               pdistribution: posteriorprob class probability at the node
               score: split score 
               weaklearner: which weaklearner to use this node, an object of WeakLearner class or its childs...

        

        self.lchild=None       
        self.rchild=None
        self.klasslabel=klasslabel
        self.pdistribution=pdistribution
        self.score=score
        self.wlearner=wlearner
        """
        self.lchild=None       
        self.rchild=None
        self.klasslabel=klasslabel        
        self.split=split
        self.score=score
        self.fidx=fidx
        self.purity=purity
       
        
        
    def set_childs(self,lchild,rchild):
        """
        function used to set the childs of the node
        input:
            lchild: assign it to node left child
            rchild: assign it to node right child
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
            
        
        #---------End of Your Code-------------------------#

    def isleaf(self):
        """
            return true, if current node is leaf node
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
            
        
        #---------End of Your Code-------------------------#
    def isless_than_eq(self, X):
        """
            This function is used to decide which child node current example 
            should be directed to. i.e. returns true, if the current example should be
            sent to left child otherwise returns false.
        """

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        # Here you will call the evaluate funciton of weaklearn on
        # the current example and return true or false...
        
        #---------End of Your Code-------------------------#

    def get_str(self):
        """
            returns a string representing the node information...
        """
        if self.isleaf():
            return 'C(posterior={},class={},Purity={})'.format(self.pdistribution, self.klasslabel,self.purity)
        else:
            return 'I(Fidx={},Score={},Split={})'.format(self.fidx,self.score,self.split)
    

class DecisionTree:
    ''' Implements the Decision Tree For Classification With Information Gain 
        as Splitting Criterion....
    '''
    def __init__(self, purityp, exthreshold,maxdepth=10,fidx=-1):
    #(self, exthreshold=5, maxdepth=10,
     #weaklearner="Conic", pdist=False, nsplits=10, nfeattest=None):        
        ''' 
        Input:
        -----------------
            exthreshold: Number of examples to stop splitting, i.e. stop if number examples at a given node are less than exthreshold
            maxdepth: maximum depth of tree upto which we should grow the tree. Remember a tree with depth=10 
            has 2^10=1K child nodes.
            weaklearner: weaklearner to use at each internal node.
            pdist: return posterior class distribution or not...
            nsplits: number of splits to use for weaklearner
         
        self.maxdepth=maxdepth
        self.exthreshold=exthreshold
        self.weaklearner=weaklearner
        self.nsplits=nsplits
        self.pdist=pdist
        self.nfeattest=nfeattest
        '''
        self.purity=purityp
        self.exthreshold=exthreshold
        self.maxdepth=maxdepth
        ###__init__(self,purity,klasslabel='',score=0,split=[],fidx=-1)
        #node=None
        self.root=None
        
        #assert (weaklearner in ["Conic", "Linear","Axis-Aligned","Axis-Aligned-Random"])
        pass
    def getWeakLearner(self):
        if self.weaklearner == "Conic":
            return wl.ConicWeakLearner(self.nsplits)            
        elif self.weaklearner== "Linear":
            return wl.LinearWeakLearner(self.nsplits)
        elif self.weaklearner == "Axis-Aligned":
            return wl.WeakLearner()    
        else:
            return wl.RandomWeakLearner(self.nsplits,self.nfeattest)

        pass
    def train(self, X, Y):
        ''' Train Decision Tree using the given 
            X [m x d] data matrix and Y labels matrix
            
            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.
            
            Returns:
            -----------
            Nothing
            '''
        ## now go and train a model for each class...
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        nexamples,nfeatures=X.shape
        ## now go and train a model for each class...
        idx=range(X.shape[0])
        np.random.shuffle(idx)
        
        if(nexamples>1):
                untill=int(nexamples*.70)
        else:
                untill=nexamples 
            
        Xt=X[0:untill]
        Yt=Y[0:untill]
            
        Xt=X[idx]
        Yt=Y[idx]
        
        self.build_tree(Xt,Yt,6)
        return 
        #---------End of Your Code-------------------------#
    
    def build_tree(self, X, Y, depth=10,currNode=None):
            nexamples, nfeatures=X.shape
            klasses=np.unique(Y);
            # YOUR CODE HERE
            #--------calculating purity-------#    
            p_arr=[]
            argmax_flag=False
            for i in klasses:
                    p_arr.append( X[Y==i].shape[0])
            #------------getting the max purity-----------#
            p_idx=np.argmax(p_arr)
            purityD=p_arr[p_idx]/float(nexamples)
            
        
        
            if (nexamples<=self.exthreshold or purityD>=self.purity or self.find_depth()>10):
                print "Tree is Build!.>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                currNode.purity=purityD
                currNode.klasslabel=klasses[p_idx]
       
                currNode.lchild=None
                currNode.rchild=None
                return self.root    
            split_point=0.0
            score=100
            
            Xlidx=[]
            Xridx=[]
            bestFeat=-1
            
            X=X.T
            for feat in range(X.shape[0]):
                #def evaluate_numerical_attribute(self,feat, Y)
                #returns split,mingain,Xlidx,Xridx
                split,mingain_score,tmpXlidx,tmpXridx = self.evaluate_numerical_attribute(X[feat], Y)
                if(mingain_score < score):    
                    split_point=split
                    score=mingain_score
                    Xlidx=tmpXlidx
                    Xridx=tmpXridx
                    bestFeat=feat
            ###now we have the best split point for best fure
            ###split the X into two parts Dy and Dn on basis of split point
            
            DY_exp=X.T[ Xlidx ]
            DN_exp=X.T[ Xridx ]
            Y_l=Y[ Xlidx ]
            Y_r=Y[ Xridx ]
            ###***********************************************************###
            #################################################################
            ###def __init__(self,purity,klasslabel='',score=0,split=[],fidx=-1):
            
            split_store=[ bestFeat,split_point]
            if(self.root==None):
                
                self.root=Node(purityD,'',score,split_store)
                self.root.lchild=Node(-1,'noclass',-1,[-1,-1],-1)
                self.root.rchild=Node(-1,'noclass',-1,[-1,-1],-1)
                
                return self.build_tree(DY_exp,Y_l,10,self.root.lchild)
                return self.build_tree(DN_exp,Y_r,10,self.root.rchild)
                
            else:
                
                currNode.purity=purityD
                currNode.klasslabel=''
                currNode.score=score
                currNode.split=split_store
                currNode.fidx=-1
                
                
                currNode.lchild=Node(-1,'noclass',-1,[-1,-1],-1)
                currNode.rchild=Node(-1,'noclass',-1,[-1,-1],-1)
                
                return self.build_tree(DY_exp,Y_l,10,currNode.lchild)
                return self.build_tree(DN_exp,Y_r,10,currNode.rchild)
                
            ###now make a recursive call for both left and right part 
            ###def build_tree(self, X, Y, depth) return RootNode
            #################################################################

        #---------End of Your Code-------------------------#
        
        
    def gain(self,split_counts,tot_counts):
        H_D=0.0
        tot_n=float(sum(tot_counts))
        
        #--------as we using min entropy so no need to calcilate the max.gain therefore no H(D)---#
        ###now H(Dy,DN)
        H_D_Y=0.0
        tot_N=float(sum(split_counts) )
        
        
        #---now do above in vectrorize---#
        
        y=np.divide(split_counts,tot_N)
        H_D_Y=np.multiply( y ,  np.log2( y + np.spacing(1) ) ).sum()
        
        #----------end this-----------#
        #----------for loop to vectorize-------#
        H_D_N=0.0
        tot_n_sub_N=float(sum(tot_counts-split_counts) )
        idx=tot_counts.size-1
        tot_counts[idx]=tot_counts[idx]+(.064)
        
        f=np.divide(  np.subtract( tot_counts , split_counts) , tot_n_sub_N )
        H_D_N=np.multiply( f ,  np.log2(  f + np.spacing(1)  ) ).sum()
        #--------end this---------#
        P_D_Y=tot_N/float(tot_n)
        P_D_N=1-P_D_Y
        
        H_DY_DN  = P_D_Y * (H_D_Y*-1) +  P_D_N * (H_D_N*-1)
        return H_DY_DN
###################################################################3
    def evaluate_numerical_attribute(self,feat, Y):
        '''
            Evaluates the numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            feat: a contiuous feature
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        # A big source of Bugs will be sorting the same array and expecting it to behave original,
        # use separate variables to store the sorted array and its corresponding classes labels...
        classes=np.unique(Y)
        nclasses=len(classes)
        sidx=np.argsort(feat)
        f=feat[sidx] # sorted features
        sY=Y[sidx] # sorted features class labels...
        
        # YOUR CODE HERE
        ###caculating the midpoints
        Mids_all=np.array([])
        
        #print "f.shape[0]:",f.shape[0]
        
        for i in range(f.shape[0]-1):
            if(f[i]!=f[i+1]):
                Mids_all=np.append( Mids_all ,( f[i]+f[i+1] ) / 2.0 )
                
        idx_r=Mids_all.shape[0]-1
        
        #print "idx_r:",idx_r 
        
        if(idx_r>11):
            idx=np.random.randint(0,idx_r,10 )
            Mids=Mids_all[idx]
            #print "idx:",idx
        
        else:
            #idx=np.random.randint(0,idx_r,idx_r )
            Mids=Mids_all        
        #idx=np.append( idx , Mids_all[idx_r] )
        
        ###now evaluate at each mid point the best split score
        ###and save the midPoint for best split point
        count_arr = [] 
        #print "f.shape:",f.shape
        #print "classes:",classes
        #print "np.unique(sY):",np.unique(sY)
        #print sY.T[0]
        for mid in Mids:
            tmp_arr=np.array([])
            for k in classes:
                #print "klass::", k
                #print "(sY==klass).shape:",(sY[0].T==k).shape
                #print f[sY[0].T==k]
                #print f[f[sY[0].T==k]<=mid]
                tmp_arr=np.append( tmp_arr , f[f[sY[0]==k]<=mid].shape[0] )
            count_arr.append(tmp_arr)
        count_arr=np.array(count_arr)    
        ########################################################
        ###now we will calculate the score at each split
        ###and split point which gives the max score we will store it                       
        split = 0.0
        mingain_score = 1000
        tot_counts=count_arr[count_arr.shape[0]-1]
        for i in range(0,count_arr.shape[0]):
                score=self.gain(count_arr[i],tot_counts)
                if(score<mingain_score):
                               split = Mids[i]
                               mingain_score = score                                   
        Xlidx=f <= split  
        Xridx=f > split               
        return split,mingain_score,Xlidx,Xridx

###########################################################    
    def test(self, X):
        
        ''' Test the trained classifiers on the given set of examples 
        
                   
            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional test examples.
           
            Returns:
            -----------
                pclass: the predicted class for each example, i.e. to which it belongs
        '''
        
        nexamples, nfeatures=X.shape
        pclasses=self.predict(X)
        
        # your code go here...
        
        return np.array(pclasses)
    def predict(self, X):
        
        """
        Test the trained classifiers on the given example X
        
                   
            Input:
            ------
            X: [1 x d] a d-dimensional test example.
           
            Returns:
            -----------
                pclass: the predicted class for the given example, i.e. to which it belongs
        """
        z=[]
        X=np.array(X)
        for idx in range(X.shape[0]):
            
            z.append(self._predict(self.root,np.atleast_2d(X[idx,:])))
        
        return z 
    
    def _predict(self,node, X):
        """
            recursively traverse the tree from root to child and return the child node label
            for the given example X
        """

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        if( node.lchild == None and node.rchild == None):
            return node.klasslabel
        else:
            #print "node.split::",node.split
            bFeat=node.split[0]
            val=node.split[1]
            X=np.array(X)
            #print "X:",X
            #print "X[bFeat]:",X[0][bFeat]
            if( X[0][bFeat] <= val):
                return self._predict(node.lchild,X)
            else:
                return self._predict(node.rchild,X)
        #---------End of Your Code-------------------------#
        
        
        
    def __str__(self):
        """
            overloaded function used by print function for printing the current tree in a
            string format
        """
        str = '---------------------------------------------------'
        str += '\n A Decision Tree With Depth={}'.format(self.find_depth())
        str += self.__print(self.root)
        str += '\n---------------------------------------------------'
        return str  # self.__print(self.tree)        
        
     
    def _print(self, node):
        """
                Recursive function traverse each node and extract each node information
                in a string and finally returns a single string for complete tree for printing purposes
        """
        if not node:
            return
        if node.isleaf():
            return node.get_str()
        
        string = node.get_str() + self._print(node.lchild)
        return string + node.get_str() + self._print(node.rchild)
    
    def find_depth(self):
        """
            returns the depth of the tree...
        """
        return self._find_depth(self.root)
    def _find_depth(self, node):
        """
            recursively traverse the tree to the depth of the tree and return the depth...
        """
        if not node:
            return 0
        if node.isleaf():
            return 1
        else:
            return max(self._find_depth(node.lchild), self._find_depth(node.rchild)) + 1
    def __print(self, node, depth=0):
        """
        
        """
        ret = ""

        # Print right branch
        if node.rchild:
            ret += self.__print(node.rchild, depth + 1)

        # Print own value
        
        ret += "\n" + ("    "*depth) + node.get_str()

        # Print left branch
        if node.lchild:
            ret += self.__print(node.lchild, depth + 1)
        
        return ret         
        