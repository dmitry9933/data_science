import random
import numpy as np
from tqdm import tqdm
import random
from sklearn.model_selection import KFold
from deap import base, creator, tools, algorithms

random.seed()
np.random.seed()

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

class Genetic_Algorithm(object):

    
    def __init__(self, loss_func, estimator, n_pop = 20, n_gen = 20, both = True, n_children = None, 
                 cxpb = 0.5, mutpb = 0.2, cx_indpb = 0.25, mu_indpb = 0.25,
                 algorithm = "one-max", predict_type = 'predict'):
        
        #### check type
        if not hasattr(estimator, 'fit'):
            raise ValueError('Estimator doesn\' have fit method')
        if not hasattr(estimator, 'predict') and not hasattr(estimator, 'predict_proba'):
            raise ValueError('Estimator doesn\' have predict or predict_proba method')
            
        for instant in [cxpb, mutpb, cx_indpb, mu_indpb]:
            if type(instant) != float:
                raise TypeError(f'{instant} should be float type')
            if (instant > 1) or (instant) < 0:
                raise ValueError(f'{instant} should be within range [0,1]')
        
        for instant in [n_pop, n_gen]:
            if type(instant) != int:
                raise TypeError(f'{instant} should be int type')      
        
        if type(both) != bool:
            raise TypeError(f'{both} should be boolean type')
            
        if predict_type not in ['predict', 'predict_proba']:
            raise ValueError('predict_type should be "predict" or "predict_proba"')

        if algorithm not in ['one-max', 'NSGA2']:
            raise ValueError('algorithm should be "one-max" or "NSGA2"')
      
        if not n_children:
            n_children = n_pop

        if type(n_children) != int:
            raise TypeError(f'{n_children} should be int type')
            
        if (cxpb + mutpb) > 1.0:
            raise ValueError(f'The sum of cxpb and mutpb shall be in [0,1]')
        
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.both = both
        self.n_children = n_children
        self.cxpb = cxpb
        self.mutpb = mutpb 
        self.cx_indpb = cx_indpb
        self.mu_indpb = mu_indpb
        self.algorithm = algorithm
        self.loss_func = loss_func
        self.estimator = estimator
        self.predict_type = predict_type
        self.loss_dict = dict()

    def _get_cost(self, X, y, estimator, loss_func, X_test = None, y_test = None):
        
        estimator.fit(X, y.ravel())
        if type(X_test) is np.ndarray:
            if self.predict_type == "predict_proba": # if loss function requires probability
                y_test_pred = estimator.predict_proba(X_test)
                return loss_func(y_test, y_test_pred)
            else:
                y_test_pred = estimator.predict(X_test)
                return loss_func(y_test, y_test_pred)
        
        y_pred = estimator.predict(X)
        
        return loss_func(y, y_pred)
    
    
    def _cross_val(self, X, y, estimator, loss_func, cv):     
        
        loss_record = []
        
        for train_index, test_index in KFold(n_splits = cv).split(X):  # k-fold
            
            try: 
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                estimator.fit(X_train, y_train.ravel())
                
                if self.predict_type == "predict_proba":
                    y_test_pred = estimator.predict_proba(X_test)
                    loss = loss_func(y_test, y_test_pred)         
                    loss_record.append(loss)
                else:
                    y_test_pred = estimator.predict(X_test)
                    loss = loss_func(y_test, y_test_pred)
                    loss_record.append(loss)
            except:
                continue
       
        return np.array(loss_record).mean()    
    
    def _eval_fitness(self, individual):
        
        individual = [True if x else False for x in individual]
        
        if sum(individual) == 0:
            current_loss = np.Inf
        else:       
            encoded_str = ''.join(['1' if x else '0' for x in individual])
            if self.loss_dict.get(encoded_str):
                current_loss = self.loss_dict.get(encoded_str)
            else:
                if self.cv:
                    current_loss = self._cross_val(self.X_train[:,individual], self.y_train, 
                                                   self.estimator, self.loss_func, self.cv)
                    current_loss = np.round(current_loss, 4)
                            
                elif type(self.X_val) is np.ndarray:
                    current_loss = self._get_cost(self.X_train[:,individual], self.y_train, 
                                                     self.estimator, self.loss_func, 
                                                     self.X_val[:,individual], self.y_val)
                    current_loss = np.round(current_loss, 4)
                            
                else:    
                    current_loss = self._get_cost(self.X_train[:,individual], self.y_train, 
                                                  self.estimator, self.loss_func, None, None)   
                    current_loss = np.round(current_loss, 4)
                self.loss_dict[encoded_str] = current_loss
                    
        if self.algorithm == "one-max":
            return current_loss,
        else:
            return current_loss, sum(individual)

    def fit(self, X_train, y_train, cv = None, X_val = None, y_val = None, 
            init_sol = None, stop_point = 5):
     
        
        # make sure input has two dimensions
        assert len(X_train.shape) == 2
        num_feature = X_train.shape[1]
        
        # save them for _eval_fitness function
        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv
        self.X_val = X_val
        self.y_val = y_val
        
        # creator
        if self.algorithm == "one-max":
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # minimize the loss
            creator.create("Individual", list, fitness=creator.FitnessMin)
        else:
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -0.1))
            creator.create("Individual", list, fitness=creator.FitnessMulti)            
        
        # register
        toolbox = base.Toolbox()
        toolbox.register("gene", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                         toolbox.gene, n = num_feature)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, 
                         n = self.n_pop)
        toolbox.register("evaluate", self._eval_fitness)
        toolbox.register("mate", tools.cxUniform, indpb = self.cx_indpb)
        toolbox.register("mutate", tools.mutFlipBit, indpb = self.mu_indpb)
        
        if self.algorithm == "one-max":
            toolbox.register("select", tools.selTournament, tournsize=5)
        else:
            toolbox.register("select", tools.selNSGA2)

        # start evolution
        # evaluate inital population
        population = toolbox.population()
        fits = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fits):
            ind.fitness.values = fit
        
        # evolving
        for gen in tqdm(range(self.n_gen)):
            if self.both:
                offspring = algorithms.varOr(population, toolbox, 
                                              lambda_ = self.n_children, cxpb = self.cxpb,
                                              mutpb = self.mutpb)
            else:
                offspring = algorithms.varAnd(population, toolbox, cxpb = self.cxpb,
                                              mutpb = self.mutpb)  
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            if self.algorithm == 'one-max':
                population = toolbox.select(offspring, k = self.n_pop)
            else:
                population = toolbox.select(offspring + population, k = self.n_pop)
        
        fits = list(toolbox.map(toolbox.evaluate, population))
        if self.algorithm != "one-max":
            fits = [x[0] for x in fits]
        
        try:
            best_idx = np.argmin(np.array(fits))
            self.best_sol = [True if x else False for x in population[best_idx]]
            self.best_loss = fits[best_idx]  
            
            if np.isinf(self.best_loss): # if best loss is inf
                best_key = min([(value, key) for key, value in self.loss_dict.items()])[1]
                self.best_sol = [True if x == '1' else False for x in best_key]
                self.best_loss = min([(value, key) for key, value in self.loss_dict.items()])[0]   
        except:
            best_key = min([(value, key) for key, value in self.loss_dict.items()])[1]
            self.best_sol = [True if x == '1' else False for x in best_key]
            self.best_loss = min([(value, key) for key, value in self.loss_dict.items()])[0]
 
    def transform(self, X):

        transform_X = X[:, self.best_sol]
        return transform_X