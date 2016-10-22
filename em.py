import pandas as pd
import numpy as np
import random
#from scipy.stats import norm
import math

#return normal distribution for x
#def N(x, M, D):
#    return norm.cdf(x, M, D)

def N(x, M, D):
    return math.e ** ( - ((x - M) ** 2) / (2 * D)) / math.sqrt(2 * D * math.pi)

#return P(sex)
def P(alpha, pm, pf):
    return (alpha * pm) / (alpha * pm + (1 - alpha) * pf)

#return 1 and 2 median
def M(df):
	return df[df['Male'] == True].height.median(), df[df['Male'] == False].height.median()

#return alphas
def alphas(df):
	alpha_m = df[df['Male'] == True].shape[0] / float(selection_len)
	alpha_f = 1 - alpha_m
	return alpha_m, alpha_f

#return normal distribution for height columns (male and female median)
def p(df):
	df['pm'] = df.height.apply(lambda x: N(x, Mm, D))
	df['pf'] = df.height.apply(lambda x: N(x, Mf, D))
	return

#return sex for every object in column height (True if Male)
def Male(df):
	df['Male'] = df[['pm', 'pf']].apply(lambda x: P(alpha_m, x['pm'], x['pf']) >= random.random(), axis=1)

def alpha_search(df, alpha_m, alpha_f):
	p(df)
	Male(df)
	alpha_m, alpha_f = alphas(df)
	rMm, rMf = M(df)
	print df.head(5)
	return alpha_m, alpha_f, rMm, rMf

def alpha_test(alpha_row):
	try:
		if len(alpha_row) < 10:
			return False
		eps_test = abs(np.mean(alpha_row[int(len(alpha_row)/2):]) - np.mean(alpha_row[:int(len(alpha_row)/2)]))
		print(str(eps_test) + ' -eps_test')
		print 'Mm:' + str(Mm) + ' Mf:' + str(Mf) + ' D:' + str(D) + ' alpha_m:' + str(alpha_m)# + ' alpha_f:' + str(alpha_f)
		return eps_test <= eps
	except:
		return False




#read csv
df = pd.read_csv('dataset.csv', header=0, sep=' ')
del df['Unnamed: 1']

#get selection len
selection_len = len(df.values)

#get dispersion and eps
D = (4.785030177866893)**2 #df.height.std()**2
eps = 0.2

#get first step parameters
Mm = df.height.median()
Mf = df[(df['height'] > Mm) | (df['height'] < Mm)].height.median()
alpha_m = 0.39
alpha_f = 1 - alpha_m
alpha_row = np.array([alpha_m])


while not alpha_test(alpha_row):
	alpha_m, alpha_f, Mm, Mf = alpha_search(df, alpha_m, alpha_f)
	alpha_row = np.append(alpha_row, alpha_m)

print('final_alpha: ' + str(alpha_m))


