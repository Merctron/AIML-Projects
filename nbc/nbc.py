# Naive Bayes Classifier Implementation for a yelp data set
# Noramlization: P(1|X) / ( P(1|X)+P(0|X) ) and P(0|X) / ( P(1|X)+P(0|X) )

from __future__ import division
import csv
import sys
import math
import random
import numpy

# Step 1: Data import mechanism


# ------------------- Old code

contAttr = [3, 4, 6, 7]
discAttr = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

def getYelpData(fileName):
	lines = csv.reader(open(fileName, "rb"))
	dataSet = list(lines)
	return dataSet

def getGFGData(dataSet, classVal):
	toReturn = []
	for i in range(len(dataSet)):
		if (dataSet[i][0] == classVal):
			toReturn.append(dataSet[i])
	return toReturn

def getTrGFGData(dataSet, classVal1, classVal2):
	toReturn = []
	for i in range(len(dataSet)):
		if (dataSet[i][0] == classVal1 or dataSet[i][0] == classVal2):
			toReturn.append(dataSet[i])
	return toReturn

def buildNBCParams(dataSet):
	paramSet = {}
	for i in range(len(dataSet[0])):
		paramArr = {}
		for j in range(len(dataSet)):
			vector = dataSet[j]
			if (vector[i] not in paramArr):
				paramArr[vector[i]] = []
			paramArr[vector[i]].append(vector)
		paramSet[i] = paramArr
	return paramSet

def buildBaseParams(dataSet):
	params = {}
	for i in range(len(dataSet[0])):
		paramArr = {}
		for j in range(1, len(dataSet)):
			vector = dataSet[j]
			if (vector[i] not in paramArr):
				paramArr[vector[i]] = []
			paramArr[vector[i]].append(vector)
		params[i] = paramArr
	consideredAtt = 0
	finalParams = {}
	for i in range(len(params)):
		#print('------------------------------------------------')
		#print(len(params[i]))
		if (i in discAttr):
			finalParams[i] = {}
			totalParamCount = 0
			for key in params[i]:
				totalParamCount = totalParamCount + len(params[i][key])
			consideredAtt = consideredAtt + 1
			for key in params[i]:
				finalParams[i][key] = len(params[i][key])/totalParamCount
				#print('Attribute Num: {0}, Attribute Value: {1}, Val Count: {2}').format(i, key, len(params[i][key]))

	#print()
	#print()
	#print('Total attributes considered: {0}').format(consideredAtt)
	#print(finalParams)
	return finalParams

def getClassParam(dataSet, classVal):
	param = len(getGFGData(dataSet, classVal)) / len(dataSet)
	return param

def trainNBCForClass(training_data, classVar):
	#training_data = getYelpData(training_data_file)
	sub_training_data = getGFGData(training_data, classVar)
	params = buildNBCParams(sub_training_data)
	consideredAtt = 0
	finalParams = {}
	for i in range(len(params)):
		#print('------------------------------------------------')
		#print(len(params[i]))
		if (i in discAttr):
			finalParams[i] = {}
			totalParamCount = 0
			for key in params[i]:
				totalParamCount = totalParamCount + len(params[i][key])
			consideredAtt = consideredAtt + 1
			for key in params[i]:
				finalParams[i][key] = len(params[i][key])/totalParamCount
				#print('Attribute Num: {0}, Attribute Value: {1}, Val Count: {2}').format(i, key, len(params[i][key]))

	#print()
	#print()
	#print('Total attributes considered: {0}').format(consideredAtt)
	#print(finalParams)
	return finalParams

def testNBC(tDataSet, clParam1, clParam0, trParams, invTrParams, bsParams):
	dataSet = getTrGFGData(tDataSet, '1', '0');
	numSuc = 0
	for i in range(1, len(dataSet)):
		prob1 = clParam1
		prob0 = clParam0
		for j in range(1, len(dataSet[i])):
			if (j in trParams):
				if (dataSet[i][j] in trParams[j]):
					prob1 = prob1 * trParams[j][dataSet[i][j]] #/ bsParams[j][dataSet[i][j]]
				else:
					prob1 = prob1 * (1 / len(trParams[j]))
				if (dataSet[i][j] in invTrParams[j]):
					prob0 = prob0 * invTrParams[j][dataSet[i][j]] #/ bsParams[j][dataSet[i][j]]
				else:
					prob0 = prob0 * (1 / len(invTrParams[j]))
		totalProb = prob1 + prob0
		prob1 = prob1/totalProb
		prob0 = prob0/totalProb
		if ((dataSet[i][0] == '1') and (prob1 >= 0.5)):
			numSuc = numSuc + 1
			#print('Success Detected, Prob: {0}').format(prob1)
		elif ((dataSet[i][0] == '0') and (prob0 >= 0.5)):
			numSuc = numSuc + 1
			
		#print('Class: {0}, Prob for Class 1: {1}, Prob for Class 0: {2}, Total Prob: {3}').format(dataSet[i][0], prob1, prob0, prob1 + prob0)
	return numSuc

def getSquaredLoss(dataSet, clParam1, clParam0, trParams, invTrParams, bsParams):
	numSuc = 0
	for i in range(1, len(dataSet)):
		prob1 = clParam1
		prob0 = clParam0
		for j in range(1, len(dataSet[i])):
			if (j in trParams):
				if (dataSet[i][j] in trParams[j]):
					prob1 = prob1 * trParams[j][dataSet[i][j]] #/ bsParams[j][dataSet[i][j]]
				else:
					prob1 = prob1 * (1 / len(trParams[j]))
				if (dataSet[i][j] in invTrParams[j]):
					prob0 = prob0 * invTrParams[j][dataSet[i][j]] #/ bsParams[j][dataSet[i][j]]
				else:
					prob0 = prob0 * (1 / len(invTrParams[j]))
		totalProb = prob1 + prob0
		prob1 = prob1/totalProb
		prob0 = prob0/totalProb
		if ((dataSet[i][0] == '1')):
			numSuc = numSuc + math.pow((1 - prob1), 2)
			#print('Success Detected, Prob: {0}').format(prob)
		elif ((dataSet[i][0] == '0')):
			numSuc = numSuc + math.pow((1 - prob0), 2)
			#print('Success Detected 2, Prob: {0}').format(prob)
	numSuc = numSuc / (len(dataSet) - 1)
	return numSuc

def getClassForVector(vector):
	cl = 0
	return cl

def getDataSample(dataSet, sample_size):
	sampleSet = random.sample(xrange(len(dataSet)), sample_size)
	return sampleSet

def eucDist(pt1, pt2):
	sum = 0
	for i in range(len(contAttr)):
		#print("{0}, {1}").format(pt1[i], pt2[i])
		sum = sum + math.pow((pt1[i] - pt2[i]), 2)
	toReturn = math.sqrt(sum)
	return toReturn

def buildParamSet(dataSet):
	paramSet = []
	for i in range(1, len(dataSet)):
		paramSet.append([])
		paramSet[i-1].append(float(dataSet[i][3]))
		paramSet[i-1].append(float(dataSet[i][4]))
		paramSet[i-1].append(float(dataSet[i][6]))
		paramSet[i-1].append(float(dataSet[i][7]))
	return paramSet

def logTransformParamSet(paramSet):
	for i in range (len(paramSet)):
		paramSet[i][2] = math.log10(paramSet[i][2])
		paramSet[i][3] = math.log10(paramSet[i][3])
	return paramSet

def logTransformClusterAnalysis(dataSet, k):
	paramSet = logTransformParamSet(buildParamSet(dataSet))
	indices = []
	means = []
	clusters = []
	tempClusters = []
	changed = True
	changedCt = 0

	ct = 0

	while (ct < k):
		randInt = random.randint(0, len(paramSet))
		if randInt not in indices:
			indices.append(randInt)
			clusters.append([])
			ct = ct + 1


	for i in range(len(indices)):
		means.append(paramSet[indices[i]])

	# Build initial clusters with random mean seeds

	for i in range(len(paramSet)):
		clusterIndex = 0
		pastDist = sys.maxint
		for j in range(len(means)):
			dist = eucDist(means[j], paramSet[i])
			if dist < pastDist:
				clusterIndex = j
				pastDist = dist
		clusters[clusterIndex].append(i)

	
	# Exit if no change in clustering
	while (changed):
		changedCt = changedCt + 1
		changed = False
		# Calculate new cluster means

		for i in range(len(means)):
			meanSum = []
			for k in range(len(means)):
				meanSum.append([0, 0, 0, 0])
			for j in range(len(clusters[i])):
				meanSum[i][0] = meanSum[i][0] + paramSet[clusters[i][j]][0]
				meanSum[i][1] = meanSum[i][1] + paramSet[clusters[i][j]][1]
				meanSum[i][2] = meanSum[i][2] + paramSet[clusters[i][j]][2]
				meanSum[i][3] = meanSum[i][3] + paramSet[clusters[i][j]][3]
			
			means[i][0] = meanSum[i][0]/len(clusters[i])
			means[i][1] = meanSum[i][1]/len(clusters[i])
			means[i][2] = meanSum[i][2]/len(clusters[i])
			means[i][3] = meanSum[i][3]/len(clusters[i])

		# Empty temp clusters
		tempClusters = []
		for i in range(len(clusters)):
			tempClusters.append([])

		# Reassign datapoints and check for changes. If change detected, repeat

		for i in range(len(paramSet)):
			clusterIndex = 0
			pastDist = sys.maxint
			for j in range(len(means)):
				dist = eucDist(means[j], paramSet[i])
				if dist < pastDist:
					clusterIndex = j
					pastDist = dist
			if i not in clusters[clusterIndex]:
				changed = True
			tempClusters[clusterIndex].append(i)

		# replace clusters with newly calculated clusters

		for i in range(len(clusters)):
			clusters[i] = tempClusters[i]

	# Return means computed for k clusters

	#wcScore = calcScore(k, clusters, means, paramSet)
	#print("Changed Count: {0}").format(changedCt)
	#print("WC-SSE={0}").format(wcScore)

	#for i in range(len(means)):
	#	print("Centroid{0}=[{1}, {2}, {3}, {4}]").format(i + 1, means[i][0], means[i][1], means[i][2], means[i][3])

	#return means
	return clusters

def printParams(params):
	for i in params:
		print('Attr: {0} -------------------------------').format(i)
		for key in params[i]:
			print('Attribute Num: {0}, Attribute Value: {1}, Prob Val: {2}').format(i, key, params[i][key])

training_data_file = 'training.csv'
testing_data_file = 'validation.csv'

numArgs = len(sys.argv)
if (numArgs <=1):
	none = 0
	#print('No file arguments detected. Attempting to use default files: training.csv, validation.csv')
else :
	training_data_file = sys.argv[1]
	testing_data_file = sys.argv[2]
	#print('Arguments accepted: {0}, {1}').format(training_data_file, testing_data_file)

#baseParams = buildBaseParams(getYelpData(testing_data_file))
#trainedParams = trainNBCForClass(getYelpData(training_data_file), '1')
#invtrainedParams = trainNBCForClass(getYelpData(training_data_file), '0')
#classParam1 = getClassParam(getYelpData(training_data_file), '1')
#classParam0 = getClassParam(getYelpData(training_data_file), '0')

#numSuc = testNBC(getYelpData(testing_data_file), classParam1, classParam0, trainedParams, invtrainedParams, baseParams)
#print('Class param 1: {0}, class param 2: {1}, total: {2}').format(classParam1, classParam0, classParam1 + classParam0)
#print('Number of NBC successes: {0}, Total instances: {1}').format(numSuc, len(getYelpData(testing_data_file)) - 1)
#print('ZERO-ONE LOSS={0}').format(((len(getYelpData(testing_data_file)) - 1) - numSuc)/(len(getYelpData(testing_data_file)) - 1))
#print('SQUARED LOSS={0}').format(getSquaredLoss(getYelpData(testing_data_file), classParam1, classParam0, trainedParams, invtrainedParams, baseParams))

#printParams(baseParams)
#printParams(trainedParams)
#printParams(invtrainedParams)

#print(getDataSample(getYelpData(training_data_file), 10))
samplingPercentages = [0.002, 0.02, 0.2] 
for i in range(len(samplingPercentages)):
	zrMeanTot = 0
	sqMeanTot = 0
	for m in range(10):
		fullSet = getYelpData(training_data_file)

		# Augment data with cluster attributes
		# kclusters = logTransformClusterAnalysis(fullSet, 4)
		# for p in range(len(kclusters)):
		# 	for q in range(len(kclusters[p])):
		# 		fullSet[kclusters[p][q] + 1].append(p) #adding 1 to index from cluster because of removal of the first row

		# for p in range(len(fullSet)):
		# 	if len(fullSet[p]) < 20:
		# 		fullSet[p].append(4)


		sampleSize = int(math.ceil(len(fullSet) * samplingPercentages[i]))
		#print('SAMPLE SIZE: {0}').format(sampleSize)
		sampleSetIndices = getDataSample(fullSet, sampleSize)
		sampleSet = []
		for j in range(len(sampleSetIndices)):
			sampleSet.append(fullSet[sampleSetIndices[j]])
		baseParams = buildBaseParams(sampleSet)
		trainedParams = trainNBCForClass(sampleSet, '1')
		invtrainedParams = trainNBCForClass(sampleSet, '0')
		classParam1 = getClassParam(sampleSet, '1')
		classParam0 = getClassParam(sampleSet, '0')
		testSet = []
		for k in range(len(fullSet)):
			if j not in sampleSetIndices:
				testSet.append(fullSet[k])
		numSuc = testNBC(testSet, classParam1, classParam0, trainedParams, invtrainedParams, baseParams)
		zrMeanTot = zrMeanTot + (((len(testSet) - 1) - numSuc)/(len(testSet) - 1))
		sqMeanTot = sqMeanTot + getSquaredLoss(testSet, classParam1, classParam0, trainedParams, invtrainedParams, baseParams)
		#print('Class param 1: {0}, class param 2: {1}, total: {2}').format(classParam1, classParam0, classParam1 + classParam0)
		#print('Number of NBC successes: {0}, Total instances: {1}').format(numSuc, len(testSet) - 1)
		#print('ZERO-ONE LOSS={0}').format(((len(testSet) - 1) - numSuc)/(len(testSet) - 1))
		#print('SQUARED LOSS={0}').format(getSquaredLoss(testSet, classParam1, classParam0, trainedParams, invtrainedParams, baseParams))
	zrMean = zrMeanTot / 10
	sqMean = sqMeanTot / 10
	print('Sampling %: {0}, Zero-One Loss: {1}, Squared Loss: {2}').format(samplingPercentages[i], zrMean, sqMean)


#stuff = buildNBCParams(getYelpData(training_data_file)) + buildNBCParams(getYelpData(testing_data_file))
#for x in stuff:
#	print(len(stuff[x]))
