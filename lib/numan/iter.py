

# takes a container (lists) of containers
# returns a list of i lists, each sublist containing the i-th element of each sub-container
# the return list has the same length as the shortest container
# (zip)
def indexSplit(lists):
	ln = min([len(subl) for subl in lists])
	return [[subl[i] for subl in lists] for i in range(ln)]

# takes a container (lists) of containers
# returns a 1-dimensional list, containing the i-th element of each sub-container, followed by the i+1-th of each
# the return list has the same length as the shortest container
def index(lists):
	return sum(indexSplit(lists), [])