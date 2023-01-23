"""
NUMERICAL ANALYSIS PACKAGE
---
modules:
	draw : classes to draw graphs based on data in input
	iterat(ion) :
		-	indexSplit : from a list of lists,
			returns a list where the the i-th sublists contains, in the same order,
			the i-th element of each starting sublist
		-	index : as indexSplit, but returns a single list
	matrix : 
		-	the following functions build and return special matrixes:
			-	tridiagonal
		-	the following functions return a list of norms, errors, condition number ... of A
    		-	getConds	: 
			-	getRelErrs	:
			-	getNorms	: 
	polynom : 
	prints :
		-	functions printing stuff
			-	matEq	: matrix equation, plus optional specified arrays/numbers to print, if present,
			-	matEq_cholesky	:
			-	matEq_lu	:
	solve : 
"""

