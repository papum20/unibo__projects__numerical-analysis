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
	methods :
		-	matIt_* : iteration matrix
		-	the following calculate solutions with their iterative method:
			-	jacobi	:
			-	gaussSiedel	:
	polynom : 
	prints :
		-	functions printing stuff
			-	matEq	: matrix equation, plus optional specified arrays/numbers to print, if present,
			-	matEq_cholesky	:
			-	matEq_lu	:
	solve : 
"""




#prints
ORDS = ["1", "2", "fro", "inf"]
FIGSIZE = (15,7)
DPI = 100
FONTSIZE = 7
STEPS = 300

#poly
FLOAT_MANT_MIN = 1.e-16		#min mantissa number
TOL_X = 1.e-7
TOL_Y = 1.e-16

#polyrn
ALPHA_0 = 1.1
C1 = 0.25
MAXIT = 1000
MAXIT_ALPHA = 10
RHO = 0.5
TOL_ALPHA = 1.e-16
TOL_DF = 1.e-5