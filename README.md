# SparseFrechet

This paper focuses on the sparse Fr\'{e}chet problem where the predictor dimension is much larger than the sample size. 
To estimate the central subspace, a multi-task regression is constructed using artificial response variables from the leading eigenvectors of a weighted inverse regression ensemble matrix. 
We incorporate a minimax concave penalty to the constructed multi-task regression to eliminate estimation biases, further improving variable selection. 
To solve the nonconvex optimization problem, we propose a novel local double approximation algorithm, which approximates the loss function and the penalty term, respectively, resulting in explicit expressions in each iteration. 

