library(R1magic)
            N <- 100 ;#  Signal
            K <- 4  ;#  Sparsity
            ;#  Up to Measurements  > K LOG (N/K)
            M <- 40
            ;# Measurement Matrix (Random Sampling Sampling)
            phi <- GaussianMatrix(N,M)
            ;# R1magic generate random signal
            xorg <- sparseSignal(N, K, nlev=1e-3)
            y <- phi %*% xorg ;# generate measurement
            T <- diag(N) ;# Do identity transform
            p <- matrix(0, N, 1) ;# initial guess
            ;# R1magic Convex Minimization ! (unoptimized-parameter)
            ll <- solveL1(phi, y, T, p)
            x1 <- ll$estimate
            plot( 1:100, seq(0.011,1.1,0.011), type = "n",xlab="",ylab="")
            title(main="Random Sparse Signal Recovery",
                  xlab="Signal Component",ylab="Spike Value")
            lines(1:100, xorg , col = "red")
            lines(1:100, x1, col = "blue", cex = 1.5) ;# shifted by 5 for clearity

