remove(list=ls())
setwd(dir = "C:\\Users\\skoci\\Documents\\psp_dust")
myfile_solo = 'data_synced\\solo_flux_readable.csv' 
myfile_psp = 'data_synced\\psp_flux_readable.csv'

#install.packages("INLA",repos=c(getOption("repos"),INLA="https://inla.r-inla-download.org/R/stable"), dep=TRUE)
#install.packages("distr")

library(INLA)
library(hexbin)
require(hexbin)
require(lattice)
library(distr)
#require(RColorBrewer)
#library(vioplot)



###################################
### DEFINE MODEL with R-generic ###
###################################

three_component_model <- function(cmd = c("graph", "Q", "mu", "initial", 
                                          "log.norm.const", "log.prior", "quit",
                                          "rate", 
                                          "prior.l_a",
                                          "prior.l_b",
                                          "prior.ddz_a",
                                          "prior.ddz_b"), 
                                  theta=NULL, feed_x=NULL){

  envir <- parent.env(environment())
  prec.high = exp(15)
  
  prior.l_a <- function(l_a=feed_x){
    return(dgamma(l_a, shape = 2,    scale = 1e-4, log=TRUE))
  }
  prior.l_b <- function(l_b=feed_x){
    return(dgamma(l_b,   shape = 2,    scale = 1e-4, log=TRUE))
  }
  prior.ddz_a <- function(ddz_a=feed_x){
    return(dgamma(ddz_a,   shape = 2,    scale = 0.05, log=TRUE))
  }
  prior.ddz_b <- function(ddz_b=feed_x){
    return(dgamma(ddz_b,   shape = 2,    scale = 0.05, log=TRUE))
  }
  
  rate <- function(#covariates
                   v_sc_r, v_sc_t, r_sc, v_sc_x, v_sc_y, v_sc_z, area,  
                   #hyperparam
                   l_a, l_b, ddz_a, ddz_b,     
                   #bound dust parameters
                   e_a_r=-1.3, e_a_v=4.15,
                   #beta met. parameters
                   v_b_a=9, v_b_r=56, e_b_r=-1.6, e_b_v=2.0,
                   #stationary Earth
                   v_earth_a=0){            
    
    deg2rad <- function(deg){
      rad = deg/180*pi
      return(rad)
    }
    
    #bound dust contribution
    ksi = -2 - e_a_r
    #ddz_factor_a = min(1,r_sc/ddz_a)
    ddz_factor_a = 1#(r_sc/ddz_a + (1-r_sc/ddz_a)*exp(-ddz_a/r_sc))/2
    r_factor = r_sc/1
    v_a_a = 29.8*(r_factor^(-1))
    v_factor = ( (
      ( v_sc_r )^2 
      + ( v_sc_t - v_a_a )^2
    )^0.5 
    ) / abs( v_earth_a - v_a_a )
    L_a = l_a * ddz_factor_a * (v_factor)^e_a_v * (r_factor)^e_a_r
    
    #beta meteoroid contribution
    ksi = -2 - e_b_r
    #ddz_factor_b = min(1,r_sc/ddz_b)
    ddz_factor_b = (r_sc/ddz_b + (1-r_sc/ddz_b)*exp(-ddz_b/r_sc))/2
    r_factor = r_sc/1
    v_factor = ( (
      ( v_sc_r - ( v_b_r*(r_factor^ksi)  ) )^2 
      + ( v_sc_t - ( v_b_a*(r_factor^(-1)) ) )^2
    )^0.5 
    ) / ( (
      ( v_b_r )^2 
      + ( v_earth_a - v_b_a )^2
    )^0.5 
    )
    L_b = l_b * ddz_factor_b * (v_factor)^e_b_v * (r_factor)^e_b_r 
    
    #normalization to hourly rate, while L_i are in m^-2 s^-1
    hourly_rate = 3600 * area * ( L_b + L_a )
    return(hourly_rate)
  }
  
  
  
  interpret.theta <- function(){
    return(list(l_a     = exp(theta[1L]),
                l_b     = exp(theta[2L]),
                ddz_a   = exp(theta[3L]),
                ddz_b   = exp(theta[4L])
               ))
  }
  
  graph <-function(){
    G <- Diagonal(n = length(vt), x=1)
    return(G)
  }
  
  Q <- function(){
    #prec.high <- interpret.theta()$prec
    Q <- prec.high*graph()
    return(Q)
  }
  
  mu <- function(){
    par = interpret.theta()
    return(log( rate(#covariates
                     vr, vt, r, vx, vy, vz, area,
                     #hyperparameters
                     par$l_a, 
                     par$l_b, 
                     par$ddz_a, 
                     par$ddz_b
                     )
              ))
  }
  
  log.norm.const <-function(){
    return(numeric(0))
  }
  
  # Log-prior for thetas
  log.prior <- function(){
    par = interpret.theta()
    
    #nice priors
    val <- (prior.l_a(    par$l_a)   + theta[1L] +
            prior.l_b(    par$l_b)   + theta[2L] +
            prior.ddz_a(  par$l_a)   + theta[3L] +
            prior.ddz_b(  par$l_b)   + theta[4L] 
           )
  
    return(val)
  }
  
  # Initial values of theta
  initial <- function(){
    #initial values set to the maxima a priori
    return(c(log(optimize(prior.l_a,   interval = c(0, 1e-2), maximum = TRUE, tol=1e-9)$maximum),
             log(optimize(prior.l_b,   interval = c(0, 1e-2), maximum = TRUE, tol=1e-9)$maximum),
             log(optimize(prior.ddz_a, interval = c(0, 1),    maximum = TRUE, tol=1e-5)$maximum),
             log(optimize(prior.ddz_b, interval = c(0, 1),    maximum = TRUE, tol=1e-5)$maximum)
            )
          )
  }
  
  quit <-function(){
    return(invisible())
  }
  
  val <- do.call(match.arg(cmd), args = list())
  return(val)
}



###################################
########## Load the data ##########
###################################

mydata_solo = read.csv(file = myfile_solo)
names(mydata_solo)[c(2,3,4,5,6,9,10,11)] = c("flux",
                                             "vr",
                                             "vt",
                                             "r",
                                             "exposure",
                                             "vx",
                                             "vy",
                                             "vz")
mydata_solo$area = 8
mydata_solo$sc_id = 1

mydata_psp = read.csv(file = myfile_psp)
names(mydata_psp)[c(2,3,4,5,6,9,10,11)] = c("flux",
                                            "vr",
                                            "vt",
                                            "r",
                                            "exposure",
                                            "vx",
                                            "vy",
                                            "vz")
mydata_psp$area = 6
mydata_psp$sc_id = 2

mydata = rbind(mydata_solo,mydata_psp)

n = length(mydata$vr)
mydata$idx = 1:n 



###################################
########## Run the model ##########
###################################


rgen = inla.rgeneric.define(model = three_component_model, 
                            vr = mydata$vr, 
                            vt = mydata$vt, 
                            r  = mydata$r,
                            vx = mydata$vx,
                            vy = mydata$vy,
                            vz = mydata$vz,
                            area = mydata$area)
result = inla(flux ~ -1 + f(idx, model = rgen) + f(sc_id, model = "iid"),
              data = mydata, family = "poisson", E = exposure, 
              control.compute = list(cpo=TRUE, dic=TRUE, config = TRUE),
              safe = TRUE, verbose = TRUE)

summary(result)

hist(result$cpo$pit)     # ok 
result$cpo$failure       # also OK
pit = result$cpo$pit
#save(pit, file = "998_generated\\inla\\pit.RData")

#plotting
par(mfrow = c(1, 1))
plot(mydata$flux/mydata$exposure, ylab="counts/E")
lines(result$summary.fitted.values$mean, col=2, lwd=3)
#lines(30+mydata$flux/mydata$exposure-result$summary.fitted.values$mean, col="blue")

span = round(max(abs(mydata$flux/mydata$exposure-result$summary.fitted.values$mean),40)+0.5)
hist(mydata$flux/mydata$exposure-result$summary.fitted.values$mean,
     breaks=c(-span:span),
     main="")
mtext(paste("residuals histogram, stdev = ",
            as.character(sqrt(var(mydata$flux/mydata$exposure-result$summary.fitted.values$mean))),
            ", log(mlik) = ",
            result$mlik[1]), side=3)
  
# Posterior means of the hyperparameters
inla.emarginal(function(x) exp(x), result$marginals.hyperpar$`Theta1 for idx`)
inla.emarginal(function(x) exp(x), result$marginals.hyperpar$`Theta2 for idx`)
inla.emarginal(function(x) exp(x), result$marginals.hyperpar$`Theta3 for idx`)
inla.emarginal(function(x) exp(x), result$marginals.hyperpar$`Theta4 for idx`)

# Create a layout with one column and four rows
par(mfrow = c(4, 1), mar = c(2, 2, 1, 1))
# Plot each function in a separate row
plot(exp(result$marginals.hyperpar$`Theta1 for idx`[1:43]),result$marginals.hyperpar$`Theta1 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "l_a", col = "red", cex = 1.5)
plot(exp(result$marginals.hyperpar$`Theta2 for idx`[1:43]),result$marginals.hyperpar$`Theta2 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "l_b", col = "red", cex = 1.5)
plot(exp(result$marginals.hyperpar$`Theta3 for idx`[1:43]),result$marginals.hyperpar$`Theta3 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "ddz_a", col = "red", cex = 1.5)
plot(exp(result$marginals.hyperpar$`Theta4 for idx`[1:43]),result$marginals.hyperpar$`Theta4 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "ddz_b", col = "red", cex = 1.5)
# Reset the layout to the default (1x1)
par(mfrow = c(1, 1))



