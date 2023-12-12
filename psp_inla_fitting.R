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
                                          "prior.l_bg",
                                          "prior.l_a",
                                          "prior.l_b",
                                          "prior.v_b_r",
                                          "prior.e_v",
                                          "prior.e_a_r",
                                          "prior.e_b_r"), 
                                  theta=NULL, feed_x=NULL){

  envir <- parent.env(environment())
  prec.high = exp(15)
  
  prior.l_bg <- function(l_bg=feed_x){
    return(dgamma(l_bg,  shape = 2,    scale = 1e-8, log=TRUE))
  }
  prior.l_a <- function(l_a=feed_x){
    return(dgamma(l_a, shape = 2,    scale = 1e-5, log=TRUE))
  }
  prior.l_b <- function(l_b=feed_x){
    return(dgamma(l_b,   shape = 2,    scale = 1e-5, log=TRUE))
  }
  prior.v_b_r <- function(v_b_r=feed_x){
    return(dnorm(v_b_r,  mean  = 60,    sd   = 0.0005,    log=TRUE))
  }
  prior.e_v <- function(e_v=feed_x){
    return(dnorm(e_v,    mean  = 2.2,   sd   = 0.05, log=TRUE))
  }
  prior.e_a_r <- function(e_a_r=feed_x){
    return(dnorm(e_a_r,    mean  = -1.65, sd   = 0.5, log=TRUE))
  }
  prior.e_b_r <- function(e_b_r=feed_x){
    return(dnorm(e_b_r,    mean  = -1.65, sd   = 0.005, log=TRUE))
  }
  
  rate <- function(#covariates
                   v_sc_r, v_sc_t, r_sc, v_sc_x, v_sc_y, v_sc_z, area,  
                   #hyperparam
                   l_bg, l_a, l_b, v_b_r, e_v, e_a_r, e_b_r,      
                   #bound dust parameters
 
                   #beta met. parameters
                   v_b_a=9, 
                   #stationary Earth
                   v_earth_a=0){            
    
    deg2rad <- function(deg){
      rad = deg/180*pi
      return(rad)
    }
    
    #background
    L_bg = l_bg
    
    #beta meteoroid contribution
    ksi = -2 - e_b_r
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
    L_b = l_b * (v_factor)^e_v * (r_factor)^e_b_r 
    
    #bound dust contribution
    ksi = -2 - e_a_r
    r_factor = r_sc/1
    v_a_a = 29.8*(r_factor^(-1))
    v_factor = ( (
      ( v_sc_r )^2 
      + ( v_sc_t - v_a_a )^2
    )^0.5 
    ) / abs( v_earth_a - v_a_a )
    L_a = l_a * (v_factor)^e_v * (r_factor)^e_a_r
    
    #normalization to hourly rate, while L_i are in m^-2 s^-1
    hourly_rate = 3600 * area * ( L_bg + L_b + L_a )
    return(hourly_rate)
  }
  
  
  
  interpret.theta <- function(){
    return(list(l_bg  = exp(theta[1L]), 
                l_a   = exp(theta[2L]),
                l_b   = exp(theta[3L]),
                v_b_r = theta[4L],
                e_v   = theta[5L], 
                e_a_r = theta[6L],
                e_b_r = theta[7L]
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
                     par$l_bg, 
                     par$l_a, 
                     par$l_b, 
                     par$v_b_r, 
                     par$e_v, 
                     par$e_a_r,
                     par$e_b_r
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
    val <- (prior.l_bg(  par$l_bg)    + theta[1L] +
            prior.l_a(   par$l_a)     + theta[2L] +
            prior.l_b(   par$l_b)     + theta[3L] +
            prior.v_b_r( par$v_b_r)   +
            prior.e_v(   par$e_v)     + 
            prior.e_a_r( par$e_a_r)   +
            prior.e_b_r( par$e_b_r)
           )
  
    return(val)
  }
  
  # Initial values of theta
  initial <- function(){
    #initial values set to the maxima a priori
    return(c(log(optimize(prior.l_bg, interval = c(0, 1e-2), maximum = TRUE, tol=1e-9)$maximum),
             log(optimize(prior.l_a, interval = c(0, 1e-2), maximum = TRUE, tol=1e-9)$maximum),
             log(optimize(prior.l_b, interval = c(0, 1e-2), maximum = TRUE, tol=1e-9)$maximum),
             optimize(prior.v_b_r, interval = c(0, 1000), maximum = TRUE, tol=1e-6)$maximum,
             optimize(prior.e_v, interval = c(-100, 100), maximum = TRUE, tol=1e-6)$maximum,
             optimize(prior.e_a_r, interval = c(-100, 100), maximum = TRUE, tol=1e-6)$maximum,
             optimize(prior.e_b_r, interval = c(-100, 100), maximum = TRUE, tol=1e-6)$maximum
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
result$summary.hyperpar$mean[4]
result$summary.hyperpar$mean[5]
result$summary.hyperpar$mean[6]
result$summary.hyperpar$mean[7]

# Create a layout with one column and six rows
par(mfrow = c(7, 1), mar = c(2, 2, 1, 1))
# Plot each function in a separate row
plot(exp(result$marginals.hyperpar$`Theta1 for idx`[1:43]),result$marginals.hyperpar$`Theta1 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "l_bg", col = "red", cex = 1.5)
plot(exp(result$marginals.hyperpar$`Theta2 for idx`[1:43]),result$marginals.hyperpar$`Theta2 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "l_a", col = "red", cex = 1.5)
plot(exp(result$marginals.hyperpar$`Theta3 for idx`[1:43]),result$marginals.hyperpar$`Theta3 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "l_b", col = "red", cex = 1.5)
plot((result$marginals.hyperpar$`Theta4 for idx`[1:43]),result$marginals.hyperpar$`Theta4 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "v_b_r", col = "red", cex = 1.5)
plot((result$marginals.hyperpar$`Theta5 for idx`[1:43]),result$marginals.hyperpar$`Theta5 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "e_v", col = "red", cex = 1.5)
plot((result$marginals.hyperpar$`Theta6 for idx`[1:43]),result$marginals.hyperpar$`Theta6 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "e_a_r", col = "red", cex = 1.5)
plot((result$marginals.hyperpar$`Theta7 for idx`[1:43]),result$marginals.hyperpar$`Theta7 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "e_b_r", col = "red", cex = 1.5)
# Reset the layout to the default (1x1)
par(mfrow = c(1, 1))



###################################
### Priors and post. evaluation ###
###################################

#posteriors to be saved in X/Y form
#l_bg
fx_l_bg = exp(result$marginals.hyperpar$`Theta1 for idx`[1:43])
fy_l_bg = result$marginals.hyperpar$`Theta1 for idx`[44:86]
#l_a
fx_l_a = exp(result$marginals.hyperpar$`Theta2 for idx`[1:43])
fy_l_a = result$marginals.hyperpar$`Theta2 for idx`[44:86]
#l_b
fx_l_b = exp(result$marginals.hyperpar$`Theta3 for idx`[1:43])
fy_l_b = result$marginals.hyperpar$`Theta3 for idx`[44:86]
#v_b_r
fx_v_b_r = result$marginals.hyperpar$`Theta4 for idx`[1:43]
fy_v_b_r = result$marginals.hyperpar$`Theta4 for idx`[44:86]
#e_v
fx_e_v = result$marginals.hyperpar$`Theta5 for idx`[1:43]
fy_e_v = result$marginals.hyperpar$`Theta5 for idx`[44:86]
#e_a_r
fx_e_a_r = result$marginals.hyperpar$`Theta6 for idx`[1:43]
fy_e_a_r = result$marginals.hyperpar$`Theta6 for idx`[44:86]
#e_b_r
fx_e_b_r = result$marginals.hyperpar$`Theta7 for idx`[1:43]
fy_e_b_r = result$marginals.hyperpar$`Theta7 for idx`[44:86]

#priors to be saved in X/Y form
#log priors extracted
prior.l_bg  <- function(x){ 
  return((three_component_model(cmd="prior.l_bg", feed_x=x))) }
prior.l_a <- function(x){
  return((three_component_model(cmd="prior.l_a",feed_x=x))) }
prior.l_b   <- function(x){
  return((three_component_model(cmd="prior.l_b",  feed_x=x))) }
prior.v_b_r <- function(x){
  return((three_component_model(cmd="prior.v_b_r",feed_x=x))) }
prior.e_v   <- function(x){
  return((three_component_model(cmd="prior.e_v",  feed_x=x))) }
prior.e_a_r   <- function(x){
  return((three_component_model(cmd="prior.e_a_r",  feed_x=x))) }
prior.e_b_r   <- function(x){
  return((three_component_model(cmd="prior.e_b_r",  feed_x=x))) }
#priors evaluated
#l_bg
p_l_bg_max = optimize(prior.l_bg, interval = c(0, 1), maximum = TRUE, tol=1e-9)$maximum
px_l_bg = seq(p_l_bg_max/1000, p_l_bg_max*100, length.out = 100000)
py_l_bg = exp(prior.l_bg(px_l_bg))
#l_a
p_l_a_max = optimize(prior.l_a, interval = c(0, 1), maximum = TRUE, tol=1e-9)$maximum
px_l_a = seq(p_l_a_max/1000, p_l_a_max*100, length.out = 100000)
py_l_a = exp(prior.l_a(px_l_a))
#l_b
p_l_b_max = optimize(prior.l_b, interval = c(0, 1), maximum = TRUE, tol=1e-9)$maximum
px_l_b = seq(p_l_b_max/1000, p_l_b_max*100, length.out = 100000)
py_l_b = exp(prior.l_b(px_l_b))
#v_b_r
p_l_v_b_r_max = optimize(prior.v_b_r, interval = c(0, 10000), maximum = TRUE, tol=1e-9)$maximum
px_v_b_r = seq(p_l_v_b_r_max/100, p_l_v_b_r_max*10, length.out = 10000)
py_v_b_r = exp(prior.v_b_r(px_v_b_r))
#e_v
p_e_v_max = optimize(prior.e_v, interval = c(0, 10), maximum = TRUE, tol=1e-9)$maximum
px_e_v = seq(p_e_v_max/5, p_e_v_max*5, length.out = 10000)
py_e_v = exp(prior.e_v(px_e_v))
#e_a_r
p_e_a_r_max = optimize(prior.e_a_r, interval = c(-10, 0), maximum = TRUE, tol=1e-9)$maximum
px_e_a_r = seq(p_e_a_r_max*5, p_e_a_r_max/5, length.out = 10000)
py_e_a_r = exp(prior.e_a_r(px_e_a_r))
#e_b_r
p_e_b_r_max = optimize(prior.e_b_r, interval = c(-10, 0), maximum = TRUE, tol=1e-9)$maximum
px_e_b_r = seq(p_e_b_r_max*5, p_e_b_r_max/5, length.out = 10000)
py_e_b_r = exp(prior.e_b_r(px_e_b_r))


###################################
######## Sample posterior  ########
###################################

s = inla.hyperpar.sample(1000000, result)

sample_l_bg  = exp(s[,1])
sample_l_a   = exp(s[,2])
sample_l_b   = exp(s[,3])
sample_v_b_r =     s[,4]
sample_e_v   =     s[,5]
sample_e_a_r =     s[,6]
sample_e_b_r =     s[,7]

hexbinplot(sample_v_b_r~sample_l_a, 
           data=data.frame(sample_v_b_r,sample_l_a), 
           colramp=colorRampPalette(c("grey", "yellow")),
           main="joint histogram v_b_r, l_a" ,  
           xlab="l_a", 
           ylab="v_b_r" ,
           panel=function(x, y, ...)
           {
             panel.hexbinplot(x, y, ...)
             panel.abline(v=c(mean(sample_l_a)), h=c(mean(sample_v_b_r)), col="black", lwd=2, lty=3)
           }
)


###################################
#### Save everything relevant #####
###################################

model_definition <- deparse(three_component_model)

current_time <- Sys.time()
formatted_time <- format(current_time, "%Y%m%d%H%M%S")

save(sample_l_bg,       #sampled posterior
     sample_l_a, 
     sample_l_b, 
     sample_v_b_r, 
     sample_e_v, 
     sample_e_a_r,
     sample_e_b_r,
     fx_l_bg,           #evaluated posterior
     fy_l_bg,
     fx_l_a,
     fy_l_a,
     fx_l_b,
     fy_l_b,
     fx_v_b_r,
     fy_v_b_r,
     fx_e_v,
     fy_e_v,
     fx_e_a_r,
     fy_e_a_r,
     fx_e_b_r,
     fy_e_b_r,
     px_l_bg,           #evaluated prior
     py_l_bg,
     px_l_a,
     py_l_a,
     px_l_b,
     py_l_b,
     px_v_b_r,
     py_v_b_r,
     px_e_v,
     py_e_v,
     px_e_a_r,
     py_e_a_r,
     px_e_b_r,
     py_e_b_r,
     model_definition,  #the model definition
     mydata,
     file = paste("998_generated\\inla\\sample_",formatted_time,".RData",
                  sep = ""))






