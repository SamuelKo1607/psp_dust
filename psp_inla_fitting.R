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
                                          "prior.v_b_r",
                                          "prior.e_a_v",
                                          "prior.e_b_v",
                                          "prior.e_a_r",
                                          "prior.e_b_r"), 
                                  theta=NULL, feed_x=NULL, feed_c=NULL, feed_h=NULL){

  envir <- parent.env(environment())
  prec.high = exp(15)
  
  prior.l_a <- function(l_a=feed_x){
    return(dgamma(l_a, shape = 2,    scale = 1e-4, log=TRUE))
  }
  prior.l_b <- function(l_b=feed_x){
    return(dgamma(l_b,   shape = 2,    scale = 1e-4, log=TRUE))
  }
  prior.v_b_r <- function(v_b_r=feed_x){
    return(dnorm(v_b_r,  mean  = 50,    sd   = 1,    log=TRUE))
  }
  prior.e_a_v <- function(e_a_v=feed_x){
    return(dnorm(e_a_v,    mean  = 2.2,   sd   = 0.05, log=TRUE))
  }
  prior.e_b_v <- function(e_b_v=feed_x){
    return(dnorm(e_b_v,    mean  = 2.2,   sd   = 0.01, log=TRUE))
  }
  prior.e_a_r <- function(e_a_r=feed_x){
    return(dnorm(e_a_r,    mean  = -1.3, sd   = 0.001, log=TRUE))
  }
  prior.e_b_r <- function(e_b_r=feed_x){
    return(dnorm(e_b_r,    mean  = -1.8, sd   = 0.001, log=TRUE))
  }
  
  rate <- function(#covariates
                   v_sc_r = feed_c[1], 
                   v_sc_t = feed_c[2], 
                   r_sc =   feed_c[3], 
                   v_sc_x = feed_c[4], 
                   v_sc_y = feed_c[5], 
                   v_sc_z = feed_c[6], 
                   area =   feed_c[7],  
                   #hyperparam
                   l_a =   feed_h[1], 
                   l_b =   feed_h[2], 
                   v_b_r = feed_h[3], 
                   e_a_v = feed_h[4], 
                   e_b_v = feed_h[5], 
                   e_a_r = feed_h[6], 
                   e_b_r = feed_h[7],      
                   #bound dust parameters
 
                   #beta met. parameters
                   v_b_a=9,
                   #stationary Earth
                   v_earth_a=0){            
    
    deg2rad <- function(deg){
      rad = deg/180*pi
      return(rad)
    }

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
    L_b = l_b * (v_factor)^e_b_v * (r_factor)^e_b_r 
    
    #bound dust contribution
    ksi = -2 - e_a_r
    r_factor = r_sc/1
    v_a_a = 29.8*(r_factor^(-1))
    v_factor = ( (
      ( v_sc_r )^2 
      + ( v_sc_t - v_a_a )^2
    )^0.5 
    ) / abs( v_earth_a - v_a_a )
    L_a = l_a * (v_factor)^e_a_v * (r_factor)^e_a_r
    
    #normalization to hourly rate, while L_i are in m^-2 s^-1
    hourly_rate = 3600 * area * ( L_b + L_a )
    return(hourly_rate)
  }
  
  
  
  interpret.theta <- function(){
    return(list(l_a   = exp(theta[1L]),
                l_b   = exp(theta[2L]),
                v_b_r = theta[3L],
                e_a_v = theta[4L], 
                e_b_v = theta[5L],
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
                     par$l_a, 
                     par$l_b, 
                     par$v_b_r, 
                     par$e_a_v, 
                     par$e_b_v, 
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
    val <- (prior.l_a(      par$l_a)     + theta[1L] +
            prior.l_b(      par$l_b)     + theta[2L] +
            prior.v_b_r(    par$v_b_r)   +
            prior.e_a_v(    par$e_a_v)   + 
            prior.e_b_v(    par$e_b_v)   + 
            prior.e_a_r(    par$e_a_r)   +
            prior.e_b_r(    par$e_b_r)
           )
  
    return(val)
  }
  
  # Initial values of theta
  initial <- function(){
    #initial values set to the maxima a priori
    return(c(log(optimize(prior.l_a, interval = c(0, 1e-2), maximum = TRUE, tol=1e-9)$maximum),
             log(optimize(prior.l_b, interval = c(0, 1e-2), maximum = TRUE, tol=1e-9)$maximum),
             optimize(prior.v_b_r, interval = c(  0, 1000), maximum = TRUE, tol=1e-6)$maximum,
             optimize(prior.e_a_v, interval = c(-100, 100), maximum = TRUE, tol=1e-6)$maximum,
             optimize(prior.e_b_v, interval = c(-100, 100), maximum = TRUE, tol=1e-6)$maximum,
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

#filterinng to far-from the Sun only
mydata_far <- subset(mydata, r > 0.25 & sc_id == 2 & exposure > 1e-6)


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
              data = mydata_far, family = "poisson", E = exposure, 
              control.compute = list(cpo=TRUE, dic=TRUE, config = TRUE),
              safe = TRUE, verbose = TRUE)

summary(result)

par(mfrow = c(1, 1))
hist(result$cpo$pit)     # ok 
max(result$cpo$failure)       # also OK
#pit = result$cpo$pit
#save(pit, file = "998_generated\\inla\\pit.RData")

#plotting histogram
span = round(max(abs(mydata_far$flux/mydata_far$exposure-result$summary.fitted.values$mean),40)+0.5)
hist(mydata_far$flux/mydata_far$exposure-result$summary.fitted.values$mean,
     breaks=c(-span:span),
     main="")
mtext(paste("residuals histogram, stdev = ",
            as.character(sqrt(var(mydata_far$flux/mydata_far$exposure-result$summary.fitted.values$mean))),
            ", log(mlik) = ",
            result$mlik[1]), side=3)
  
# Posterior means of the hyperparameters
l_a.mean = inla.emarginal(function(x) exp(x), result$marginals.hyperpar$`Theta1 for idx`)
l_b.mean = inla.emarginal(function(x) exp(x), result$marginals.hyperpar$`Theta2 for idx`)
v_b_r.mean = result$summary.hyperpar$mean[3]
e_a_v.mean = result$summary.hyperpar$mean[4]
e_b_v.mean = result$summary.hyperpar$mean[5]
e_a_r.mean = result$summary.hyperpar$mean[6]
e_b_r.mean = result$summary.hyperpar$mean[7]

# Create a layout with one column and seven rows
par(mfrow = c(7, 1), mar = c(2, 2, 1, 1))
# Plot each function in a separate row
plot(exp(result$marginals.hyperpar$`Theta1 for idx`[1:43]),result$marginals.hyperpar$`Theta1 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "l_a", col = "red", cex = 1.5)
plot(exp(result$marginals.hyperpar$`Theta2 for idx`[1:43]),result$marginals.hyperpar$`Theta2 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "l_b", col = "red", cex = 1.5)
plot((result$marginals.hyperpar$`Theta3 for idx`[1:43]),result$marginals.hyperpar$`Theta3 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "v_b_r", col = "red", cex = 1.5)
plot((result$marginals.hyperpar$`Theta4 for idx`[1:43]),result$marginals.hyperpar$`Theta4 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "e_a_v", col = "red", cex = 1.5)
plot((result$marginals.hyperpar$`Theta5 for idx`[1:43]),result$marginals.hyperpar$`Theta5 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "e_b_v", col = "red", cex = 1.5)
plot((result$marginals.hyperpar$`Theta6 for idx`[1:43]),result$marginals.hyperpar$`Theta6 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "e_a_r", col = "red", cex = 1.5)
plot((result$marginals.hyperpar$`Theta7 for idx`[1:43]),result$marginals.hyperpar$`Theta7 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "e_b_r", col = "red", cex = 1.5)
# Reset the layout to the default (1x1)
par(mfrow = c(1, 1))


###################################
####### Rate decomposition ########
###################################

len = length(mydata_far$Julian.date)
total_rate = numeric(len)
bound_rate = numeric(len)
beta_rate = numeric(len)

samples = 100
s = inla.hyperpar.sample(samples, result)
sample_l_a   = exp(s[,1])
sample_l_b   = exp(s[,2])
sample_v_b_r =     s[,3]
sample_e_a_v =     s[,4]
sample_e_b_v =     s[,5]
sample_e_a_r =     s[,6]
sample_e_b_r =     s[,7]

for (i in 1:len) {
  particuler_total_rates = numeric(samples)
  particuler_bound_rates = numeric(samples)
  particuler_beta_rates  = numeric(samples)
  for (j in 1:samples) {
    particuler_total_rates[j] <- three_component_model(cmd="rate", 
                                           feed_c = c(mydata_far$vr[i],
                                                      mydata_far$vt[i],
                                                      mydata_far$r[i],
                                                      mydata_far$vx[i],
                                                      mydata_far$vy[i],
                                                      mydata_far$vz[i],
                                                      mydata_far$area[i]),
                                           feed_h = c(sample_l_a[j],
                                                      sample_l_b[j],
                                                      sample_v_b_r[j],
                                                      sample_e_a_v[j],
                                                      sample_e_b_v[j],
                                                      sample_e_a_r[j],
                                                      sample_e_b_r[j]))
    particuler_bound_rates[j] <- three_component_model(cmd="rate", 
                                           feed_c = c(mydata_far$vr[i],
                                                      mydata_far$vt[i],
                                                      mydata_far$r[i],
                                                      mydata_far$vx[i],
                                                      mydata_far$vy[i],
                                                      mydata_far$vz[i],
                                                      mydata_far$area[i]),
                                           feed_h = c(sample_l_a[j],
                                                      0,
                                                      sample_v_b_r[j],
                                                      sample_e_a_v[j],
                                                      sample_e_b_v[j],
                                                      sample_e_a_r[j],
                                                      sample_e_b_r[j]))
    particuler_beta_rates[j] <- three_component_model(cmd="rate", 
                                           feed_c = c(mydata_far$vr[i],
                                                      mydata_far$vt[i],
                                                      mydata_far$r[i],
                                                      mydata_far$vx[i],
                                                      mydata_far$vy[i],
                                                      mydata_far$vz[i],
                                                      mydata_far$area[i]),
                                           feed_h = c(0,
                                                      sample_l_b[j],
                                                      sample_v_b_r[j],
                                                      sample_e_a_v[j],
                                                      sample_e_b_v[j],
                                                      sample_e_a_r[j],
                                                      sample_e_b_r[j]))

  }
  total_rate[i] <- mean(particuler_total_rates)
  bound_rate[i] <- mean(particuler_bound_rates)
  beta_rate[i]  <- mean(particuler_beta_rates)
}

par(mfrow = c(1, 1))
plot(mydata_far$flux/mydata_far$exposure, ylab="counts/E")
lines(result$summary.fitted.values$mean, col=2, lwd=3)
lines(bound_rate / total_rate * result$summary.fitted.values$mean, col=3, lwd=3)
lines(beta_rate / total_rate * result$summary.fitted.values$mean, col=4, lwd=3)
legend(0, 50, legend=c("total", "bound", "beta"),
       col = c(2, 3, 4),
       lty = c(1, 1, 1),
       lwd = c(3, 3, 3))

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






